#!/usr/bin/env python3
"""
cloud_uncertainty_ablation.py  --  ALSAT-EO-1  IMP-07  Cloud Uncertainty Ablation
===================================================================================
Runs the three conditions for the cloud-uncertainty ablation study:

  (a) standard: CNN forecast with sigma=0.05 (normal training + evaluation)
  (b) oracle:   cloud_cover_forecast = cloud_cover (sigma=0, trained + tested)
  (c) test_oracle: standard policy (trained with noise), oracle at test time

Comparing (a) vs (c) reveals how much the policy suffers from cloud uncertainty.
Comparing (a) vs (b) reveals whether training with noise is beneficial (regularization).

Usage
-----
    python scripts/training/cloud_uncertainty_ablation.py --seeds 42 123 456
    python scripts/training/cloud_uncertainty_ablation.py --quick
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import path_setup  # noqa

ROOT = path_setup.root_path()
for _d in ["scripts/core", "scripts/training", "scripts/wrappers", "scripts"]:
    _p = os.path.join(ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

TARGETS   = os.path.join(ROOT, "config/targets/global_45_targets.json")
CLOUD     = os.path.join(ROOT, "config/cloud_reality/global_45_clouds.json")
MODELS    = os.path.join(ROOT, "models")
OUT_DIR   = os.path.join(ROOT, "results/ablation/cloud_uncertainty")

DEFAULT_SEEDS    = [42, 123, 456]
DEFAULT_EPISODES = 200
EVAL_EPISODES    = 30


def _make_env(oracle_cloud: bool, seed: int, event_rate: float = 1.0):
    from env_dynamic_factory import make_env, Config
    from oracle_cloud_wrapper import OracleCloudWrapper

    env = make_env(Config.DYN_MODIS, TARGETS, CLOUD,
                   event_rate=event_rate, seed=seed, with_safety=False)
    return OracleCloudWrapper(env, oracle_cloud=oracle_cloud)


def _train_policy(oracle_cloud: bool, seed: int, n_episodes: int) -> object:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from env_alsat_debug import SIM_DURATION_S, SCHED_STEP_S

    steps_per_ep = int(SIM_DURATION_S / SCHED_STEP_S)
    total_steps  = n_episodes * steps_per_ep

    def _make():
        return Monitor(_make_env(oracle_cloud=oracle_cloud, seed=seed))

    vec = DummyVecEnv([_make])
    model = PPO(
        "MlpPolicy", vec,
        learning_rate=3e-4, n_steps=2048, batch_size=64,
        n_epochs=10, gamma=0.99, ent_coef=0.05, verbose=0,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        seed=seed, device="cpu",
    )
    model.learn(total_timesteps=total_steps, reset_num_timesteps=True)
    vec.close()
    return model


def _evaluate_policy(model, oracle_cloud_eval: bool, seed: int) -> dict:
    rewards, dyn_rates, cf_rates = [], [], []
    for ep in range(EVAL_EPISODES):
        env = _make_env(oracle_cloud=oracle_cloud_eval, seed=seed + 9999 + ep)
        obs, _ = env.reset(seed=seed + 9999 + ep)
        done, ep_r = False, 0.0
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(act))
            ep_r += r; done = term or trunc
        env.close()
        rewards.append(ep_r)
        m = info.get("episode_metrics", {})
        n_dyn = max(1, m.get("n_dyn_detected", 1))
        n_ok  = m.get("n_dyn_imaged", 0)
        cf    = m.get("n_cloud_free", 0)
        n_img = max(1, m.get("n_imaged", 1))
        dyn_rates.append(n_ok / n_dyn)
        cf_rates.append(cf / n_img)
    return {
        "mean_reward":  float(np.mean(rewards)),
        "std_reward":   float(np.std(rewards)),
        "mean_dyn_suc": float(np.mean(dyn_rates)),
        "mean_cf_rate": float(np.mean(cf_rates)),
    }


def run_ablation(
    seeds:      list = DEFAULT_SEEDS,
    n_episodes: int  = DEFAULT_EPISODES,
    quick:      bool = False,
) -> list[dict]:
    os.makedirs(OUT_DIR, exist_ok=True)
    if quick:
        seeds = [seeds[0]]
        n_episodes = 50

    all_results = []

    for seed in seeds:
        logger.info(f"\n=== Seed {seed} ===")

        # (a) Standard: noisy CNN train + noisy eval
        logger.info("  (a) Training standard policy (CNN noise)...")
        t0 = time.time()
        model_std = _train_policy(oracle_cloud=False, seed=seed,
                                  n_episodes=n_episodes)
        train_t = time.time() - t0

        eval_a = _evaluate_policy(model_std, oracle_cloud_eval=False, seed=seed)
        eval_a.update({"condition": "standard_train_std_eval",
                       "seed": seed, "train_time_min": round(train_t/60, 2)})
        all_results.append(eval_a)
        logger.info(f"    (a) reward={eval_a['mean_reward']:+.3f}  "
                    f"dyn_suc={eval_a['mean_dyn_suc']:.1%}")

        # (c) Test-time oracle: standard policy, perfect cloud at test
        eval_c = _evaluate_policy(model_std, oracle_cloud_eval=True, seed=seed)
        eval_c.update({"condition": "standard_train_oracle_eval",
                       "seed": seed, "train_time_min": round(train_t/60, 2)})
        all_results.append(eval_c)
        logger.info(f"    (c) reward={eval_c['mean_reward']:+.3f}  "
                    f"dyn_suc={eval_c['mean_dyn_suc']:.1%}")

        # (b) Oracle: train AND eval with perfect cloud knowledge
        logger.info("  (b) Training oracle policy (perfect cloud knowledge)...")
        t1 = time.time()
        model_oracle = _train_policy(oracle_cloud=True, seed=seed,
                                     n_episodes=n_episodes)
        train_t2 = time.time() - t1

        eval_b = _evaluate_policy(model_oracle, oracle_cloud_eval=True, seed=seed)
        eval_b.update({"condition": "oracle_train_oracle_eval",
                       "seed": seed, "train_time_min": round(train_t2/60, 2)})
        all_results.append(eval_b)
        logger.info(f"    (b) reward={eval_b['mean_reward']:+.3f}  "
                    f"dyn_suc={eval_b['mean_dyn_suc']:.1%}")

        # Save models
        model_std.save(os.path.join(MODELS, f"ppo_cloud_std_s{seed}.zip"))
        model_oracle.save(os.path.join(MODELS, f"ppo_cloud_oracle_s{seed}.zip"))

    out_json = os.path.join(OUT_DIR, "cloud_uncertainty_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    logger.info(f"\nResults saved → {out_json}")

    _print_summary(all_results)
    return all_results


def _print_summary(results: list[dict]) -> None:
    from collections import defaultdict
    agg: dict = defaultdict(list)
    for r in results:
        agg[r["condition"]].append(r)

    print("\n" + "=" * 70)
    print("CLOUD UNCERTAINTY ABLATION SUMMARY")
    print(f"{'Condition':<35} {'Reward':>10} {'±':>8} {'DynSuc':>8}")
    print("-" * 70)
    for cond, rows in sorted(agg.items()):
        rs = [r["mean_reward"] for r in rows]
        ds = [r["mean_dyn_suc"] for r in rows]
        print(f"  {cond:<33} {np.mean(rs):>+10.4f} {np.std(rs):>8.4f} "
              f"{np.mean(ds):>8.1%}")
    print("=" * 70)

    std_rows    = agg.get("standard_train_std_eval", [])
    oracle_rows = agg.get("standard_train_oracle_eval", [])
    if std_rows and oracle_rows:
        delta = (np.mean([r["mean_reward"] for r in oracle_rows]) -
                 np.mean([r["mean_reward"] for r in std_rows]))
        print(f"\n  Δ reward (oracle_eval - std_eval) = {delta:+.4f}")
        print("  (positive = policy is hurt by cloud uncertainty at test time)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",    type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--quick",    action="store_true")
    args = parser.parse_args()
    run_ablation(seeds=args.seeds, n_episodes=args.episodes, quick=args.quick)