#!/usr/bin/env python3
"""
smdp_ablation.py  --  ALSAT-EO-1  IMP-08  SMDP vs Flat MDP Ablation
====================================================================
Validates IMP-08: "SMDP Discounting vs Standard MDP Fixed Timestep"

Trains two policies:
  A) SMDP  env  — proper gamma^(tau/STEP_REF_S) discounting (current)
  B) Flat MDP   — reward/n_sub per sub-step, standard gamma^1

Compares final performance. If SMDP outperforms flat, confirms SMDP value.

Usage
-----
    python scripts/training/smdp_ablation.py --seeds 42 123 456
    python scripts/training/smdp_ablation.py --quick
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
OUT_DIR   = os.path.join(ROOT, "results/ablation/smdp_vs_flat")

DEFAULT_SEEDS    = [42, 123, 456]
DEFAULT_EPISODES = 500
EVAL_EPISODES    = 30


def _train_and_eval(
    use_flat_mdp: bool,
    seed:         int,
    n_episodes:   int,
) -> dict:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from env_dynamic_factory import make_env, Config
    from env_alsat_debug import SIM_DURATION_S, SCHED_STEP_S

    label        = "flat_mdp" if use_flat_mdp else "smdp"
    steps_per_ep = int(SIM_DURATION_S / SCHED_STEP_S)
    total_steps  = n_episodes * steps_per_ep

    def _make():
        env = make_env(Config.DYN_MODIS, TARGETS, CLOUD,
                       event_rate=1.0, seed=seed, with_safety=False)
        if use_flat_mdp:
            try:
                from flat_mdp_wrapper import FlatMDPWrapper
                env = FlatMDPWrapper(env, gamma=0.99, redistribute=True)
            except ImportError:
                logger.warning("  FlatMDPWrapper not available, using raw SMDP env")
        return Monitor(env)

    vec = DummyVecEnv([_make])

    model = PPO(
        "MlpPolicy", vec,
        learning_rate=3e-4, n_steps=2048,
        batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
        ent_coef=0.05, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=0, seed=seed, device="cpu",
    )

    t0 = time.time()
    model.learn(total_timesteps=total_steps, reset_num_timesteps=True,
                progress_bar=False)
    elapsed = time.time() - t0
    vec.close()

    # Save
    out_path = os.path.join(MODELS, f"ppo_{label}_s{seed}.zip")
    model.save(out_path)

    # Evaluate on SMDP env (always — so comparison is fair)
    eval_env = make_env(Config.DYN_MODIS, TARGETS, CLOUD,
                        event_rate=1.0, seed=seed + 9999, with_safety=False)
    rewards, dyn_rates, cf_rates = [], [], []
    for ep in range(EVAL_EPISODES):
        obs, _ = eval_env.reset(seed=seed + 9999 + ep)
        done, ep_r = False, 0.0
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = eval_env.step(int(act))
            ep_r += r; done = term or trunc
        rewards.append(ep_r)
        m = info.get("episode_metrics", {})
        n_dyn = max(1, m.get("n_dyn_detected", 1))
        n_ok  = m.get("n_dyn_imaged", 0)
        cf    = m.get("n_cloud_free", 0)
        n_img = max(1, m.get("n_imaged", 1))
        dyn_rates.append(n_ok / n_dyn)
        cf_rates.append(cf / n_img)
    eval_env.close()

    result = {
        "label":        label,
        "use_flat_mdp": use_flat_mdp,
        "seed":         seed,
        "mean_reward":  float(np.mean(rewards)),
        "std_reward":   float(np.std(rewards)),
        "mean_dyn_suc": float(np.mean(dyn_rates)),
        "mean_cf_rate": float(np.mean(cf_rates)),
        "elapsed_min":  round(elapsed / 60, 2),
        "model_path":   out_path,
    }
    logger.info(
        f"  [{label}|s{seed}] reward={result['mean_reward']:+.3f}  "
        f"dyn_suc={result['mean_dyn_suc']:.1%}  elapsed={result['elapsed_min']:.1f}min"
    )
    return result


def run_ablation(
    seeds:      list = DEFAULT_SEEDS,
    n_episodes: int  = DEFAULT_EPISODES,
    quick:      bool = False,
) -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    if quick:
        seeds = [seeds[0]]
        n_episodes = 50

    smdp_results = []
    flat_results = []

    for seed in seeds:
        logger.info(f"\n[SMDP|s{seed}] Training with SMDP discounting...")
        smdp_results.append(_train_and_eval(False, seed, n_episodes))

        logger.info(f"\n[Flat|s{seed}] Training with Flat MDP baseline...")
        flat_results.append(_train_and_eval(True, seed, n_episodes))

    # Aggregate
    def _agg(rows):
        return {
            "mean_reward":   float(np.mean([r["mean_reward"]  for r in rows])),
            "std_reward":    float(np.std( [r["mean_reward"]  for r in rows])),
            "mean_dyn_suc":  float(np.mean([r["mean_dyn_suc"] for r in rows])),
        }

    smdp_agg = _agg(smdp_results)
    flat_agg = _agg(flat_results)

    summary = {
        "smdp": smdp_agg,
        "flat": flat_agg,
        "delta_reward":   smdp_agg["mean_reward"]  - flat_agg["mean_reward"],
        "delta_dyn_suc":  smdp_agg["mean_dyn_suc"] - flat_agg["mean_dyn_suc"],
        "all_results": smdp_results + flat_results,
    }

    out_json = os.path.join(OUT_DIR, "smdp_vs_flat_results.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    logger.info(f"\nResults saved → {out_json}")

    print("\n" + "=" * 60)
    print("SMDP vs FLAT MDP ABLATION")
    print(f"  SMDP:   reward={smdp_agg['mean_reward']:+.4f}  dyn_suc={smdp_agg['mean_dyn_suc']:.1%}")
    print(f"  Flat:   reward={flat_agg['mean_reward']:+.4f}  dyn_suc={flat_agg['mean_dyn_suc']:.1%}")
    print(f"  Δ:      reward={summary['delta_reward']:+.4f}  dyn_suc={summary['delta_dyn_suc']:+.4f}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",    type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--quick",    action="store_true")
    args = parser.parse_args()
    run_ablation(seeds=args.seeds, n_episodes=args.episodes, quick=args.quick)