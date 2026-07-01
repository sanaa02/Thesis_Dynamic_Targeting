#!/usr/bin/env python3
"""
evaluate_real_data.py  --  ALSAT-EO-1  IMP-13  Real Data Evaluation
====================================================================
Evaluates a trained PPO policy on the real-data environment
(RealDataWrapper with TLE + FIRMS/GDACS + ERA5 + MODIS patches).

Tests the "real world transfer gap": how much does performance drop
when transitioning from synthetic to real data?

Usage
-----
    python scripts/evaluation/evaluate_real_data.py --model models/ppo_full_system_s42.zip
    python scripts/evaluation/evaluate_real_data.py --model ppo.zip --episodes 30
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

TARGETS  = os.path.join(ROOT, "config/targets/global_45_targets.json")
CLOUD    = os.path.join(ROOT, "config/cloud_reality/global_45_clouds.json")
OUT_DIR  = os.path.join(ROOT, "results/evaluation/real_data")

DEFAULT_SEEDS    = [42, 123, 456]
DEFAULT_EPISODES = 30


def _make_real_env(seed: int):
    from env_dynamic_factory import make_env, Config
    from env_alsat_real import RealDataWrapper, RealDataConfig

    env = make_env(Config.DYN_MODIS, TARGETS, CLOUD,
                   event_rate=1.0, seed=seed, with_safety=False)

    cfg = RealDataConfig.auto(root=ROOT, verbose=False)
    env = RealDataWrapper(env, cfg)
    return env


def _make_synthetic_env(seed: int):
    from env_dynamic_factory import make_env, Config
    return make_env(Config.DYN_MODIS, TARGETS, CLOUD,
                    event_rate=1.0, seed=seed, with_safety=False)


def _eval_policy(model, make_fn, seed, n_episodes, label):
    rewards, dyn_rates, cf_rates = [], [], []
    t_start = time.time()

    for ep in range(n_episodes):
        env = make_fn(seed=seed + ep)
        obs, _ = env.reset(seed=seed + ep)
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
        if (ep + 1) % 5 == 0:
            logger.info(f"  [{label}] ep {ep+1}/{n_episodes}  "
                        f"running avg={np.mean(rewards):+.3f}")

    elapsed = time.time() - t_start
    return {
        "label":        label,
        "n_episodes":   n_episodes,
        "mean_reward":  float(np.mean(rewards)),
        "std_reward":   float(np.std(rewards)),
        "mean_dyn_suc": float(np.mean(dyn_rates)),
        "mean_cf_rate": float(np.mean(cf_rates)),
        "elapsed_min":  round(elapsed / 60, 2),
    }


def run_real_data_eval(
    model_path: str,
    seeds: list = DEFAULT_SEEDS,
    n_episodes: int = DEFAULT_EPISODES,
) -> dict:
    from stable_baselines3 import PPO

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load model
    env_stub = _make_synthetic_env(seed=42)
    model = PPO.load(model_path, env=env_stub)
    env_stub.close()
    logger.info(f"Loaded model: {model_path}")

    all_syn, all_real = [], []

    for seed in seeds:
        logger.info(f"\n=== Seed {seed} ===")

        # Synthetic evaluation
        logger.info("  Evaluating on synthetic env...")
        syn = _eval_policy(model, _make_synthetic_env, seed,
                           n_episodes, "synthetic")
        all_syn.append(syn)
        logger.info(f"    synthetic: reward={syn['mean_reward']:+.3f}  "
                    f"dyn_suc={syn['mean_dyn_suc']:.1%}")

        # Real data evaluation
        logger.info("  Evaluating on real-data env...")
        try:
            real = _eval_policy(model, _make_real_env, seed,
                                n_episodes, "real_data")
        except Exception as exc:
            logger.warning(f"  Real data eval failed: {exc}. Using synthetic.")
            real = syn.copy()
            real["label"] = "real_data_FAILED"

        all_real.append(real)
        logger.info(f"    real_data: reward={real['mean_reward']:+.3f}  "
                    f"dyn_suc={real['mean_dyn_suc']:.1%}")

    def _agg(rows):
        rs = [r["mean_reward"] for r in rows]
        ds = [r["mean_dyn_suc"] for r in rows]
        return {"mean_reward": float(np.mean(rs)), "std": float(np.std(rs)),
                "mean_dyn_suc": float(np.mean(ds))}

    syn_agg  = _agg(all_syn)
    real_agg = _agg(all_real)
    gap      = syn_agg["mean_reward"] - real_agg["mean_reward"]

    result = {
        "model_path":   model_path,
        "seeds":        seeds,
        "n_episodes":   n_episodes,
        "synthetic":    syn_agg,
        "real_data":    real_agg,
        "transfer_gap": round(gap, 4),
        "all_results":  all_syn + all_real,
    }

    out_json = os.path.join(OUT_DIR, "real_data_eval.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, default=float)
    logger.info(f"\nResults saved → {out_json}")

    print("\n" + "=" * 60)
    print("REAL DATA EVALUATION SUMMARY")
    print(f"  Synthetic: reward={syn_agg['mean_reward']:+.4f}  "
          f"dyn_suc={syn_agg['mean_dyn_suc']:.1%}")
    print(f"  Real data: reward={real_agg['mean_reward']:+.4f}  "
          f"dyn_suc={real_agg['mean_dyn_suc']:.1%}")
    print(f"  Transfer gap: {gap:+.4f}  "
          f"({'within 5% threshold' if abs(gap) <= 0.05 * abs(syn_agg['mean_reward']) else 'EXCEEDS 5%'})")
    print("=" * 60)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    required=True)
    parser.add_argument("--seeds",    type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    args = parser.parse_args()
    run_real_data_eval(args.model, args.seeds, args.episodes)
