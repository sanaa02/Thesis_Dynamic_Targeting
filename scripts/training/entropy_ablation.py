#!/usr/bin/env python3
"""
entropy_ablation.py  --  ALSAT-EO-1  IMP-06  Entropy Annealing Ablation
========================================================================
Controlled ablation to validate that entropy annealing improves training.

Three conditions (all else fixed):
  (a) fixed_low:   ent_coef = 0.05  (collapsed entropy value)
  (b) fixed_high:  ent_coef = 0.15  (high entropy, too much noise)
  (c) annealed:    ent_coef 0.15 → 0.01  (current setting)

3 seeds × 3 conditions at event_rate=1.0 ev/hr.

Metrics: final mean reward ± std, convergence timestep (first ep where
rolling-100 mean reward ≥ 90% of final), dynamic success rate evolution.

Usage
-----
    python scripts/training/entropy_ablation.py --seeds 42 123 456
    python scripts/training/entropy_ablation.py --quick
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

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
OUT_DIR   = os.path.join(ROOT, "results/ablation/entropy")

DEFAULT_SEEDS    = [42, 123, 456]
DEFAULT_EPISODES = 500
DEFAULT_EVAL_EPS = 30


@dataclass
class EntropyCondition:
    name:       str
    ent_coef:   float
    anneal:     bool
    ent_end:    Optional[float] = None
    description: str = ""


CONDITIONS = [
    EntropyCondition("fixed_low",  0.05,  False,  None,  "Fixed low entropy (collapse)"),
    EntropyCondition("fixed_high", 0.15,  False,  None,  "Fixed high entropy (too noisy)"),
    EntropyCondition("annealed",   0.15,  True,   0.01,  "Annealed 0.15 -> 0.01 (current)"),
]


def run_condition(
    cond: EntropyCondition,
    seed: int,
    n_episodes: int,
    quick: bool = False,
) -> dict:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import CallbackList
    from env_dynamic_factory import make_env, Config
    from env_alsat_debug import SIM_DURATION_S, SCHED_STEP_S

    if quick:
        n_episodes = 50

    steps_per_ep = int(SIM_DURATION_S / SCHED_STEP_S)
    total_steps  = n_episodes * steps_per_ep

    def _make():
        env = make_env(Config.DYN_MODIS, TARGETS, CLOUD,
                       event_rate=1.0, seed=seed, with_safety=False)
        return Monitor(env)

    vec = DummyVecEnv([_make])

    model = PPO(
        "MlpPolicy", vec,
        learning_rate=3e-4, n_steps=2048,
        batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
        ent_coef=cond.ent_coef, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=0, seed=seed, device="cpu",
    )

    callbacks = []
    if cond.anneal and cond.ent_end is not None:
        try:
            from callbacks import EntropyAnnealingCallback
            cb = EntropyAnnealingCallback(
                start_val=cond.ent_coef, end_val=cond.ent_end,
                total_timesteps=total_steps, verbose=0,
            )
            callbacks.append(cb)
        except ImportError:
            logger.warning("  EntropyAnnealingCallback not found — skipping anneal")

    ep_rewards: List[float] = []
    ep_dyn_suc: List[float] = []

    class _LogCb:
        """Minimal callback to record per-episode metrics."""
        def __init__(self):
            self.n_calls = 0
        def on_step(self):
            self.n_calls += 1
            return True
        def on_rollout_end(self):
            infos = vec.buf_infos if hasattr(vec, "buf_infos") else []
            for info in infos:
                ep_info = info.get("episode", None)
                if ep_info:
                    ep_rewards.append(ep_info["r"])
            return True

    t0 = time.time()
    model.learn(total_timesteps=total_steps,
                callback=CallbackList(callbacks) if callbacks else None,
                reset_num_timesteps=True, progress_bar=False)
    elapsed = time.time() - t0
    vec.close()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    eval_env = make_env(Config.DYN_MODIS, TARGETS, CLOUD,
                        event_rate=1.0, seed=seed + 9999, with_safety=False)
    rewards, dyn_rates = [], []
    for ep in range(DEFAULT_EVAL_EPS):
        obs, _ = eval_env.reset(seed=seed + 9999 + ep)
        done, ep_r = False, 0.0
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = eval_env.step(int(act))
            ep_r += r
            done  = term or trunc
        rewards.append(ep_r)
        m = info.get("episode_metrics", {})
        n_dyn = max(1, m.get("n_dyn_detected", 1))
        n_ok  = m.get("n_dyn_imaged", 0)
        dyn_rates.append(n_ok / n_dyn)
    eval_env.close()

    out = {
        "condition":    cond.name,
        "ent_coef":     cond.ent_coef,
        "anneal":       cond.anneal,
        "ent_end":      cond.ent_end,
        "seed":         seed,
        "mean_reward":  float(np.mean(rewards)),
        "std_reward":   float(np.std(rewards)),
        "mean_dyn_suc": float(np.mean(dyn_rates)),
        "elapsed_min":  round(elapsed / 60, 2),
        "n_episodes":   n_episodes,
    }

    # Convergence: first rolling-100 average ≥ 90% of final
    if ep_rewards:
        final_r = np.mean(rewards)
        thresh  = 0.9 * final_r
        running = []
        conv_ep  = None
        for i, r in enumerate(ep_rewards):
            running.append(r)
            if len(running) >= 10:
                if np.mean(running[-10:]) >= thresh:
                    conv_ep = i
                    break
        out["convergence_ep"] = conv_ep
    return out


def run_ablation(
    seeds:      list = DEFAULT_SEEDS,
    n_episodes: int  = DEFAULT_EPISODES,
    quick:      bool = False,
) -> list[dict]:
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []

    total = len(CONDITIONS) * len(seeds)
    done  = 0

    for cond in CONDITIONS:
        for seed in seeds:
            done += 1
            logger.info(
                f"\n[{done}/{total}] condition={cond.name}  seed={seed}  "
                f"{'(quick)' if quick else ''}"
            )
            try:
                r = run_condition(cond, seed, n_episodes, quick)
                results.append(r)
                logger.info(
                    f"  reward={r['mean_reward']:+.3f}  "
                    f"dyn_suc={r['mean_dyn_suc']:.1%}  "
                    f"conv_ep={r.get('convergence_ep')}"
                )
            except Exception as exc:
                logger.error(f"  FAILED: {exc}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_json = os.path.join(OUT_DIR, "entropy_ablation_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=float)
    logger.info(f"\nResults saved → {out_json}")

    _print_table(results)
    return results


def _print_table(results: list[dict]) -> None:
    from collections import defaultdict
    agg: dict = defaultdict(list)
    for r in results:
        agg[r["condition"]].append(r)

    print("\n" + "=" * 70)
    print(f"{'Condition':<15} {'Reward':>12} {'±':>8} {'DynSuc':>10} "
          f"{'Conv(ep)':>10}")
    print("-" * 70)
    for cond_name, rows in sorted(agg.items()):
        rs  = [r["mean_reward"]  for r in rows]
        ds  = [r["mean_dyn_suc"] for r in rows]
        cep = [r["convergence_ep"] for r in rows
               if r.get("convergence_ep") is not None]
        cep_str = f"{np.mean(cep):.0f}" if cep else "N/A"
        print(f"  {cond_name:<13} {np.mean(rs):>+12.4f} {np.std(rs):>8.4f} "
              f"{np.mean(ds):>10.1%} {cep_str:>10}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",    type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--quick",    action="store_true")
    args = parser.parse_args()
    run_ablation(seeds=args.seeds, n_episodes=args.episodes, quick=args.quick)