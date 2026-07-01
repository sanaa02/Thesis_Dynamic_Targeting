#!/usr/bin/env python3
"""
fixed_curriculum.py  --  ALSAT-EO-1  IMP-11  Fixed Curriculum n_steps
======================================================================
IMP-11 fix: change n_steps in each curriculum stage to match the
episode length (144 steps for a 2-day episode at BASE_STEP_S=1200s).

Problem: n_steps=576 (default 2048) accumulates 4 episodes before a
gradient step, making curriculum 4× less sample-efficient than intended.

Fix: set n_steps = steps_per_episode = SIM_DURATION_S / SCHED_STEP_S = 144
so each gradient update corresponds to exactly one episode of experience.

Also corrects curriculum schedule:
  static_clear   (50 eps)  → static only, 0.0 ev/hr
  static_clouds  (50 eps)  → clouds, 0.0 ev/hr
  dynamic_sparse (100 eps) → 0.5 ev/hr
  dynamic_dense  (remaining) → 2.0 ev/hr

Usage
-----
    python scripts/training/fixed_curriculum.py --seed 42
    python scripts/training/fixed_curriculum.py --seed 42 --compare-original
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
OUT_DIR   = os.path.join(ROOT, "results/curriculum_fixed")

# IMP-11: correct n_steps per episode
from env_alsat_debug import SIM_DURATION_S, SCHED_STEP_S
STEPS_PER_EP = max(1, int(SIM_DURATION_S / SCHED_STEP_S))   # = 144

CURRICULUM_SCHEDULE = [
    {"name": "static_clear",    "event_rate": 0.0, "clear_sky": True,  "n_episodes": 50},
    {"name": "static_clouds",   "event_rate": 0.0, "clear_sky": False, "n_episodes": 50},
    {"name": "dynamic_sparse",  "event_rate": 0.5, "clear_sky": False, "n_episodes": 100},
    {"name": "dynamic_dense",   "event_rate": 2.0, "clear_sky": False, "n_episodes": 300},
]


def run_fixed_curriculum(
    seed: int = 42,
    use_fixed_n_steps: bool = True,
) -> dict:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from env_dynamic_factory import make_env, Config

    os.makedirs(OUT_DIR, exist_ok=True)

    # IMP-11: n_steps matches episode length
    n_steps = STEPS_PER_EP if use_fixed_n_steps else 2048
    logger.info(
        f"[FixedCurriculum|s{seed}]  "
        f"n_steps={'fixed=' + str(n_steps) if use_fixed_n_steps else 'default=2048'}  "
        f"steps_per_ep={STEPS_PER_EP}"
    )

    def _make(event_rate, clear_sky, stage_seed):
        def _fn():
            env = make_env(Config.DYN_MODIS, TARGETS, CLOUD,
                           event_rate=event_rate, seed=stage_seed,
                           with_safety=False)
            if clear_sky:
                from curriculum import ClearSkyWrapper
                env = ClearSkyWrapper(env)
            return Monitor(env)
        return _fn

    history = []
    t0 = time.time()

    # Initial model on static_clear
    first_stage = CURRICULUM_SCHEDULE[0]
    vec = DummyVecEnv([_make(first_stage["event_rate"],
                              first_stage["clear_sky"], seed)])
    model = PPO(
        "MlpPolicy", vec,
        learning_rate=3e-4,
        n_steps=n_steps,   # IMP-11 key fix
        batch_size=min(64, n_steps),
        n_epochs=10, gamma=0.99, gae_lambda=0.95,
        ent_coef=0.15, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=0, seed=seed, device="cpu",
    )

    for stage in CURRICULUM_SCHEDULE:
        stage_steps = stage["n_episodes"] * STEPS_PER_EP
        logger.info(
            f"  Stage '{stage['name']}': {stage['n_episodes']} eps  "
            f"({stage_steps} steps)  rate={stage['event_rate']}"
        )

        # Rebuild vec env for new stage
        vec.close()
        vec = DummyVecEnv([_make(stage["event_rate"],
                                  stage["clear_sky"],
                                  seed + stage["n_episodes"])])
        model.set_env(vec)
        model.learn(
            total_timesteps=stage_steps,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        history.append({
            "stage":       stage["name"],
            "event_rate":  stage["event_rate"],
            "n_episodes":  stage["n_episodes"],
            "total_steps": stage_steps,
        })

    vec.close()
    elapsed = time.time() - t0

    # Save model
    label = "fixed" if use_fixed_n_steps else "original"
    out_path = os.path.join(MODELS, f"ppo_curriculum_{label}_s{seed}.zip")
    model.save(out_path)

    # Evaluate on final dynamic-dense
    eval_env = make_env(Config.DYN_MODIS, TARGETS, CLOUD,
                        event_rate=2.0, seed=seed + 99999, with_safety=False)
    rewards, dyn_rates = [], []
    for ep in range(30):
        obs, _ = eval_env.reset(seed=seed + 99999 + ep)
        done, ep_r = False, 0.0
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = eval_env.step(int(act))
            ep_r += r; done = term or trunc
        rewards.append(ep_r)
        m = info.get("episode_metrics", {})
        n_dyn = max(1, m.get("n_dyn_detected", 1))
        n_ok  = m.get("n_dyn_imaged", 0)
        dyn_rates.append(n_ok / n_dyn)
    eval_env.close()

    result = {
        "label":          label,
        "use_fixed_n_steps": use_fixed_n_steps,
        "n_steps_used":   n_steps,
        "steps_per_ep":   STEPS_PER_EP,
        "seed":           seed,
        "mean_reward":    float(np.mean(rewards)),
        "std_reward":     float(np.std(rewards)),
        "mean_dyn_suc":   float(np.mean(dyn_rates)),
        "elapsed_min":    round(elapsed / 60, 2),
        "model_path":     out_path,
        "history":        history,
    }
    logger.info(
        f"  Done: reward={result['mean_reward']:+.3f}  "
        f"dyn_suc={result['mean_dyn_suc']:.1%}  elapsed={result['elapsed_min']:.1f}min"
    )

    out_json = os.path.join(OUT_DIR, f"curriculum_{label}_s{seed}.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, default=float)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare-original", action="store_true",
                        help="Also run with original n_steps=2048 for comparison")
    args = parser.parse_args()

    logger.info("Running curriculum with IMP-11 fixed n_steps...")
    r_fixed = run_fixed_curriculum(seed=args.seed, use_fixed_n_steps=True)

    if args.compare_original:
        logger.info("\nRunning curriculum with original n_steps=2048...")
        r_orig  = run_fixed_curriculum(seed=args.seed, use_fixed_n_steps=False)
        delta   = r_fixed["mean_reward"] - r_orig["mean_reward"]
        print(f"\n--- IMP-11 Fix Comparison ---")
        print(f"  Fixed n_steps={STEPS_PER_EP}: reward={r_fixed['mean_reward']:+.3f}")
        print(f"  Orig  n_steps=2048:           reward={r_orig['mean_reward']:+.3f}")
        print(f"  Δ reward = {delta:+.4f}  "
              f"({'IMP-11 fix helps' if delta > 0 else 'no significant improvement'})")