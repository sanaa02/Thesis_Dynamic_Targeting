#!/usr/bin/env python3
"""
bc_demo_collection.py  --  ALSAT-EO-1  IMP-16 / IMP-02  BC Demo Collection
============================================================================
Collects expert demonstrations for behaviour cloning (BC) pre-training.

Implements the IMP-02 fix: applies TargetIDObsWrapper to BOTH the demo
collection env and the BC training env so that obs dimensions match and
the obs→action mapping is learnable.

Expert policy: greedy heuristic that picks the static target with:
  1. Lowest cloud cover forecast (obs[2 + i*5])
  2. Highest priority × cloud_free probability
  3. Among dynamic slots: earliest expiring cloud-free event

Usage
-----
    python scripts/training/bc_demo_collection.py \
        --n-demos 1000 \
        --out data/demos/bc_demos.npz

    # Quick test:
    python scripts/training/bc_demo_collection.py --n-demos 100 --quick
"""
from __future__ import annotations

import argparse
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
OUT_DIR  = os.path.join(ROOT, "data/demos")

N_STATIC  = 20
N_DYN     = 3
N_ACTIONS = 24   # 20 + 3 + 1 drift


def _expert_action(obs: np.ndarray, imaged_set: set = None) -> int:
    """
    Greedy expert heuristic.

    Scoring:
      static targets: priority × (1 - cloud) × (1 - slew_norm)
      dynamic slots:  1.5 × priority × (1 - cloud) × urgency (1 - tta_norm)
                      only selected if cloud < 0.3 and tta < 0.5
      drift (23):     only if all other actions are blocked

    Parameters
    ----------
    obs : np.ndarray
        The observation vector
    imaged_set : set
        Set of static target indices already imaged this episode

    Returns the action index with highest score.
    """
    if imaged_set is None:
        imaged_set = set()
    
    scores = np.full(N_ACTIONS, -np.inf)

    # ── Static targets (actions 0–19) ────────────────────────────────────────
    N_VISIBLE = 6   # n_ahead_observe — only 6 targets visible in obs[13:43]
    for i in range(min(N_STATIC, N_VISIBLE)):
        if i in imaged_set:
            continue
        base = 13 + i * 5     # ← obs[13 + i*5] = priority of i-th upcoming opportunity
        if base + 4 >= len(obs):
            break
        priority   = float(obs[base])
        cloud_fcst = float(obs[base + 1])
        cloud_std  = float(obs[base + 2])
        opp_open   = float(obs[base + 3])
        slew_norm  = float(obs[base + 4])
        if opp_open < 0.5:
            # Still in window but barely open — score it low, not masked
            # This avoids producing all-drift demos when few opps are fully open
            scores[i] = (0.5 + 0.5 * priority) * (1.0 - cloud_fcst) * (1.0 - 0.3 * slew_norm) * 0.3
        else:
            scores[i] = (0.5 + 0.5 * priority) * (1.0 - cloud_fcst) * (1.0 - 0.3 * slew_norm)

    # ── Dynamic slots (actions 20–22) ─────────────────────────────────────────
    dyn_start = 43
    for slot in range(N_DYN):
        base = dyn_start + slot * 4
        if base + 3 >= len(obs):
            break
        prio_d   = float(obs[base])
        cloud_d  = float(obs[base + 1])
        tta_norm = float(obs[base + 2])
        slew_d   = float(obs[base + 3])

        if prio_d <= 0.01 and cloud_d >= 0.9:
            # Empty slot
            scores[N_STATIC + slot] = -2.0
        elif cloud_d < 0.3 and tta_norm < 0.5:
            urgency = 1.0 - tta_norm
            scores[N_STATIC + slot] = 1.0 * prio_d * (1.0 - cloud_d) * urgency
        else:
            scores[N_STATIC + slot] = -0.5

    # PATCH 3-1: drift gets a slightly negative score so the expert only
    # chooses it when ALL other options are genuinely poor (score < -0.10).
    # Previously 0.05 meant drift beat most borderline static opportunities,
    # causing 37% of demos to be drift — polluting the BC prior.
    # The -0.10 floor is above the -1.0 / -2.0 blocked/empty slot scores,
    # so it still functions as a fallback, just not the default.
    scores[N_STATIC + N_DYN] = -0.10

    return int(np.argmax(scores))


def collect_demos(
    n_demos:    int = 1000,
    out_path:   str = "",
    seed:       int = 42,
    event_rate: float = 1.0,
    use_target_id_wrapper: bool = True,
) -> str:
    """
    Collect expert demonstrations.

    Returns path to saved .npz file.
    """
    from env_dynamic_factory import make_env, Config
    from target_id_obs_wrapper import TargetIDObsWrapper

    if not out_path:
        out_path = os.path.join(OUT_DIR, f"bc_demos_n{n_demos}_s{seed}.npz")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def _make(ep_seed):
        env = make_env(Config.DYN_MODIS, TARGETS, CLOUD,
                       event_rate=event_rate, seed=ep_seed, with_safety=False)
        if use_target_id_wrapper:
            env = TargetIDObsWrapper(env)   # IMP-02 fix
        return env

    all_obs, all_acts, all_rews = [], [], []
    n_eps = 0
    t0    = time.time()

    ep_seed = seed
    while len(all_obs) < n_demos:
        env  = _make(ep_seed)
        obs, _ = env.reset(seed=ep_seed)
        done = False
        ep_obs, ep_acts, ep_rews = [], [], []
        imaged_static = set()                 # ← move this here from Loop 1
        while not done:
            action = _expert_action(obs, imaged_static)   # ← pass imaged_static
            ep_obs.append(obs)
            ep_acts.append(action)
            obs, r, term, trunc, info = env.step(action)
            ep_rews.append(r)
            done = term or trunc
            # imaged_static tracking from Loop 1 ...
            try:
                obj = env
                while hasattr(obj, "env"): obj = obj.env
                sat = getattr(obj, "unwrapped", obj).satellites[0]
                n_static = len(sat.scenario.targets)
                if action < n_static:
                    tgt = sat.scenario.targets[action]
                    if getattr(tgt, "imaged", False):
                        imaged_static.add(action)
            except Exception:
                pass
        env.close()
        all_obs.extend(ep_obs)
        all_acts.extend(ep_acts)
        all_rews.extend(ep_rews)
        n_eps   += 1
        ep_seed += 1

        if n_eps % 20 == 0:
            logger.info(
                f"  episodes={n_eps}  transitions={len(all_obs)}  "
                f"avg_r={np.mean(all_rews[-200:]):+.3f}  "
                f"elapsed={time.time()-t0:.0f}s"
            )

    # Trim to exact n_demos
    all_obs  = np.array(all_obs[:n_demos],  dtype=np.float32)
    all_acts = np.array(all_acts[:n_demos], dtype=np.int64)
    all_rews = np.array(all_rews[:n_demos], dtype=np.float32)

    np.savez_compressed(out_path, obs=all_obs, actions=all_acts, rewards=all_rews)
    elapsed = time.time() - t0
    logger.info(
        f"Saved {len(all_obs)} demos → {out_path}  "
        f"({n_eps} eps, {elapsed/60:.1f} min)"
    )

    # ── Action distribution ────────────────────────────────────────────────────
    unique, counts = np.unique(all_acts, return_counts=True)
    logger.info("Action distribution (top 8):")
    top = sorted(zip(counts, unique), reverse=True)[:8]
    for cnt, act in top:
        atype = "static" if act < N_STATIC else ("dynamic" if act < N_STATIC + N_DYN else "drift")
        logger.info(f"  action={act:2d} ({atype:8s})  {cnt/len(all_acts):5.1%}")

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-demos",    type=int,   default=1000)
    parser.add_argument("--out",        default="")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--event-rate", type=float, default=1.0)
    parser.add_argument("--no-wrapper", action="store_true",
                        help="Disable TargetIDObsWrapper (disable IMP-02 fix)")
    parser.add_argument("--quick",      action="store_true",
                        help="Collect 200 demos only (quick test)")
    args = parser.parse_args()

    n = 200 if args.quick else args.n_demos
    out = collect_demos(
        n_demos=n,
        out_path=args.out,
        seed=args.seed,
        event_rate=args.event_rate,
        use_target_id_wrapper=not args.no_wrapper,
    )
    print(f"\nDemos saved: {out}")
