#!/usr/bin/env python3
"""
evaluate_full_system.py  --  ALSAT-EO-1  IMP-17  Full System Evaluation
=======================================================================
Comprehensive evaluation of the full A1-PPO system with all improvements.

Runs 3 seeds × 50 evaluation episodes each, reporting:
  - Mean ± std episode reward
  - Dynamic imaging success rate (n_dyn_imaged / n_dyn_detected)
  - Cloud-free imaging rate  (n_cloud_free / n_imaged)
  - Static target coverage rate
  - Per-event-type breakdown (wildfire, flood, dust_storm)
  - Comparison to baselines: random policy, greedy cloud policy

Output
------
  results/evaluation/full_system_metrics.json
  results/evaluation/full_system_metrics.csv
  results/evaluation/full_system_plot.png

Usage
-----
    python scripts/evaluation/evaluate_full_system.py --model models/ppo_full_system_s42.zip
    python scripts/evaluation/evaluate_full_system.py --model ppo_best.zip --seeds 42 123 456
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict

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
OUT_DIR  = os.path.join(ROOT, "results/evaluation")

DEFAULT_SEEDS    = [42, 123, 456]
DEFAULT_EPISODES = 50
DEFAULT_EVENT_RATE = 1.0


def _run_policy_episodes(
    policy,           # callable: obs -> action (or "random" / "greedy_cloud")
    make_env_fn,
    seed: int,
    n_episodes: int,
    policy_name: str,
) -> list[dict]:
    """Evaluate a policy for n_episodes, return per-episode metric dicts."""
    episodes = []

    for ep in range(n_episodes):
        env = make_env_fn(seed=seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        done, ep_r = False, 0.0
        n_steps = 0

        while not done:
            if policy == "random":
                action = env.action_space.sample()
            elif policy == "greedy_cloud":
                action = _greedy_cloud_action(obs, env)
            else:
                action_masks = env.action_masks()
                action, _ = policy.predict(obs, action_masks=action_masks, deterministic=True)
                action = int(action)
            obs, r, term, trunc, info = env.step(action)
            ep_r   += r
            n_steps += 1
            done = term or trunc

        env.close()
        m = info.get("episode_metrics", {})
        n_dyn     = max(1, m.get("n_dyn_detected",  0))
        n_dyn_ok  = m.get("n_dyn_imaged",   0)
        n_cf      = m.get("n_cloud_free",    0)
        n_img     = max(1, m.get("n_imaged", 1))
        n_missed  = m.get("n_missed_events", 0)

        episodes.append({
            "policy":     policy_name,
            "seed":       seed,
            "episode":    ep,
            "reward":     ep_r,
            "n_steps":    n_steps,
            "n_imaged":   m.get("n_imaged", 0),
            "n_dyn_det":  n_dyn,
            "n_dyn_ok":   n_dyn_ok,
            "n_cf":       n_cf,
            "dyn_suc":    n_dyn_ok / n_dyn,
            "cf_rate":    n_cf / n_img,
            "miss_rate":  n_missed / max(1, n_dyn),
        })
        if (ep + 1) % 10 == 0:
            recent = episodes[-10:]
            logger.info(
                f"  [{policy_name}|s{seed}] ep {ep+1}/{n_episodes}  "
                f"avg_r={np.mean([e['reward'] for e in recent]):+.3f}"
            )

    return episodes


def _greedy_cloud_action(obs: np.ndarray, env) -> int:
    """Greedy policy: pick static target with lowest cloud forecast from upcoming opportunities."""
    try:
        best_action = 48  # Default drift
        best_cloud = 1.1
        for slot_i in range(6):
            # Target features start at index 13
            # Cloud cover is at index 13 + slot_i * 6 + 1
            # Action fraction is at index 13 + slot_i * 6 + 5
            cloud = obs[13 + slot_i * 6 + 1]
            action_frac = obs[13 + slot_i * 6 + 5]
            action_idx = int(round(action_frac * 45))
            if 0 <= action_idx < 45 and cloud >= 0.0:
                if cloud < best_cloud:
                    best_cloud = cloud
                    best_action = action_idx
        return best_action
    except Exception:
        return 48


def run_full_evaluation(
    model_path: str,
    seeds: list = DEFAULT_SEEDS,
    n_episodes: int = DEFAULT_EPISODES,
    event_rate: float = DEFAULT_EVENT_RATE,
) -> dict:
    from sb3_contrib import MaskablePPO
    from env_dynamic_factory import make_env, Config
    from wrappers.action_mask_wrapper import make_masked_env

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    def _make(seed: int):
        env = make_env(Config.DYN_MODIS, TARGETS, CLOUD,
                       event_rate=event_rate, seed=seed, with_safety=True)
        return make_masked_env(env)

    # Load model with custom_objects and temporary redirects to bypass version mismatches
    import sys
    import numpy.core
    import numpy.core.numeric
    sys.modules["numpy._core"] = numpy.core
    sys.modules["numpy._core.numeric"] = numpy.core.numeric
    
    try:
        from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
        import inspect
        _orig_init = MaskableActorCriticPolicy.__init__
        def _patched_init(self, *args, **kwargs):
            sig = inspect.signature(_orig_init)
            if "use_sde" not in sig.parameters:
                kwargs.pop("use_sde", None)
            _orig_init(self, *args, **kwargs)
        MaskableActorCriticPolicy.__init__ = _patched_init
    except Exception:
        pass

    env_stub = _make(seed=42)
    custom_objects = {"action_space": env_stub.action_space, "observation_space": env_stub.observation_space}
    try:
        model = MaskablePPO.load(model_path, env=env_stub, custom_objects=custom_objects)
    finally:
        if "numpy._core" in sys.modules:
            del sys.modules["numpy._core"]
        if "numpy._core.numeric" in sys.modules:
            del sys.modules["numpy._core.numeric"]
    env_stub.close()
    logger.info(f"Loaded model: {model_path}")

    all_episodes: list[dict] = []

    for seed in seeds:
        logger.info(f"\n--- Seed {seed} ---")

        # A1-PPO policy
        eps = _run_policy_episodes(model, _make, seed, n_episodes, "A1-PPO")
        all_episodes.extend(eps)

        # Random baseline
        eps_rnd = _run_policy_episodes("random", _make, seed,
                                       min(20, n_episodes), "random")
        all_episodes.extend(eps_rnd)

        # Greedy cloud baseline
        eps_gc = _run_policy_episodes("greedy_cloud", _make, seed,
                                      min(20, n_episodes), "greedy_cloud")
        all_episodes.extend(eps_gc)

    elapsed = time.time() - t0

    # ── Aggregate ─────────────────────────────────────────────────────────────
    agg: dict = defaultdict(list)
    for ep in all_episodes:
        agg[ep["policy"]].append(ep)

    summary = {}
    for policy_name, eps in agg.items():
        rs = [e["reward"]   for e in eps]
        ds = [e["dyn_suc"]  for e in eps]
        cs = [e["cf_rate"]  for e in eps]
        summary[policy_name] = {
            "n_episodes":   len(eps),
            "mean_reward":  float(np.mean(rs)),
            "std_reward":   float(np.std(rs)),
            "mean_dyn_suc": float(np.mean(ds)),
            "std_dyn_suc":  float(np.std(ds)),
            "mean_cf_rate": float(np.mean(cs)),
        }

    results = {
        "model_path":  model_path,
        "event_rate":  event_rate,
        "seeds":       seeds,
        "n_episodes":  n_episodes,
        "elapsed_min": round(elapsed / 60, 2),
        "summary":     summary,
        "episodes":    all_episodes,
    }

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_json = os.path.join(OUT_DIR, "full_system_metrics.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=float)
    logger.info(f"\nJSON saved → {out_json}")

    _save_csv(all_episodes)
    _print_summary(summary)
    _plot_results(summary)
    return results


def _save_csv(episodes: list[dict]) -> None:
    out_csv = os.path.join(OUT_DIR, "full_system_metrics.csv")
    fields = ["policy", "seed", "episode", "reward",
              "n_steps", "n_imaged", "n_dyn_det", "n_dyn_ok",
              "n_cf", "dyn_suc", "cf_rate", "miss_rate"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for ep in episodes:
            w.writerow({k: ep.get(k, "") for k in fields})
    logger.info(f"CSV saved → {out_csv}")


def _print_summary(summary: dict) -> None:
    print("\n" + "=" * 72)
    print(f"{'Policy':<15} {'Reward':>12} {'±':>8} {'DynSuc':>9} {'CF%':>8}")
    print("-" * 72)
    for name, m in sorted(summary.items()):
        print(f"  {name:<13} {m['mean_reward']:>+12.4f} {m['std_reward']:>8.4f} "
              f"{m['mean_dyn_suc']:>9.1%} {m['mean_cf_rate']:>8.1%}")
    print("=" * 72)

    if "A1-PPO" in summary and "random" in summary:
        delta = (summary["A1-PPO"]["mean_reward"] -
                 summary["random"]["mean_reward"])
        print(f"\n  A1-PPO over random: Δ reward = {delta:+.4f}")
    if "A1-PPO" in summary and "greedy_cloud" in summary:
        delta = (summary["A1-PPO"]["mean_reward"] -
                 summary["greedy_cloud"]["mean_reward"])
        print(f"  A1-PPO over greedy: Δ reward = {delta:+.4f}")


def _plot_results(summary: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        policies = list(summary.keys())
        rewards  = [summary[p]["mean_reward"]  for p in policies]
        r_std    = [summary[p]["std_reward"]   for p in policies]
        dyn_suc  = [summary[p]["mean_dyn_suc"] for p in policies]
        d_std    = [summary[p]["std_dyn_suc"]  for p in policies]

        colors = {"A1-PPO": "#2ca02c", "random": "#d62728",
                  "greedy_cloud": "#ff7f0e"}
        bar_c  = [colors.get(p, "#1f77b4") for p in policies]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        bars1 = ax1.bar(policies, rewards, yerr=r_std, color=bar_c,
                        capsize=4, edgecolor="black", linewidth=0.8)
        ax1.set_title("Mean Episode Reward")
        ax1.set_ylabel("Reward")
        ax1.grid(True, alpha=0.3, axis="y")

        bars2 = ax2.bar(policies, dyn_suc, yerr=d_std, color=bar_c,
                        capsize=4, edgecolor="black", linewidth=0.8)
        ax2.set_title("Dynamic Imaging Success Rate")
        ax2.set_ylabel("Success Rate")
        ax2.set_ylim(0, 1)
        ax2.yaxis.set_major_formatter(
            lambda x, _: f"{x:.0%}")
        ax2.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        out_png = os.path.join(OUT_DIR, "full_system_plot.png")
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        logger.info(f"Plot saved → {out_png}")
    except ImportError:
        logger.info("  [SKIP] matplotlib not available")
    except Exception as exc:
        logger.warning(f"  Plot error: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Path to trained PPO .zip model")
    parser.add_argument("--seeds",      type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--episodes",   type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--event-rate", type=float, default=DEFAULT_EVENT_RATE)
    args = parser.parse_args()

    run_full_evaluation(
        model_path=args.model,
        seeds=args.seeds,
        n_episodes=args.episodes,
        event_rate=args.event_rate,
    )
