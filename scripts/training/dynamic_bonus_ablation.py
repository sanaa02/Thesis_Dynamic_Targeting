#!/usr/bin/env python3
"""
dynamic_bonus_ablation.py  --  ALSAT-EO-1  IMP-09  Dynamic Bonus Sensitivity
=============================================================================
Grid search over DYNAMIC_BONUS and DYN_MULTIPLIER at event_rate=1.0 ev/hr.

Answers the reviewer question: "are your results robust to reward shaping?"

Grid
----
  DYNAMIC_BONUS   ∈ {0.0, 0.5, 1.0, 2.0}
  DYN_MULTIPLIER  ∈ {1.0, 1.5, 2.0}
  → 12 conditions × 3 seeds × 10 eval episodes each

Output
------
  results/ablation/dynamic_bonus/bonus_sensitivity_table.csv
  results/ablation/dynamic_bonus/bonus_sensitivity_heatmap.png

Usage
-----
    python scripts/training/dynamic_bonus_ablation.py --seeds 42 123 456
    python scripts/training/dynamic_bonus_ablation.py --quick  # 1 seed, 5 eps
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
OUT_DIR  = os.path.join(ROOT, "results/ablation/dynamic_bonus")

BONUS_GRID      = [0.0, 0.5, 1.0, 2.0]
MULT_GRID       = [1.0, 1.5, 2.0]
DEFAULT_SEEDS   = [42, 123, 456]
DEFAULT_EPISODES = 10
TRAIN_STEPS     = 50_000   # short training per grid point


def _patch_reward_constants(dynamic_bonus: float, dyn_multiplier: float) -> None:
    """Hot-patch DYNAMIC_BONUS and DYN_MULTIPLIER in the loaded module."""
    try:
        import dynamic_event as _de
        _de.DYNAMIC_BONUS  = dynamic_bonus
        _de.DYN_MULTIPLIER = dyn_multiplier
        logger.debug(
            f"  Patched: DYNAMIC_BONUS={dynamic_bonus}  DYN_MULTIPLIER={dyn_multiplier}")
    except Exception as exc:
        logger.warning(f"  Could not patch reward constants: {exc}")


def _run_one(
    dynamic_bonus: float,
    dyn_multiplier: float,
    seed: int,
    n_episodes: int,
    train_steps: int,
) -> dict:
    """Train and evaluate one (bonus, multiplier, seed) combination."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from env_dynamic_factory import make_env, Config
    from env_alsat_debug import SIM_DURATION_S, SCHED_STEP_S

    _patch_reward_constants(dynamic_bonus, dyn_multiplier)

    def _make():
        env = make_env(
            Config.DYN_MODIS, TARGETS, CLOUD,
            event_rate=1.0, seed=seed, with_safety=False,
        )
        return Monitor(env)

    vec = DummyVecEnv([_make])

    model = PPO(
        "MlpPolicy", vec,
        learning_rate=3e-4,
        n_steps=max(128, 2048),
        batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
        ent_coef=0.05, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=0, seed=seed, device="cpu",
    )
    model.learn(total_timesteps=train_steps, reset_num_timesteps=True)
    vec.close()

    # ── Evaluate ─────────────────────────────────────────────────────────────
    eval_env = make_env(
        Config.DYN_MODIS, TARGETS, CLOUD,
        event_rate=1.0, seed=seed + 9999, with_safety=False,
    )
    rewards, dyn_suc_rates, cf_rates = [], [], []
    for ep in range(n_episodes):
        obs, _ = eval_env.reset(seed=seed + 9999 + ep)
        done, ep_r, n_dyn, n_dyn_ok, n_cf = False, 0.0, 0, 0, 0
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = eval_env.step(int(act))
            ep_r += r
            done  = term or trunc
        rewards.append(ep_r)
        try:
            m = info.get("episode_metrics", {})
            n_dyn_ok = m.get("n_dyn_imaged", 0)
            n_dyn    = max(1, m.get("n_dyn_detected", n_dyn_ok))
            n_cf     = m.get("n_cloud_free", 0)
            n_total  = max(1, m.get("n_imaged", 1))
            dyn_suc_rates.append(n_dyn_ok / n_dyn)
            cf_rates.append(n_cf / n_total)
        except Exception:
            dyn_suc_rates.append(0.0)
            cf_rates.append(0.0)
    eval_env.close()

    return {
        "dynamic_bonus":    dynamic_bonus,
        "dyn_multiplier":   dyn_multiplier,
        "seed":             seed,
        "mean_reward":      float(np.mean(rewards)),
        "std_reward":       float(np.std(rewards)),
        "mean_dyn_suc":     float(np.mean(dyn_suc_rates)),
        "mean_cf_rate":     float(np.mean(cf_rates)),
    }


def run_grid(
    seeds: list[int] = DEFAULT_SEEDS,
    n_episodes: int  = DEFAULT_EPISODES,
    train_steps: int = TRAIN_STEPS,
    quick: bool      = False,
) -> list[dict]:
    if quick:
        seeds = [seeds[0]]
        n_episodes = 5
        train_steps = 10_000

    os.makedirs(OUT_DIR, exist_ok=True)
    results = []

    total = len(BONUS_GRID) * len(MULT_GRID) * len(seeds)
    done  = 0
    t0    = time.time()

    for bonus in BONUS_GRID:
        for mult in MULT_GRID:
            for seed in seeds:
                done += 1
                logger.info(
                    f"  [{done}/{total}] bonus={bonus}  mult={mult}  seed={seed}"
                )
                try:
                    r = _run_one(bonus, mult, seed, n_episodes, train_steps)
                    results.append(r)
                    logger.info(
                        f"    reward={r['mean_reward']:+.3f}  "
                        f"dyn_suc={r['mean_dyn_suc']:.1%}  "
                        f"cf={r['mean_cf_rate']:.1%}"
                    )
                except Exception as exc:
                    logger.error(f"    FAILED: {exc}")

    elapsed = time.time() - t0
    logger.info(f"\nGrid search complete in {elapsed/60:.1f} min")

    # ── Save results ──────────────────────────────────────────────────────────
    out_json = os.path.join(OUT_DIR, "bonus_sensitivity_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=float)
    logger.info(f"Results saved → {out_json}")

    _save_table(results)
    _save_heatmap(results)
    return results


def _save_table(results: list[dict]) -> None:
    """Save aggregated CSV table (mean over seeds)."""
    import csv
    from collections import defaultdict

    agg: dict = defaultdict(list)
    for r in results:
        key = (r["dynamic_bonus"], r["dyn_multiplier"])
        agg[key].append(r)

    out_csv = os.path.join(OUT_DIR, "bonus_sensitivity_table.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "dynamic_bonus", "dyn_multiplier",
            "mean_reward", "std_reward",
            "mean_dyn_suc", "mean_cf_rate",
        ])
        for (bonus, mult), rows in sorted(agg.items()):
            rewards    = [r["mean_reward"]  for r in rows]
            dyn_suc    = [r["mean_dyn_suc"] for r in rows]
            cf         = [r["mean_cf_rate"] for r in rows]
            w.writerow([
                bonus, mult,
                f"{np.mean(rewards):+.4f}", f"{np.std(rewards):.4f}",
                f"{np.mean(dyn_suc):.4f}",  f"{np.mean(cf):.4f}",
            ])
    logger.info(f"Table saved → {out_csv}")


def _save_heatmap(results: list[dict]) -> None:
    """Save reward heatmap (bonus × multiplier)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from collections import defaultdict

        agg: dict = defaultdict(list)
        for r in results:
            agg[(r["dynamic_bonus"], r["dyn_multiplier"])].append(r["mean_reward"])

        bonuses = sorted(set(r["dynamic_bonus"]  for r in results))
        mults   = sorted(set(r["dyn_multiplier"] for r in results))

        grid = np.zeros((len(bonuses), len(mults)))
        for i, b in enumerate(bonuses):
            for j, m in enumerate(mults):
                vals = agg.get((b, m), [0.0])
                grid[i, j] = float(np.mean(vals))

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(grid, cmap="RdYlGn", aspect="auto",
                       vmin=grid.min(), vmax=grid.max())
        ax.set_xticks(range(len(mults)));   ax.set_xticklabels(mults)
        ax.set_yticks(range(len(bonuses))); ax.set_yticklabels(bonuses)
        ax.set_xlabel("DYN_MULTIPLIER");    ax.set_ylabel("DYNAMIC_BONUS")
        ax.set_title("Mean Reward: DYNAMIC_BONUS × DYN_MULTIPLIER")
        plt.colorbar(im, ax=ax, label="Mean episode reward")
        for i in range(len(bonuses)):
            for j in range(len(mults)):
                ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center",
                        color="black", fontsize=8)
        fig.tight_layout()
        out_png = os.path.join(OUT_DIR, "bonus_sensitivity_heatmap.png")
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        logger.info(f"Heatmap saved → {out_png}")
    except ImportError:
        logger.info("  [SKIP] matplotlib not available — heatmap skipped")
    except Exception as exc:
        logger.warning(f"  Heatmap error: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic bonus sensitivity ablation")
    parser.add_argument("--seeds",    type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--steps",    type=int, default=TRAIN_STEPS)
    parser.add_argument("--quick",    action="store_true",
                        help="Quick run: 1 seed, 5 episodes, 10k steps")
    args = parser.parse_args()
    run_grid(
        seeds=args.seeds,
        n_episodes=args.episodes,
        train_steps=args.steps,
        quick=args.quick,
    )
