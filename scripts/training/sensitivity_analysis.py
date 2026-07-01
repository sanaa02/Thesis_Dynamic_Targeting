#!/usr/bin/env python3
"""
sensitivity_analysis.py  --  ALSAT-EO-1  IMP-15  Generalisation to Unseen Event Rates
========================================================================================
Evaluates a trained A1-PPO policy at event rates it was NOT trained on.

Tests zero-shot generalisation: policy trained at λ=1.0 ev/hr evaluated
at λ ∈ [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0] ev/hr.

Outputs
-------
  results/sensitivity/rate_generalisation.json
  results/sensitivity/rate_generalisation.png  (reward + dyn_suc vs rate)

Usage
-----
    python scripts/training/sensitivity_analysis.py --model models/ppo_full_system_s42.zip
    python scripts/training/sensitivity_analysis.py --model ppo_best.zip --rates 0.5 1.0 2.0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

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
OUT_DIR  = os.path.join(ROOT, "results/sensitivity")

DEFAULT_RATES    = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
DEFAULT_EPISODES = 30
DEFAULT_SEED     = 42


def evaluate_at_rate(
    model,
    event_rate: float,
    seed: int = DEFAULT_SEED,
    n_episodes: int = DEFAULT_EPISODES,
) -> dict:
    """Evaluate loaded PPO model at a specific event rate."""
    from env_dynamic_factory import make_env, Config

    env = make_env(
        Config.DYN_MODIS, TARGETS, CLOUD,
        event_rate=event_rate, seed=seed + int(event_rate * 1000),
        with_safety=False,
    )

    rewards, dyn_suc_rates, cf_rates = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ep_r = False, 0.0
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(act))
            ep_r += r
            done  = term or trunc
        rewards.append(ep_r)
        try:
            m = info.get("episode_metrics", {})
            dyn_ok  = m.get("n_dyn_imaged",   0)
            dyn_det = max(1, m.get("n_dyn_detected", max(1, dyn_ok)))
            cf      = m.get("n_cloud_free",    0)
            n_img   = max(1, m.get("n_imaged",        1))
            dyn_suc_rates.append(dyn_ok / dyn_det)
            cf_rates.append(cf / n_img)
        except Exception:
            dyn_suc_rates.append(0.0)
            cf_rates.append(0.0)

    env.close()

    result = {
        "event_rate":    event_rate,
        "mean_reward":   float(np.mean(rewards)),
        "std_reward":    float(np.std(rewards)),
        "mean_dyn_suc":  float(np.mean(dyn_suc_rates)),
        "std_dyn_suc":   float(np.std(dyn_suc_rates)),
        "mean_cf_rate":  float(np.mean(cf_rates)),
        "n_episodes":    n_episodes,
    }
    logger.info(
        f"  rate={event_rate:.2f}  reward={result['mean_reward']:+.3f}  "
        f"dyn_suc={result['mean_dyn_suc']:.1%}  cf={result['mean_cf_rate']:.1%}"
    )
    return result


def run_sensitivity(
    model_path: str,
    rates: list[float] = DEFAULT_RATES,
    seed:  int = DEFAULT_SEED,
    n_episodes: int = DEFAULT_EPISODES,
) -> list[dict]:
    from stable_baselines3 import PPO
    from env_dynamic_factory import make_env, Config

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load model
    env_stub = make_env(Config.DYN_MODIS, TARGETS, CLOUD,
                        event_rate=1.0, seed=seed)
    model = PPO.load(model_path, env=env_stub)
    env_stub.close()
    logger.info(f"Loaded model: {model_path}")

    results = []
    for rate in rates:
        logger.info(f"\nEvaluating at event_rate={rate:.2f} ev/hr...")
        r = evaluate_at_rate(model, event_rate=rate, seed=seed,
                             n_episodes=n_episodes)
        results.append(r)

    # Save
    out_json = os.path.join(OUT_DIR, "rate_generalisation.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=float)
    logger.info(f"\nResults saved → {out_json}")

    _plot_results(results)
    return results


def _plot_results(results: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rates   = [r["event_rate"]   for r in results]
        rewards = [r["mean_reward"]  for r in results]
        r_std   = [r["std_reward"]   for r in results]
        dyn_suc = [r["mean_dyn_suc"] for r in results]
        d_std   = [r["std_dyn_suc"]  for r in results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.errorbar(rates, rewards, yerr=r_std, marker="o", color="steelblue",
                     capsize=4, label="PPO")
        ax1.axvline(x=1.0, linestyle="--", color="grey", alpha=0.5,
                    label="Training rate")
        ax1.set_xlabel("Event rate (ev/hr)")
        ax1.set_ylabel("Mean episode reward")
        ax1.set_title("Reward vs Event Rate (Zero-shot Generalisation)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.errorbar(rates, dyn_suc, yerr=d_std, marker="s", color="darkorange",
                     capsize=4, label="PPO")
        ax2.axvline(x=1.0, linestyle="--", color="grey", alpha=0.5,
                    label="Training rate")
        ax2.set_xlabel("Event rate (ev/hr)")
        ax2.set_ylabel("Dynamic success rate")
        ax2.set_title("Dynamic Success vs Event Rate")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        out_png = os.path.join(OUT_DIR, "rate_generalisation.png")
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        logger.info(f"Plot saved → {out_png}")
    except ImportError:
        logger.info("  [SKIP] matplotlib not available — plot skipped")
    except Exception as exc:
        logger.warning(f"  Plot error: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Path to trained PPO .zip model")
    parser.add_argument("--rates", type=float, nargs="+",
                        default=DEFAULT_RATES,
                        help="Event rates to evaluate at (ev/hr)")
    parser.add_argument("--seed",     type=int, default=DEFAULT_SEED)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    args = parser.parse_args()

    run_sensitivity(
        model_path=args.model,
        rates=args.rates,
        seed=args.seed,
        n_episodes=args.episodes,
    )
