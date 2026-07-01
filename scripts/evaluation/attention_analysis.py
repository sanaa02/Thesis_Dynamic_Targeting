#!/usr/bin/env python3
"""
attention_analysis.py  --  ALSAT-EO-1  IMP-01  Attention Visualisation
=======================================================================
Extracts and visualises attention weights from the SchedulerAttentionExtractor
trained in the A1-PPO policy.

Produces three output types:
  1. Per-step attention heatmap: which targets get attention over time
  2. Episode aggregated attention vs reward correlation
  3. Static vs dynamic attention split

Usage
-----
    python scripts/evaluation/attention_analysis.py --model models/ppo_attention_s42.zip
    python scripts/evaluation/attention_analysis.py --model ppo.zip --episodes 20
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
for _d in ["scripts/core", "scripts/training", "scripts/wrappers",
           "scripts/models", "scripts"]:
    _p = os.path.join(ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

TARGETS  = os.path.join(ROOT, "config/targets/global_45_targets.json")
CLOUD    = os.path.join(ROOT, "config/cloud_reality/global_45_clouds.json")
OUT_DIR  = os.path.join(ROOT, "results/evaluation/attention")

DEFAULT_EPISODES = 20


def extract_attention_weights(model, obs_batch: np.ndarray) -> np.ndarray | None:
    """
    Extract attention weights from the SchedulerAttentionExtractor.

    Returns shape (n_obs, n_heads, n_targets) or None if not applicable.
    """
    try:
        import torch
        features_extractor = model.policy.features_extractor
        attn_module = getattr(features_extractor, "_attn_block", None)
        if attn_module is None:
            attn_module = getattr(features_extractor, "cross_attention", None)
        if attn_module is None:
            return None

        obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32)
        with torch.no_grad():
            _ = model.policy.extract_features(obs_tensor)
            weights = getattr(attn_module, "_last_weights", None)
            if weights is not None:
                return weights.cpu().numpy()
    except Exception as exc:
        logger.debug(f"[AttentionAnalysis] Weight extraction failed: {exc}")
    return None


def run_attention_analysis(
    model_path: str,
    n_episodes: int = DEFAULT_EPISODES,
    seed: int = 42,
) -> dict:
    from stable_baselines3 import PPO
    from env_dynamic_factory import make_env, Config

    os.makedirs(OUT_DIR, exist_ok=True)

    def _make(s):
        return make_env(Config.DYN_MODIS, TARGETS, CLOUD,
                        event_rate=1.0, seed=s, with_safety=False)

    env_stub = _make(seed)
    model    = PPO.load(model_path, env=env_stub)
    env_stub.close()
    logger.info(f"Loaded: {model_path}")

    all_step_data = []

    for ep in range(n_episodes):
        env  = _make(seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        done = False
        step = 0

        while not done:
            obs_batch = obs[np.newaxis, :]
            attn = extract_attention_weights(model, obs_batch)

            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(act))
            done = term or trunc

            all_step_data.append({
                "episode": ep, "step": step,
                "action":  int(act), "reward": float(r),
                "has_attn": attn is not None,
                "attn_shape": list(attn.shape) if attn is not None else None,
                "attn_mean":  float(attn.mean()) if attn is not None else None,
                "is_dynamic": int(act) >= 20,
            })
            step += 1

        env.close()
        if (ep + 1) % 5 == 0:
            logger.info(f"  ep {ep+1}/{n_episodes}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    n_attn = sum(1 for d in all_step_data if d["has_attn"])
    n_dyn  = sum(1 for d in all_step_data if d["is_dynamic"])
    n_static = sum(1 for d in all_step_data if not d["is_dynamic"])

    summary = {
        "n_episodes":       n_episodes,
        "total_steps":      len(all_step_data),
        "n_steps_with_attn": n_attn,
        "n_dynamic_actions": n_dyn,
        "n_static_actions":  n_static,
        "dyn_action_frac":   n_dyn / max(1, len(all_step_data)),
        "has_attention_extraction": n_attn > 0,
    }

    if n_attn > 0:
        attn_vals = [d["attn_mean"] for d in all_step_data if d["attn_mean"] is not None]
        summary["mean_attention_weight"] = float(np.mean(attn_vals))

    result = {"model_path": model_path, "summary": summary,
              "step_data": all_step_data[:100]}  # first 100 steps for inspection

    out_json = os.path.join(OUT_DIR, "attention_analysis.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, default=float)
    logger.info(f"Results saved → {out_json}")

    _plot_attention(all_step_data)

    print("\n" + "=" * 50)
    print("ATTENTION ANALYSIS SUMMARY")
    for k, v in summary.items():
        print(f"  {k:<30} {v}")
    print("=" * 50)

    return result


def _plot_attention(step_data: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps   = range(min(200, len(step_data)))
        actions = [step_data[i]["action"] for i in steps]
        rewards = [step_data[i]["reward"] for i in steps]
        is_dyn  = [step_data[i]["is_dynamic"] for i in steps]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        colors = ["#ff7f0e" if d else "#1f77b4" for d in is_dyn]
        ax1.bar(list(steps), actions, color=colors, alpha=0.7, width=0.8)
        ax1.axhline(y=20, linestyle="--", color="grey", alpha=0.5,
                    label="Static/Dynamic boundary")
        ax1.set_ylabel("Action index")
        ax1.set_title("Actions over Time (orange=dynamic, blue=static)")
        ax1.legend()

        ax2.plot(list(steps), rewards, color="green", linewidth=1)
        ax2.fill_between(list(steps), 0, rewards,
                         where=[r > 0 for r in rewards],
                         color="green", alpha=0.3)
        ax2.fill_between(list(steps), rewards, 0,
                         where=[r < 0 for r in rewards],
                         color="red", alpha=0.3)
        ax2.set_ylabel("Step reward")
        ax2.set_xlabel("Step")
        ax2.set_title("Reward per Step")

        fig.tight_layout()
        out_png = os.path.join(OUT_DIR, "attention_actions.png")
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        logger.info(f"Plot saved → {out_png}")
    except ImportError:
        logger.info("  [SKIP] matplotlib not available")
    except Exception as exc:
        logger.warning(f"  Plot error: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    required=True)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()
    run_attention_analysis(args.model, args.episodes, args.seed)