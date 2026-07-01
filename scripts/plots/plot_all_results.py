#!/usr/bin/env python3
"""
plot_all_results.py  --  ALSAT-EO-1  IMP-17  Master Results Plotter
====================================================================
Reads all ablation/evaluation JSON results and produces publication-quality
figures for the A1 paper.

Figures generated:
  fig1_full_system.png         -- A1-PPO vs baselines (reward + dyn_suc)
  fig2_smdp_ablation.png       -- SMDP vs Flat MDP
  fig3_entropy_ablation.png    -- Fixed vs annealed entropy
  fig4_transfer_learning.png   -- Static pretrain vs scratch
  fig5_rate_sensitivity.png    -- Reward vs event rate (zero-shot)
  fig6_cloud_ablation.png      -- Standard vs oracle cloud
  fig7_attention_heatmap.png   -- Attention weights visualisation
  fig8_bonus_heatmap.png       -- DYNAMIC_BONUS × DYN_MULTIPLIER grid

Usage
-----
    python scripts/plots/plot_all_results.py
    python scripts/plots/plot_all_results.py --results_dir results/ --out_dir results/figures/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)

ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR  = os.path.join(ROOT, "results/figures")
RESULTS  = os.path.join(ROOT, "results")

# ─────────────────────────────────────────────────────────────────────────────
# MPL setup
# ─────────────────────────────────────────────────────────────────────────────

def _init_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 13,
        "axes.labelsize": 12, "legend.fontsize": 10,
        "figure.dpi": 120,
    })
    return plt

# ─────────────────────────────────────────────────────────────────────────────
# Individual figure generators
# ─────────────────────────────────────────────────────────────────────────────

def fig1_full_system(out_dir: str) -> None:
    """Bar chart: A1-PPO vs random vs greedy_cloud (reward + dyn_suc)."""
    path = os.path.join(RESULTS, "evaluation/full_system_metrics.json")
    if not os.path.exists(path):
        logger.info(f"  [SKIP fig1] {path} not found")
        return

    plt = _init_mpl()
    with open(path) as f:
        data = json.load(f)

    summary = data.get("summary", {})
    policies = ["A1-PPO", "random", "greedy_cloud"]
    labels   = ["A1-PPO", "Random", "Greedy\nCloud"]
    colors   = ["#2ca02c", "#d62728", "#ff7f0e"]

    rewards = [summary.get(p, {}).get("mean_reward",  0.0) for p in policies]
    r_std   = [summary.get(p, {}).get("std_reward",   0.0) for p in policies]
    dyn_suc = [summary.get(p, {}).get("mean_dyn_suc", 0.0) for p in policies]
    d_std   = [summary.get(p, {}).get("std_dyn_suc",  0.0) for p in policies]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(labels, rewards, yerr=r_std, color=colors, capsize=4,
            edgecolor="black", linewidth=0.8)
    ax1.set_title("Mean Episode Reward")
    ax1.set_ylabel("Reward")
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(labels, dyn_suc, yerr=d_std, color=colors, capsize=4,
            edgecolor="black", linewidth=0.8)
    ax2.set_title("Dynamic Imaging Success Rate")
    ax2.set_ylabel("Success Rate")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("A1-PPO: Full System Evaluation", fontweight="bold")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "fig1_full_system.png"))


def fig2_smdp_ablation(out_dir: str) -> None:
    """Side-by-side: SMDP vs Flat MDP reward + dyn_suc."""
    path = os.path.join(RESULTS, "ablation/smdp_vs_flat/smdp_vs_flat_results.json")
    if not os.path.exists(path):
        logger.info(f"  [SKIP fig2] {path} not found")
        return

    plt = _init_mpl()
    with open(path) as f:
        data = json.load(f)

    smdp = data.get("smdp", {})
    flat = data.get("flat", {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    labels = ["SMDP\n(proposed)", "Flat MDP\n(baseline)"]
    colors = ["#2ca02c", "#d62728"]

    ax1.bar(labels, [smdp.get("mean_reward", 0), flat.get("mean_reward", 0)],
            color=colors, edgecolor="black")
    ax1.set_title("SMDP vs Flat MDP: Reward")
    ax1.set_ylabel("Mean Episode Reward")
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(labels, [smdp.get("mean_dyn_suc", 0), flat.get("mean_dyn_suc", 0)],
            color=colors, edgecolor="black")
    ax2.set_title("Dynamic Imaging Success")
    ax2.set_ylabel("Success Rate")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("IMP-08: SMDP Discounting Ablation", fontweight="bold")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "fig2_smdp_ablation.png"))


def fig3_entropy_ablation(out_dir: str) -> None:
    """Bar chart: fixed-low vs fixed-high vs annealed entropy."""
    path = os.path.join(RESULTS, "ablation/entropy/entropy_ablation_results.json")
    if not os.path.exists(path):
        logger.info(f"  [SKIP fig3] {path} not found")
        return

    plt = _init_mpl()
    with open(path) as f:
        results = json.load(f)

    from collections import defaultdict
    agg: dict = defaultdict(list)
    for r in results:
        agg[r["condition"]].append(r["mean_reward"])

    conditions = ["fixed_low", "fixed_high", "annealed"]
    labels     = ["Fixed Low\n(0.05)", "Fixed High\n(0.15)", "Annealed\n(0.15→0.01)"]
    means      = [np.mean(agg.get(c, [0])) for c in conditions]
    stds       = [np.std( agg.get(c, [0])) for c in conditions]
    colors     = ["#d62728", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, means, yerr=stds, color=colors, capsize=4,
           edgecolor="black", linewidth=0.8)
    ax.set_title("IMP-06: Entropy Coefficient Ablation")
    ax.set_ylabel("Mean Episode Reward")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "fig3_entropy_ablation.png"))


def fig4_transfer_learning(out_dir: str) -> None:
    """Bar: transfer vs scratch."""
    t_path = os.path.join(RESULTS, "transfer_learning")
    if not os.path.isdir(t_path):
        logger.info(f"  [SKIP fig4] {t_path} not found")
        return

    plt = _init_mpl()
    transfer_rewards, scratch_rewards = [], []

    for fname in os.listdir(t_path):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(t_path, fname)) as f:
            d = json.load(f)
        if d.get("label") == "transfer":
            transfer_rewards.append(d["mean_reward"])
        elif d.get("label") == "scratch":
            scratch_rewards.append(d["mean_reward"])

    if not transfer_rewards or not scratch_rewards:
        logger.info("  [SKIP fig4] no transfer/scratch results found")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Static Pretrain\n+ Dynamic Fine-tune", "Scratch\n(dynamic only)"]
    means  = [np.mean(transfer_rewards), np.mean(scratch_rewards)]
    stds   = [np.std(transfer_rewards),  np.std(scratch_rewards)]
    colors = ["#2ca02c", "#d62728"]
    ax.bar(labels, means, yerr=stds, color=colors, capsize=4,
           edgecolor="black", linewidth=0.8)
    ax.set_title("IMP-12: Transfer Learning vs From Scratch")
    ax.set_ylabel("Mean Episode Reward (dynamic env)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "fig4_transfer_learning.png"))


def fig5_rate_sensitivity(out_dir: str) -> None:
    """Line: reward + dyn_suc vs event rate."""
    path = os.path.join(RESULTS, "sensitivity/rate_generalisation.json")
    if not os.path.exists(path):
        logger.info(f"  [SKIP fig5] {path} not found")
        return

    plt = _init_mpl()
    with open(path) as f:
        results = json.load(f)

    rates   = [r["event_rate"]   for r in results]
    rewards = [r["mean_reward"]  for r in results]
    r_std   = [r["std_reward"]   for r in results]
    dyn_suc = [r["mean_dyn_suc"] for r in results]
    d_std   = [r["std_dyn_suc"]  for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.errorbar(rates, rewards, yerr=r_std, marker="o", color="steelblue", capsize=4)
    ax1.axvline(x=1.0, linestyle="--", color="grey", alpha=0.5, label="Train rate")
    ax1.set_title("Reward vs Event Rate")
    ax1.set_xlabel("Event rate (ev/hr)")
    ax1.set_ylabel("Mean Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.errorbar(rates, dyn_suc, yerr=d_std, marker="s", color="darkorange", capsize=4)
    ax2.axvline(x=1.0, linestyle="--", color="grey", alpha=0.5, label="Train rate")
    ax2.set_title("Dynamic Success vs Event Rate")
    ax2.set_xlabel("Event rate (ev/hr)")
    ax2.set_ylabel("Dynamic Success Rate")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("IMP-15: Zero-shot Generalisation to Unseen Event Rates", fontweight="bold")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "fig5_rate_sensitivity.png"))


def fig8_bonus_heatmap(out_dir: str) -> None:
    """Heatmap: DYNAMIC_BONUS × DYN_MULTIPLIER grid search."""
    path = os.path.join(RESULTS, "ablation/dynamic_bonus/bonus_sensitivity_results.json")
    if not os.path.exists(path):
        logger.info(f"  [SKIP fig8] {path} not found")
        return

    plt = _init_mpl()
    with open(path) as f:
        results = json.load(f)

    from collections import defaultdict
    agg: dict = defaultdict(list)
    for r in results:
        agg[(r["dynamic_bonus"], r["dyn_multiplier"])].append(r["mean_reward"])

    bonuses = sorted(set(r["dynamic_bonus"]  for r in results))
    mults   = sorted(set(r["dyn_multiplier"] for r in results))
    grid    = np.zeros((len(bonuses), len(mults)))
    for i, b in enumerate(bonuses):
        for j, m in enumerate(mults):
            vals = agg.get((b, m), [0.0])
            grid[i, j] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid, cmap="RdYlGn", aspect="auto",
                   vmin=grid.min(), vmax=grid.max())
    ax.set_xticks(range(len(mults)));   ax.set_xticklabels(mults)
    ax.set_yticks(range(len(bonuses))); ax.set_yticklabels(bonuses)
    ax.set_xlabel("DYN_MULTIPLIER")
    ax.set_ylabel("DYNAMIC_BONUS")
    ax.set_title("IMP-09: Dynamic Reward Sensitivity")
    plt.colorbar(im, ax=ax, label="Mean reward")
    for i in range(len(bonuses)):
        for j in range(len(mults)):
            ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center",
                    fontsize=8)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "fig8_bonus_heatmap.png"))


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    logger.info(f"  Saved → {path}")


def run_all(out_dir: str = OUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Generating figures → {out_dir}")
    try:
        fig1_full_system(out_dir)
        fig2_smdp_ablation(out_dir)
        fig3_entropy_ablation(out_dir)
        fig4_transfer_learning(out_dir)
        fig5_rate_sensitivity(out_dir)
        fig8_bonus_heatmap(out_dir)
    except ImportError:
        logger.warning("matplotlib not available — skipping all figures")
    logger.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=RESULTS)
    parser.add_argument("--out_dir",     default=OUT_DIR)
    args = parser.parse_args()
    RESULTS = args.results_dir
    run_all(args.out_dir)
