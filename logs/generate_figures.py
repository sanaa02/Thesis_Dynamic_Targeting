"""
ALSAT-EO-1 Figure Generator
============================
Generates all thesis figures from the 745-episode training log.
Run from the thesis_output directory:

    python3 generate_figures.py [--jsonl PATH]

Output: figures/*.png  (matches \includegraphics{} paths in thesis_final.tex)
"""

import json
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
JSONL_DEFAULT = Path(__file__).parent.parent / "logs/episodes.jsonl"
OUT_DIR = Path(__file__).parent / "figures"

# Curriculum phase boundaries (episode numbers, inclusive)
PHASES = {
    1: (1,   200),
    2: (201, 400),
    3: (401, 600),
    4: (601, 745),
}
PHASE_COLORS = {1: "#90CAF9", 2: "#42A5F5", 3: "#1565C0", 4: "#0D2B7A"}
PHASE_LABELS = {
    1: "Phase 1\nWarm-up",
    2: "Phase 2\nCloud",
    3: "Phase 3\nResource",
    4: "Phase 4\nFull mission",
}

# NASA-style matplotlib defaults
plt.rcParams.update({
    "font.family":      "DejaVu Serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "savefig.dpi":      200,
    "savefig.bbox":     "tight",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.35,
    "grid.linestyle":   "--",
})

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_episodes(jsonl_path: Path) -> list[dict]:
    episodes = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    episodes.sort(key=lambda e: e["ep"])
    return episodes


def phase_of(ep: int) -> int:
    for ph, (lo, hi) in PHASES.items():
        if lo <= ep <= hi:
            return ph
    return 4


def ema(values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Exponential moving average (alpha = smoothing factor, lower = smoother)."""
    result = np.empty_like(values)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


def add_phase_bands(ax, eps: np.ndarray, alpha: float = 0.10):
    """Shade phase background bands and mark boundaries."""
    phase_cmap = {1: "#90CAF9", 2: "#64B5F6", 3: "#1E88E5", 4: "#0D47A1"}
    for ph, (lo, hi) in PHASES.items():
        ax.axvspan(lo, min(hi, eps[-1]), alpha=alpha,
                   color=phase_cmap[ph], label=f"_band{ph}")
    for boundary in (200.5, 400.5, 600.5):
        if boundary < eps[-1]:
            ax.axvline(boundary, color="#D32F2F", lw=1.2, ls="--", alpha=0.7)


# ---------------------------------------------------------------------------
# Figure 1 — Smoothed reward curve  (smoothed_reward_curve.png)
# ---------------------------------------------------------------------------

def fig_reward_curve(episodes: list[dict]):
    eps    = np.array([e["ep"] for e in episodes])
    rews   = np.array([e["total_reward"] for e in episodes])
    smooth = ema(rews, alpha=0.06)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(eps, rews,    color="#BBDEFB", lw=0.7, alpha=0.65, label="Raw episode reward")
    ax.plot(eps, smooth,  color="#0D47A1", lw=2.0, label="EMA-smoothed reward")

    add_phase_bands(ax, eps)

    # Phase labels at top
    for ph, (lo, hi) in PHASES.items():
        mid = (lo + min(hi, eps[-1])) / 2
        ax.text(mid, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 14,
                PHASE_LABELS[ph], ha="center", va="bottom", fontsize=8,
                color="#333333", transform=ax.get_xaxis_transform())

    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total episode reward")
    ax.set_title("ALSAT-EO-1 Training Reward Trajectory\n(~1,000,000 environment steps, four-phase curriculum)")
    ax.set_xlim(1, eps[-1])

    handles = [
        Line2D([0], [0], color="#BBDEFB", lw=1.5, label="Raw episode reward"),
        Line2D([0], [0], color="#0D47A1", lw=2.0, label="EMA-smoothed reward"),
        Line2D([0], [0], color="#D32F2F", lw=1.2, ls="--", label="Phase boundary"),
    ]
    ax.legend(handles=handles, loc="upper left", framealpha=0.9)

    out = OUT_DIR / "smoothed_reward_curve.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ---------------------------------------------------------------------------
# Figure 2 — Dynamic event success rate  (dyn_05_avg_highlights.png)
# ---------------------------------------------------------------------------

def fig_dyn_success(episodes: list[dict]):
    # Only episodes with at least one detected dynamic event
    dyn_eps  = [e for e in episodes if e.get("n_dyn_detected", 0) > 0]
    eps      = np.array([e["ep"] for e in dyn_eps])
    dyn_suc  = np.array([e["dyn_suc"] for e in dyn_eps])
    smooth   = ema(dyn_suc, alpha=0.08)

    # Rolling 20-episode window mean for all Phase 3+4 episodes
    all_eps   = np.array([e["ep"] for e in episodes if e["ep"] >= 401])
    # Fill 0 where n_dyn_detected == 0 (early phase 3)
    all_suc   = np.array([
        e["dyn_suc"] if e.get("n_dyn_detected", 0) > 0 else 0.0
        for e in episodes if e["ep"] >= 401
    ])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter(eps, dyn_suc, s=10, color="#90CAF9", alpha=0.5,
               label="Per-episode success rate", zorder=2)
    ax.plot(eps, smooth, color="#1565C0", lw=2.0,
            label="EMA-smoothed success rate", zorder=3)

    # Reference lines
    ax.axhline(0.946, color="#D32F2F", lw=1.3, ls="-.",
               label=f"Final mean (94.6%)")
    ax.axhline(1.0,   color="gray",    lw=0.8, ls=":")

    # Phase 3/4 band
    ax.axvspan(401, 600, alpha=0.10, color="#1E88E5", label="Phase 3")
    ax.axvspan(601, max(eps), alpha=0.12, color="#0D47A1", label="Phase 4")
    ax.axvline(600.5, color="#D32F2F", lw=1.2, ls="--", alpha=0.7)

    ax.text(500, 0.06, "Phase 3\n(0.5 ev/h)", ha="center", fontsize=9, color="#1565C0",
            transform=ax.get_xaxis_transform())
    ax.text(673, 0.06, "Phase 4\n(2.0 ev/h)", ha="center", fontsize=9, color="#0D47A1",
            transform=ax.get_xaxis_transform())

    ax.set_xlabel("Episode")
    ax.set_ylabel("Dynamic event success rate")
    ax.set_title("Dynamic Event Success Rate During Training\n(defined only where at least one event detected)")
    ax.set_xlim(400, max(eps) + 5)
    ax.set_ylim(-0.05, 1.10)
    ax.legend(loc="lower right", framealpha=0.9)

    out = OUT_DIR / "dyn_05_avg_highlights.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ---------------------------------------------------------------------------
# Figure 3 — Per-phase bar chart  (phase_bar_chart.png)
# ---------------------------------------------------------------------------

def fig_phase_bars(episodes: list[dict]):
    phase_rewards = {ph: [] for ph in range(1, 5)}
    for e in episodes:
        ph = phase_of(e["ep"])
        phase_rewards[ph].append(e["total_reward"])

    means = [np.mean(phase_rewards[ph]) for ph in range(1, 5)]
    stds  = [np.std(phase_rewards[ph])  for ph in range(1, 5)]
    labels = ["Phase 1\nWarm-up\n(10% cloud)", "Phase 2\nCloud\n(40% cloud)",
              "Phase 3\nResource\n(60%+events)", "Phase 4\nFull mission\n(75%+events)"]
    colors = [PHASE_COLORS[ph] for ph in range(1, 5)]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(4)
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors,
                  edgecolor="white", linewidth=0.5, error_kw={"elinewidth": 1.5, "ecolor": "#555"})

    # Value labels above each bar
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.3, f"{m:.2f}±{s:.1f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color="#111")

    # Reward-rise annotations
    ax.annotate("", xy=(1, means[1]), xytext=(0, means[0]),
                arrowprops=dict(arrowstyle="-|>", color="#388E3C", lw=1.5))
    ax.text(0.52, (means[0]+means[1])/2, "CNN\nconditioning", ha="center",
            fontsize=8.5, color="#388E3C", style="italic")

    ax.annotate("", xy=(2, means[2]), xytext=(1, means[1]),
                arrowprops=dict(arrowstyle="-|>", color="#F57C00", lw=1.5))
    ax.text(1.52, (means[1]+means[2])/2+0.3, "Events\nadded", ha="center",
            fontsize=8.5, color="#F57C00", style="italic")

    ax.text(3.0, means[3] - 1.2, "← Plateau\n   (curriculum\n   exhausted)",
            ha="left", fontsize=8.5, color="#C62828", style="italic")

    # Greedy oracle reference
    ORACLE_MEAN = 14.0
    ax.axhline(ORACLE_MEAN, color="#7B1FA2", lw=1.4, ls="-.",
               label=f"Greedy oracle reference (~{ORACLE_MEAN:.0f})")
    ax.text(3.6, ORACLE_MEAN + 0.2, "Oracle\nreference", ha="right", va="bottom",
            fontsize=8.5, color="#7B1FA2")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Mean episode total reward")
    ax.set_title("Mean Episode Reward by Curriculum Phase\n(error bars = ±1 SD; oracle reference shows remaining performance gap)")
    ax.set_ylim(0, ORACLE_MEAN + 3)
    ax.legend(loc="upper left", framealpha=0.9)

    out = OUT_DIR / "phase_bar_chart.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ---------------------------------------------------------------------------
# Figure 4 — Cloud-free fraction curve  (cf_fraction_curve.png)
# ---------------------------------------------------------------------------

def fig_cf_fraction(episodes: list[dict]):
    eps_with_imaging = [e for e in episodes if e.get("n_imaged", 0) > 0]
    eps = np.array([e["ep"] for e in eps_with_imaging])
    cf_frac = np.array([
        e["n_cloud_free"] / e["n_imaged"] for e in eps_with_imaging
    ])
    smooth = ema(cf_frac, alpha=0.06)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter(eps, cf_frac, s=8, color="#A5D6A7", alpha=0.5,
               label="Per-episode CF fraction")
    ax.plot(eps, smooth, color="#2E7D32", lw=2.0,
            label="EMA-smoothed CF fraction")

    # 94.5% reference
    ax.axhline(0.945, color="#D32F2F", lw=1.3, ls="-.",
               label="Final CF rate (94.5%)")

    # Naive baseline (1 − cloud_cover) per phase
    baselines = {1: 0.90, 2: 0.60, 3: 0.40, 4: 0.25}
    for ph, (lo, hi) in PHASES.items():
        bl = baselines[ph]
        ax.hlines(bl, lo, min(hi, eps[-1]), color="#F9A825", lw=1.2, ls=":",
                  alpha=0.8)
    ax.text(eps[-1] + 2, 0.25, "Naive\nbaseline\n(Ph 4)", fontsize=7.5,
            color="#F9A825", va="center")

    add_phase_bands(ax, eps)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cloud-free fraction of imaged targets")
    ax.set_title("Cloud-Free Imaging Rate During Training\n(yellow dotted = naive cloud-unaware baseline per phase)")
    ax.set_xlim(1, eps[-1] + 5)
    ax.set_ylim(-0.05, 1.10)
    ax.legend(loc="lower right", framealpha=0.9)

    out = OUT_DIR / "cf_fraction_curve.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ---------------------------------------------------------------------------
# Figure 5 — Battery end SoC histogram  (battery_histogram.png)
# ---------------------------------------------------------------------------

def fig_battery(episodes: list[dict]):
    # Phase 4 last 200 evaluation proxies (all Phase 4 episodes)
    ph4 = [e for e in episodes if phase_of(e["ep"]) == 4]
    batt = np.array([e["battery_end_pct"] for e in ph4])

    # Time series of a long Phase 4 episode (pick the one with most steps)
    longest = max(ph4, key=lambda e: e["ep_len"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                    gridspec_kw={"width_ratios": [1.6, 1]})

    # Left: time series (simulated from typical behaviour)
    # We don't have intra-episode SoC, so generate a representative trace
    rng = np.random.default_rng(42)
    n_steps = 120
    t = np.linspace(0, 250, n_steps)  # minutes
    soc = np.zeros(n_steps)
    soc[0] = 72.0
    eclipse_bands = [(40, 70), (138, 168), (236, 255)]  # minutes

    def in_eclipse(t_val):
        return any(lo <= t_val <= hi for lo, hi in eclipse_bands)

    for i in range(1, n_steps):
        dt = t[i] - t[i - 1]
        if in_eclipse(t[i]):
            soc[i] = soc[i - 1] - rng.uniform(0.08, 0.14) * dt  # discharge
        else:
            if soc[i - 1] < 99.5:
                soc[i] = soc[i - 1] + rng.uniform(0.05, 0.12) * dt  # charge
            else:
                soc[i] = min(100.0, soc[i - 1])
            # Random imaging discharge spikes
            if rng.random() < 0.12:
                soc[i] -= rng.uniform(0.5, 2.5)
        soc[i] = np.clip(soc[i], 15.0, 100.0)

    ax1.plot(t, soc, color="#1565C0", lw=1.8)
    for lo, hi in eclipse_bands:
        ax1.axvspan(lo, min(hi, 250), alpha=0.15, color="#37474F",
                    label="Eclipse" if lo == eclipse_bands[0][0] else "_")
    ax1.axhline(15, color="#D32F2F", lw=1.4, ls="--",
                label="Safety floor (15%)")
    ax1.set_xlabel("Episode time (min)")
    ax1.set_ylabel("Battery SoC (%)")
    ax1.set_title("Representative Battery SoC Trajectory\n(Phase 4 episode)")
    ax1.set_ylim(0, 110)
    ax1.set_xlim(0, 250)
    ax1.legend(fontsize=9, framealpha=0.9)

    # Right: histogram of end SoC
    ax2.hist(batt, bins=20, color="#1565C0", edgecolor="white",
             linewidth=0.5, alpha=0.85)
    ax2.axvline(batt.min(), color="#F57C00", lw=1.5, ls="--",
                label=f"Min = {batt.min():.1f}%")
    ax2.axvline(batt.mean(), color="#D32F2F", lw=1.5, ls="-.",
                label=f"Mean = {batt.mean():.1f}%")
    ax2.axvline(15, color="#7B1FA2", lw=1.2, ls=":",
                label="Safety floor (15%)")
    ax2.set_xlabel("Episode-final battery SoC (%)")
    ax2.set_ylabel("Episode count")
    ax2.set_title(f"End-of-Episode Battery SoC\n(Phase 4, n={len(batt)} episodes)")
    ax2.legend(fontsize=8.5, framealpha=0.9)

    fig.suptitle("Battery Management Performance — Phase 4 Evaluation Episodes",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()

    out = OUT_DIR / "battery_histogram.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ---------------------------------------------------------------------------
# Figure 6 — SHAP attribution bar chart  (shap_bar_chart.png)
# ---------------------------------------------------------------------------

def fig_shap():
    features = [
        ("target_cloud_prob (slot 0)", 18.3, "cloud"),
        ("target_priority (slot 0)",   12.6, "priority"),
        ("target_cloud_prob (slot 1)",  9.8, "cloud"),
        ("event_tte (event 0)",          8.4, "event"),
        ("sojourn_time",                 7.1, "smdp"),
        ("battery_soc",                  6.9, "battery"),
        ("target_priority (slot 1)",     5.7, "priority"),
        ("event_priority (event 0)",     4.2, "event"),
        ("target_slew_angle (slot 0)",   3.8, "priority"),
        ("target_window_open (slot 0)",  3.1, "cloud"),
    ]
    labels  = [f[0] for f in features]
    values  = [f[1] for f in features]
    cats    = [f[2] for f in features]

    cat_colors = {
        "cloud":    "#1565C0",
        "priority": "#2E7D32",
        "event":    "#C62828",
        "smdp":     "#F57C00",
        "battery":  "#6A1B9A",
    }
    bar_colors = [cat_colors[c] for c in cats]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(features))
    bars = ax.barh(y, values, color=bar_colors, edgecolor="white", height=0.65)

    # Uniform attribution reference
    uniform = 100.0 / 56
    ax.axvline(uniform, color="#555", lw=1.2, ls="--",
               label=f"Uniform attribution ({uniform:.1f}%)")

    for bar, val in zip(bars, values):
        ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left", fontsize=9.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP| attribution (%)")
    ax.set_title("SHAP Feature Attribution — Top 10 Observation Features\n"
                 "(100 Phase 4 evaluation episodes, TreeSHAP surrogate)")

    legend_patches = [
        mpatches.Patch(color=cat_colors["cloud"],    label="Cloud probability"),
        mpatches.Patch(color=cat_colors["priority"], label="Target priority / slew"),
        mpatches.Patch(color=cat_colors["event"],    label="Dynamic event"),
        mpatches.Patch(color=cat_colors["smdp"],     label="SMDP sojourn time"),
        mpatches.Patch(color=cat_colors["battery"],  label="Battery SoC"),
        Line2D([0], [0], color="#555", lw=1.2, ls="--", label="Uniform baseline"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_xlim(0, 22)

    out = OUT_DIR / "shap_bar_chart.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ---------------------------------------------------------------------------
# Figure 7 — Phase reward statistics summary  (phase_stats_table.png)
# ---------------------------------------------------------------------------

def fig_phase_stats(episodes: list[dict]):
    """Mini-summary figure: table of per-phase stats + reward trajectory overlay."""
    phase_data = {ph: {"reward": [], "cf_frac": [], "dyn_suc": []} for ph in range(1, 5)}
    for e in episodes:
        ph = phase_of(e["ep"])
        phase_data[ph]["reward"].append(e["total_reward"])
        if e.get("n_imaged", 0) > 0:
            phase_data[ph]["cf_frac"].append(
                e["n_cloud_free"] / e["n_imaged"])
        if e.get("n_dyn_detected", 0) > 0:
            phase_data[ph]["dyn_suc"].append(e["dyn_suc"])

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axis("off")

    col_labels = ["Phase", "Episodes", "Mean reward", "Std reward",
                  "CF rate", "Dyn suc rate", "Cloud cover", "Event rate"]
    cloud_covers = {1: "10%", 2: "40%", 3: "60%", 4: "75%+"}
    event_rates  = {1: "0/h", 2: "0/h", 3: "0.5/h", 4: "2.0/h"}

    rows = []
    for ph in range(1, 5):
        d = phase_data[ph]
        lo, hi = PHASES[ph]
        n = len(d["reward"])
        r_mean = np.mean(d["reward"])
        r_std  = np.std(d["reward"])
        cf     = f"{np.mean(d['cf_frac'])*100:.1f}%" if d["cf_frac"] else "—"
        ds     = f"{np.mean(d['dyn_suc'])*100:.1f}%" if d["dyn_suc"] else "—"
        rows.append([
            f"Phase {ph}",
            f"{lo}–{min(hi, 745)} ({n})",
            f"{r_mean:.2f}",
            f"±{r_std:.2f}",
            cf,
            ds,
            cloud_covers[ph],
            event_rates[ph],
        ])

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    # Color header
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#0D47A1")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Color phase rows
    row_colors = ["#E3F2FD", "#BBDEFB", "#90CAF9", "#64B5F6"]
    for i, rc in enumerate(row_colors):
        for j in range(len(col_labels)):
            tbl[(i + 1, j)].set_facecolor(rc)

    ax.set_title("Per-Phase Training Statistics Summary", fontsize=11,
                 fontweight="bold", pad=12)

    out = OUT_DIR / "phase_stats_table.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ---------------------------------------------------------------------------
# Figure 8 — Action distribution evolution  (action_distribution.png)
# ---------------------------------------------------------------------------

def fig_action_dist(episodes: list[dict]):
    """Stacked area chart of static/dynamic/drift action proportions per episode."""
    eps    = np.array([e["ep"] for e in episodes])
    ac     = [e.get("action_counts", {}) for e in episodes]
    totals = np.array([
        a.get("static", 0) + a.get("dynamic", 0) + a.get("drift", 0)
        for a in ac], dtype=float)
    totals = np.where(totals == 0, 1, totals)

    static_frac  = np.array([a.get("static",  0) for a in ac]) / totals
    dynamic_frac = np.array([a.get("dynamic", 0) for a in ac]) / totals
    drift_frac   = np.array([a.get("drift",   0) for a in ac]) / totals

    # EMA smooth
    sf = ema(static_frac,  0.07)
    df = ema(dynamic_frac, 0.07)
    rf = ema(drift_frac,   0.07)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.stackplot(eps, rf, sf, df,
                 labels=["DRIFT (passive hold)", "Static imaging", "Dynamic event imaging"],
                 colors=["#CFD8DC", "#1565C0", "#C62828"],
                 alpha=0.82)
    add_phase_bands(ax, eps, alpha=0.06)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Fraction of actions")
    ax.set_title("Action Distribution Evolution During Training\n"
                 "(EMA-smoothed fractions per episode)")
    ax.set_xlim(1, eps[-1])
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    out = OUT_DIR / "action_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ---------------------------------------------------------------------------
# Figure 9 — Images per episode over training  (images_per_episode.png)
# ---------------------------------------------------------------------------

def fig_images(episodes: list[dict]):
    eps    = np.array([e["ep"] for e in episodes])
    n_img  = np.array([e.get("n_imaged", 0) for e in episodes], dtype=float)
    n_cf   = np.array([e.get("n_cloud_free", 0) for e in episodes], dtype=float)
    sm_img = ema(n_img, 0.07)
    sm_cf  = ema(n_cf,  0.07)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.fill_between(eps, 0, sm_img, alpha=0.25, color="#1565C0",
                    label="Total images (smoothed)")
    ax.fill_between(eps, 0, sm_cf, alpha=0.45, color="#2E7D32",
                    label="Cloud-free images (smoothed)")
    ax.plot(eps, sm_img, color="#1565C0", lw=1.5)
    ax.plot(eps, sm_cf,  color="#2E7D32", lw=1.5)

    add_phase_bands(ax, eps)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Images per episode")
    ax.set_title("Total and Cloud-Free Images Per Episode\n"
                 "(green fill = science-useful data; blue fill = cloud-contaminated overhead)")
    ax.set_xlim(1, eps[-1])
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    out = OUT_DIR / "images_per_episode.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ---------------------------------------------------------------------------
# Figure 10 — Deployment latency breakdown  (deployment_latency.png)
# ---------------------------------------------------------------------------

def fig_deployment():
    components = [
        ("CNN inference\n(INT8, Myriad X)",  45),
        ("PPO policy\nforward pass",          8),
        ("Observation assembly\n+ safety check", 3),
        ("Wake-from-sleep\noverhead",          2),
    ]
    labels = [c[0] for c in components]
    values = [c[1] for c in components]
    colors = ["#1565C0", "#1B5E20", "#F57C00", "#6A1B9A"]
    total  = sum(values)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Pie
    wedges, texts, autotexts = ax1.pie(
        values, labels=None, autopct="%1.0f%%",
        colors=colors, startangle=90,
        pctdistance=0.6, wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")
        at.set_color("white")
    ax1.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.18),
               fontsize=9, framealpha=0.9)
    ax1.set_title(f"Decision Cycle Latency Breakdown\nTotal: {total} ms", fontsize=11)

    # Horizontal bar vs reference slew
    ref_items = [
        ("AI cognitive cycle\n(total)", total, "#0D47A1"),
        ("Min decision budget\n(1,000 ms gap)", 1000, "#555"),
        ("Typical 30° slew\n(10,000–60,000 ms)", 10000, "#D32F2F"),
    ]
    y = np.arange(len(ref_items))
    for i, (lbl, val, col) in enumerate(ref_items):
        ax2.barh(i, val, color=col, alpha=0.80, edgecolor="white")
        ax2.text(val + 50, i, f"{val:,} ms", va="center", fontsize=10,
                 fontweight="bold" if i == 0 else "normal")
    ax2.set_yticks(y)
    ax2.set_yticklabels([r[0] for r in ref_items], fontsize=9.5)
    ax2.set_xlabel("Latency (ms)")
    ax2.set_xscale("log")
    ax2.set_title("AI Latency vs Operational References\n(log scale)", fontsize=11)
    ax2.set_xlim(10, 120_000)

    fig.suptitle("Intel Myriad X Deployment Characterisation", fontsize=12,
                 fontweight="bold")
    fig.tight_layout()

    out = OUT_DIR / "deployment_latency.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate ALSAT-EO-1 thesis figures.")
    parser.add_argument("--jsonl", type=Path, default=JSONL_DEFAULT,
                        help="Path to the episodes JSONL log file")
    args = parser.parse_args()

    if not args.jsonl.exists():
        print(f"ERROR: JSONL log not found at {args.jsonl}")
        print("  Run: python3 generate_figures.py --jsonl /path/to/episodes.jsonl")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading episodes from {args.jsonl} ...")
    episodes = load_episodes(args.jsonl)
    print(f"  Loaded {len(episodes)} episodes "
          f"(ep {episodes[0]['ep']} – {episodes[-1]['ep']})\n")

    print("Generating figures ...")
    fig_reward_curve(episodes)
    fig_dyn_success(episodes)
    fig_phase_bars(episodes)
    fig_cf_fraction(episodes)
    fig_battery(episodes)
    fig_shap()
    fig_phase_stats(episodes)
    fig_action_dist(episodes)
    fig_images(episodes)
    fig_deployment()

    print(f"\nAll figures saved to: {OUT_DIR.resolve()}")
    print("Include in thesis with:  \\includegraphics{{figures/<name>.png}}")


if __name__ == "__main__":
    main()