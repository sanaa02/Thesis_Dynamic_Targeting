#!/usr/bin/env python3

"""
Dynamic Success Analysis Plots
==============================

Reads:
    episode_summaries.log

Creates:
    dyn_01_avg.png
    dyn_02_avg_max.png
    dyn_03_avg_p90.png
    dyn_04_avg_p95.png
    dyn_05_avg_highlights.png
    dyn_06_running_max.png
    dyn_07_boxplot_bins.png

Goal:
    Help determine whether averaging is hiding important high-success episodes.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "episode_summaries.log"

# --------------------------------------------------
# Load data
# --------------------------------------------------

episodes = []
dyn_sucs = []

pattern = re.compile(
    r"EPISODE\s+(\d+).*?dyn_suc=(\d+)%"
)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            episodes.append(int(m.group(1)))
            dyn_sucs.append(float(m.group(2)))

df = pd.DataFrame({
    "episode": episodes,
    "dyn_suc": dyn_sucs
})

if len(df) == 0:
    raise RuntimeError("No dynamic success data found.")

# --------------------------------------------------
# Automatic bin size
# --------------------------------------------------

N = len(df)

if N < 1000:
    BIN = 25
elif N < 5000:
    BIN = 50
elif N < 10000:
    BIN = 100
else:
    BIN = 200

print(f"Loaded {N} episodes")
print(f"Using BIN={BIN}")

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def p90(x):
    return np.percentile(x, 90)

def p95(x):
    return np.percentile(x, 95)

# --------------------------------------------------
# Binned stats
# --------------------------------------------------

binned = (
    df.groupby(df.index // BIN)
      .agg(
          episode=("episode", "mean"),
          mean=("dyn_suc", "mean"),
          max=("dyn_suc", "max"),
          p90=("dyn_suc", p90),
          p95=("dyn_suc", p95),
          std=("dyn_suc", "std"),
      )
)

# --------------------------------------------------
# 1. Average only
# --------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    binned["episode"],
    binned["mean"],
    linewidth=3,
    marker="o"
)

plt.title(f"Dynamic Success Average ({BIN}-Episode Bins)")
plt.xlabel("Episode")
plt.ylabel("Success (%)")
plt.ylim(0, 100)
plt.grid(True)
plt.tight_layout()
plt.savefig("dyn_01_avg.png", dpi=300)
plt.close()

# --------------------------------------------------
# 2. Average + Max
# --------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    binned["episode"],
    binned["mean"],
    linewidth=3,
    label="Average"
)

plt.plot(
    binned["episode"],
    binned["max"],
    linewidth=2,
    linestyle="--",
    label="Maximum"
)

plt.title("Dynamic Success: Average vs Maximum")
plt.xlabel("Episode")
plt.ylabel("Success (%)")
plt.ylim(0, 100)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("dyn_02_avg_max.png", dpi=300)
plt.close()

# --------------------------------------------------
# 3. Average + P90
# --------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    binned["episode"],
    binned["mean"],
    linewidth=3,
    label="Average"
)

plt.plot(
    binned["episode"],
    binned["p90"],
    linewidth=2,
    linestyle="--",
    label="90th Percentile"
)

plt.title("Dynamic Success: Average vs P90")
plt.xlabel("Episode")
plt.ylabel("Success (%)")
plt.ylim(0, 100)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("dyn_03_avg_p90.png", dpi=300)
plt.close()

# --------------------------------------------------
# 4. Average + P95
# --------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    binned["episode"],
    binned["mean"],
    linewidth=3,
    label="Average"
)

plt.plot(
    binned["episode"],
    binned["p95"],
    linewidth=2,
    linestyle="--",
    label="95th Percentile"
)

plt.title("Dynamic Success: Average vs P95")
plt.xlabel("Episode")
plt.ylabel("Success (%)")
plt.ylim(0, 100)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("dyn_04_avg_p95.png", dpi=300)
plt.close()

# --------------------------------------------------
# 5. Average + highlight exceptional episodes
# --------------------------------------------------

# Automatic threshold:
# top 10% of episodes

highlight_threshold = np.percentile(
    df["dyn_suc"],
    90
)

good = df[
    df["dyn_suc"] >= highlight_threshold
]

plt.figure(figsize=(12, 6))

plt.plot(
    binned["episode"],
    binned["mean"],
    linewidth=3,
    label="Average"
)

plt.scatter(
    good["episode"],
    good["dyn_suc"],
    s=20,
    alpha=0.8,
    label=f"Top 10% Episodes (>{highlight_threshold:.1f}%)"
)

plt.title("Dynamic Success with High-Performing Episodes")
plt.xlabel("Episode")
plt.ylabel("Success (%)")
plt.ylim(0, 100)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("dyn_05_avg_highlights.png", dpi=300)
plt.close()

# --------------------------------------------------
# 6. Running max
# --------------------------------------------------

df["running_max"] = df["dyn_suc"].cummax()

plt.figure(figsize=(12, 6))

plt.plot(
    df["episode"],
    df["running_max"],
    linewidth=3
)

plt.title("Best Dynamic Success Seen So Far")
plt.xlabel("Episode")
plt.ylabel("Success (%)")
plt.ylim(0, 100)
plt.grid(True)
plt.tight_layout()
plt.savefig("dyn_06_running_max.png", dpi=300)
plt.close()

# --------------------------------------------------
# 7. Boxplot per bin
# --------------------------------------------------

groups = [
    group["dyn_suc"].values
    for _, group in df.groupby(df.index // BIN)
]

positions = binned["episode"]

plt.figure(figsize=(14, 6))

plt.boxplot(
    groups,
    positions=positions,
    widths=BIN * 0.6,
    showfliers=True
)

plt.title("Dynamic Success Distribution per Bin")
plt.xlabel("Episode")
plt.ylabel("Success (%)")
plt.ylim(0, 100)
plt.grid(True)
plt.tight_layout()
plt.savefig("dyn_07_boxplot_bins.png", dpi=300)
plt.close()

print("\nSaved:")
print("  dyn_01_avg.png")
print("  dyn_02_avg_max.png")
print("  dyn_03_avg_p90.png")
print("  dyn_04_avg_p95.png")
print("  dyn_05_avg_highlights.png")
print("  dyn_06_running_max.png")
print("  dyn_07_boxplot_bins.png")

print("\nRecommended viewing order:")
print("  1) dyn_05_avg_highlights.png")
print("  2) dyn_03_avg_p90.png")
print("  3) dyn_04_avg_p95.png")
print("  4) dyn_02_avg_max.png")