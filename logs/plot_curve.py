#!/usr/bin/env python3

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "episode_summaries.log"

# --------------------------------------------------
# Read data
# --------------------------------------------------

episodes = []
rewards = []
dyn_sucs = []

pattern = re.compile(
    r"EPISODE\s+(\d+).*?reward=([+-]?\d+(?:\.\d+)?).*?dyn_suc=(\d+)%"
)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            episodes.append(int(m.group(1)))
            rewards.append(float(m.group(2)))
            dyn_sucs.append(float(m.group(3)))

df = pd.DataFrame({
    "episode": episodes,
    "reward": rewards,
    "dyn_suc": dyn_sucs
})

if len(df) == 0:
    raise RuntimeError("No data found")

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

print(f"Episodes : {N}")
print(f"Using bin size = {BIN}")

# --------------------------------------------------
# Bin statistics
# --------------------------------------------------

binned = (
    df.groupby(df.index // BIN)
      .agg({
          "episode": "mean",

          "reward": ["mean", "std", "min", "max"],

          "dyn_suc": ["mean", "std", "min", "max"]
      })
)

binned.columns = [
    "_".join(col).strip()
    for col in binned.columns.values
]

# --------------------------------------------------
# 1. Reward averaged
# --------------------------------------------------

plt.figure(figsize=(12,6))

plt.plot(
    binned["episode_mean"],
    binned["reward_mean"],
    linewidth=3,
    marker="o"
)

plt.title(f"Reward (Average of {BIN} Episodes)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.tight_layout()
plt.savefig("reward_avg.png", dpi=300)

# --------------------------------------------------
# 2. Dynamic success averaged
# --------------------------------------------------

plt.figure(figsize=(12,6))

plt.plot(
    binned["episode_mean"],
    binned["dyn_suc_mean"],
    linewidth=3,
    marker="o"
)

plt.ylim(0,100)

plt.title(f"Dynamic Success (Average of {BIN} Episodes)")
plt.xlabel("Episode")
plt.ylabel("Success (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig("dyn_success_avg.png", dpi=300)

# --------------------------------------------------
# 3. Reward + std band
# --------------------------------------------------

plt.figure(figsize=(12,6))

plt.plot(
    binned["episode_mean"],
    binned["reward_mean"],
    linewidth=3
)

plt.fill_between(
    binned["episode_mean"],
    binned["reward_mean"] - binned["reward_std"],
    binned["reward_mean"] + binned["reward_std"],
    alpha=0.25
)

plt.title("Reward Trend with Variability")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.tight_layout()
plt.savefig("reward_std_band.png", dpi=300)

# --------------------------------------------------
# 4. Dynamic success + std band
# --------------------------------------------------

plt.figure(figsize=(12,6))

plt.plot(
    binned["episode_mean"],
    binned["dyn_suc_mean"],
    linewidth=3
)

plt.fill_between(
    binned["episode_mean"],
    binned["dyn_suc_mean"] - binned["dyn_suc_std"],
    binned["dyn_suc_mean"] + binned["dyn_suc_std"],
    alpha=0.25
)

plt.ylim(0,100)

plt.title("Dynamic Success Trend with Variability")
plt.xlabel("Episode")
plt.ylabel("Success (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig("dyn_success_std_band.png", dpi=300)

# --------------------------------------------------
# 5. Compare trends
# --------------------------------------------------

reward_norm = (
    binned["reward_mean"] -
    binned["reward_mean"].min()
)

reward_norm /= (
    binned["reward_mean"].max() -
    binned["reward_mean"].min()
)

dyn_norm = binned["dyn_suc_mean"] / 100.0

plt.figure(figsize=(12,6))

plt.plot(
    binned["episode_mean"],
    reward_norm,
    linewidth=3,
    label="Reward"
)

plt.plot(
    binned["episode_mean"],
    dyn_norm,
    linewidth=3,
    label="Dynamic Success"
)

plt.title("Normalized Learning Curves")
plt.xlabel("Episode")
plt.ylabel("Normalized Performance")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("combined_normalized.png", dpi=300)

plt.show()