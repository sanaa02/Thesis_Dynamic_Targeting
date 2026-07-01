#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Load episodes.jsonl
# ----------------------------
rewards = []
episodes = []

with open("logs/episodes.jsonl", "r") as f:
    for line in f:
        if not line.strip():
            continue

        row = json.loads(line)

        # If training was restarted and episode numbering resets,
        # create a continuous episode index.
        episodes.append(len(episodes) + 1)
        rewards.append(row["total_reward"])

df = pd.DataFrame({
    "episode": episodes,
    "reward": rewards
})

# ----------------------------
# Moving average
# ----------------------------
window = 10
df["reward_ma"] = df["reward"].rolling(
    window=window,
    min_periods=1
).mean()

# ----------------------------
# Linear trend
# ----------------------------
z = np.polyfit(df["episode"], df["reward"], 1)
trend = np.poly1d(z)

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(12, 6))

plt.plot(
    df["episode"],
    df["reward"],
    alpha=0.4,
    label="Episode Reward"
)

plt.plot(
    df["episode"],
    df["reward_ma"],
    linewidth=3,
    label=f"{window}-Episode Moving Average"
)

plt.plot(
    df["episode"],
    trend(df["episode"]),
    linestyle="--",
    linewidth=2,
    label="Linear Trend"
)

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Reward Convergence")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("reward_convergence.png", dpi=300)
plt.show()

print(f"Trend slope: {z[0]:.6f}")

if z[0] > 0:
    print("Overall trend: UPWARD")
elif z[0] < 0:
    print("Overall trend: DOWNWARD")
else:
    print("Overall trend: FLAT")