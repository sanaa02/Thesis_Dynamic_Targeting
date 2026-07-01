#!/usr/bin/env python3

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


# ==============================
# SETTINGS
# ==============================

ROLLING_WINDOW = 50
OUTPUT_DIR = "training_plots"


# ==============================
# LOAD CSV
# ==============================

if len(sys.argv) < 2:
    print("Usage: python plot_training_results.py training.csv")
    sys.exit(1)

csv_file = sys.argv[1]

df = pd.read_csv(csv_file)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loaded {len(df)} training points")


# ==============================
# Helper
# ==============================

def save_plot(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print("saved:", path)


def plot(x, y, title, xlabel, ylabel, filename):

    plt.figure(figsize=(10,5))
    plt.plot(df[x], df[y])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    save_plot(filename)



# ==============================
# 1 Reward
# ==============================

plot(
    "step",
    "reward",
    "Episode Reward",
    "Training Steps",
    "Reward",
    "reward.png"
)


# Rolling reward

plt.figure(figsize=(10,5))

plt.plot(
    df.step,
    df.reward.rolling(ROLLING_WINDOW).mean()
)

plt.title(
    f"Reward Rolling Average ({ROLLING_WINDOW} episodes)"
)

plt.xlabel("Training Steps")
plt.ylabel("Average Reward")

save_plot("reward_rolling.png")



# ==============================
# 2 Dynamic success
# ==============================

df["dyn_success"] = (
    df["n_dyn_imaged_ep"] /
    df["n_dyn_detected_ep"].replace(0,1)
) * 100


plt.figure(figsize=(10,5))

plt.plot(
    df.step,
    df.dyn_success
)

plt.title("Dynamic Target Success Rate")

plt.xlabel("Training Steps")
plt.ylabel("Dyn Success (%)")

save_plot("dyn_success.png")



# ==============================
# 3 Dynamic targets
# ==============================

plt.figure(figsize=(10,5))

plt.plot(
    df.step,
    df.n_dyn_detected_ep,
    label="Detected"
)

plt.plot(
    df.step,
    df.n_dyn_imaged_ep,
    label="Imaged"
)

plt.title("Dynamic Targets")

plt.xlabel("Training Steps")
plt.ylabel("Targets per Episode")

plt.legend()

save_plot("dynamic_targets.png")



# ==============================
# 4 Battery
# ==============================

plot(
    "step",
    "battery_end_pct",
    "Final Battery Level",
    "Training Steps",
    "Battery %",
    "battery.png"
)



# ==============================
# 5 Episode length
# ==============================

plot(
    "step",
    "ep_len",
    "Episode Length",
    "Training Steps",
    "Steps",
    "episode_length.png"
)



# ==============================
# 6 PPO losses
# ==============================

plt.figure(figsize=(10,5))

for col in [
    "loss",
    "pg_loss",
    "vf_loss"
]:

    if col in df:
        plt.plot(
            df.step,
            df[col],
            label=col
        )

plt.title("PPO Losses")

plt.xlabel("Training Steps")

plt.ylabel("Loss")

plt.legend()

save_plot("losses.png")



# ==============================
# 7 Entropy
# ==============================

plot(
    "step",
    "entropy",
    "Policy Entropy (Exploration)",
    "Training Steps",
    "Entropy",
    "entropy.png"
)



# ==============================
# 8 KL divergence
# ==============================

plot(
    "step",
    "kl",
    "KL Divergence",
    "Training Steps",
    "KL",
    "kl.png"
)



# ==============================
# 9 Explained variance
# ==============================

plot(
    "step",
    "explained_var",
    "Critic Explained Variance",
    "Training Steps",
    "Explained Variance",
    "explained_variance.png"
)



# ==============================
# 10 Clip fraction
# ==============================

plot(
    "step",
    "clip_frac",
    "PPO Clip Fraction",
    "Training Steps",
    "Clip Fraction",
    "clip_fraction.png"
)



# ==============================
# 11 Learning rate
# ==============================

plot(
    "step",
    "lr",
    "Learning Rate",
    "Training Steps",
    "LR",
    "learning_rate.png"
)



# ==============================
# 12 FPS
# ==============================

plot(
    "step",
    "fps",
    "Training Speed",
    "Training Steps",
    "FPS",
    "fps.png"
)



print("\nAll plots generated in:", OUTPUT_DIR)