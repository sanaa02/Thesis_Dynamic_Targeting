#!/usr/bin/env python3
"""
Plot key metrics from episodes.jsonl vs. global_step.
Saves the figure as 'episode_analysis.png'.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# 1. Read the data
# ----------------------------------------------------------------------
data = []
with open("/home/sanaa/Videos/Thesis_Dynamic_Targeting_copy2/logs/episodes.jsonl", "r") as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

# Sort by global_step just in case
data.sort(key=lambda x: x["global_step"])

# Extract x‑axis and metrics
steps = [d["global_step"] for d in data]
rewards = [d["total_reward"] for d in data]
n_imaged = [d["n_imaged"] for d in data]
n_dyn_imaged = [d["n_dyn_imaged"] for d in data]
n_dyn_detected = [d["n_dyn_detected"] for d in data]
n_missed = [d["n_missed_events"] for d in data]
n_cloud_free = [d["n_cloud_free"] for d in data]
n_cloudy = [d["n_cloudy"] for d in data]
battery = [d["battery_end_pct"] for d in data]
slew_deg = [d["total_slew_deg"] for d in data]
ep_len = [d["ep_len"] for d in data]

# Action counts
static = [d["action_counts"].get("static", 0) for d in data]
dynamic = [d["action_counts"].get("dynamic", 0) for d in data]
drift = [d["action_counts"].get("drift", 0) for d in data]

# ----------------------------------------------------------------------
# 2. Create the figure
# ----------------------------------------------------------------------
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
fig.suptitle("Episode Metrics vs. Global Step", fontsize=16)

# (0,0) Total reward
axes[0,0].plot(steps, rewards, 'b-', alpha=0.7, linewidth=0.8)
axes[0,0].set_ylabel("Total Reward")
axes[0,0].grid(True, linestyle=':')

# (0,1) n_imaged & related
axes[0,1].plot(steps, n_imaged, label="n_imaged", alpha=0.7)
axes[0,1].plot(steps, n_dyn_imaged, label="n_dyn_imaged", alpha=0.7)
axes[0,1].plot(steps, n_dyn_detected, label="n_dyn_detected", alpha=0.7)
axes[0,1].plot(steps, n_missed, label="n_missed_events", alpha=0.7)
axes[0,1].set_ylabel("Counts")
axes[0,1].legend(loc='upper right', fontsize='x-small')
axes[0,1].grid(True, linestyle=':')

# (0,2) Cloud stats
axes[0,2].plot(steps, n_cloud_free, label="Cloud‑free", alpha=0.7)
axes[0,2].plot(steps, n_cloudy, label="Cloudy", alpha=0.7)
axes[0,2].set_ylabel("Cloud counts")
axes[0,2].legend()
axes[0,2].grid(True, linestyle=':')

# (1,0) Battery end %
axes[1,0].plot(steps, battery, 'g-', alpha=0.7)
axes[1,0].set_ylabel("Battery end (%)")
axes[1,0].set_ylim(0, 105)
axes[1,0].grid(True, linestyle=':')

# (1,1) Total slew (deg)
axes[1,1].plot(steps, slew_deg, 'r-', alpha=0.7)
axes[1,1].set_ylabel("Total slew (deg)")
axes[1,1].grid(True, linestyle=':')

# (1,2) Episode length
axes[1,2].plot(steps, ep_len, 'm-', alpha=0.7)
axes[1,2].set_ylabel("Episode length")
axes[1,2].grid(True, linestyle=':')

# (2,0) Action counts (stacked area)
axes[2,0].stackplot(steps, static, dynamic, drift,
                    labels=["static", "dynamic", "drift"],
                    alpha=0.7)
axes[2,0].set_ylabel("Action counts")
axes[2,0].legend(loc='upper right')
axes[2,0].grid(True, linestyle=':')

# (2,1) Rolling average of reward (optional)
window = 20
if len(steps) >= window:
    roll_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    roll_steps = steps[window-1:]
    axes[2,1].plot(steps, rewards, 'b-', alpha=0.3, linewidth=0.5, label="raw")
    axes[2,1].plot(roll_steps, roll_avg, 'r-', linewidth=1.5, label=f"{window}-ep rolling avg")
    axes[2,1].set_ylabel("Total Reward")
    axes[2,1].legend()
    axes[2,1].grid(True, linestyle=':')
else:
    axes[2,1].text(0.5, 0.5, "Not enough data for rolling average", 
                   ha='center', va='center', transform=axes[2,1].transAxes)
    axes[2,1].set_ylabel("Total Reward")

# (2,2) Empty or extra – we can put a summary or leave blank
axes[2,2].axis('off')
# Optionally add some info text
textstr = f"Total episodes: {len(data)}\n"
textstr += f"Global step range: {steps[0]} – {steps[-1]}"
axes[2,2].text(0.5, 0.5, textstr, ha='center', va='center', fontsize=12,
               transform=axes[2,2].transAxes)

# Adjust layout and save
plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig("episode_analysis.png", dpi=150)
print("Saved figure as episode_analysis.png")
# If you want to display the plot interactively, uncomment:
# plt.show()