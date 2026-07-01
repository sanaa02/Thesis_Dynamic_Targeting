
#!/usr/bin/env python3
"""
thesis_logger.py  --  ALSAT-EO-1  Thesis Jury Verification Logger
==================================================================
Saves the following for jury verification:

  results/verification/
    cloud_predictions.csv          â CNN forecast vs ground truth per target per episode
    cloud_patches/                 â PNG: actual MODIS patch + forecast/truth overlay
    event_decisions.jsonl          â every dynamic event: appeared, what agent did, outcome
    episode_summary.csv            â per-episode high-level stats
    algeria_map_ep<N>.png          â map showing events, decisions, satellite passes
    agent_decisions_ep<N>.png      â timeline: what the agent chose, reward, cloud state

Usage: import and pass the callback to model.learn():
    from thesis_logger import ThesisLogger
    cb = ThesisLogger(env=env, log_dir="results/verification", every_n_episodes=5)
    model.learn(..., callback=CallbackList([alsat_cb, cb]))
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ââ Try optional imports ââââââââââââââââââââââââââââââââââââââââââââââââââââââ
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    MPL_OK = True
except ImportError:
    MPL_OK = False
    logger.warning("[ThesisLogger] matplotlib not available â plots disabled")

try:
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_OK = True
except ImportError:
    SB3_OK = False
    class BaseCallback:
        def __init__(self, verbose=0): self.verbose = verbose
        def _on_step(self): return True

# Algeria bounding box
ALG_LAT_MIN, ALG_LAT_MAX = 18.9, 37.1
ALG_LON_MIN, ALG_LON_MAX = -8.7, 12.0

EVENT_COLORS = {
    "wildfire":   "#FF4500",
    "flood":      "#1E90FF",
    "earthquake": "#8B4513",
    "eruption":   "#9400D3",
    "plume":      "#808080",
}
EVENT_MARKERS = {
    "wildfire": "^", "flood": "v", "earthquake": "s",
    "eruption": "D", "plume": "P",
}


class ThesisLogger(BaseCallback):
    """
    SB3 callback that saves verification artefacts every N episodes.

    Parameters
    ----------
    log_dir          : where to save everything
    every_n_episodes : save detailed plots every N episodes (not every step)
    patches_dir      : path to your MODIS patches (for cloud imagery)
    """

    def __init__(self,
                 log_dir:          str = "results/verification",
                 every_n_episodes: int = 5,
                 patches_dir:      str = "data/modis_patches",
                 verbose:          int = 0):
        super().__init__(verbose=verbose)
        self.log_dir          = log_dir
        self.every_n          = every_n_episodes
        self.patches_dir      = patches_dir

        # Create output dirs
        for d in [log_dir, os.path.join(log_dir, "cloud_patches"),
                  os.path.join(log_dir, "maps"), os.path.join(log_dir, "timelines")]:
            os.makedirs(d, exist_ok=True)

        # Episode-level state
        self._ep         = 0
        self._step_in_ep = 0
        self._ep_actions: list = []        # (sim_time, action, reward, event_info)
        self._ep_events:  list = []        # events that appeared this episode
        self._ep_imaged:  list = []        # events successfully imaged
        self._ep_missed:  list = []        # events that expired unimaged
        self._ep_clouds:  list = []        # cloud forecast vs truth per step

        # Writers
        self._cloud_csv   = self._open_csv(
            os.path.join(log_dir, "cloud_predictions.csv"),
            ["episode", "date", "target_id", "target_name",
             "cnn_forecast", "ground_truth", "abs_error", "cloudy_thresh"])

        self._ep_csv      = self._open_csv(
            os.path.join(log_dir, "episode_summary.csv"),
            ["episode", "total_steps", "total_reward",
             "n_static_imaged", "n_dyn_detected", "n_dyn_imaged",
             "n_dyn_missed", "dyn_success_rate", "mean_cloud_error",
             "n_safety_vetoes"])

        self._event_log   = open(os.path.join(log_dir, "event_decisions.jsonl"), "a")

        self._n_vetoes = 0  # safety vetoes this episode

        logger.info(f"[ThesisLogger] Saving verification artefacts â {log_dir}/")
        print(f"\n[ThesisLogger] Jury verification active â {log_dir}/")

    # ââ SB3 hooks âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

    def _on_step(self) -> bool:
        self._step_in_ep += 1

        try:
            info   = (self.locals.get("infos") or [{}])[0]
            action = int((self.locals.get("actions") or [23])[0])
            reward = float((self.locals.get("rewards") or [0.0])[0])
        except Exception:
            info = {}; action = 23; reward = 0.0

        # ── Episode-end detection via Monitor's "episode" key ──────────────
        done = "episode" in info

        # Track safety vetoes
        if info.get("safety_vetoed"):
            self._n_vetoes += 1

        # Get satellite state
        sat, now = self._get_sat()
        if sat is not None:
            # Record cloud forecasts vs truth for all static targets
            self._record_cloud_state(sat, now, action, reward)
            # Record the action taken
            self._record_action(sat, now, action, reward, info)

        if done:
            self._on_episode_end(info)

        return True

    def _on_episode_end(self, info: dict):
        self._ep += 1
        ep_m  = info.get("episode_metrics", {})
        n_det = ep_m.get("n_dyn_detected", 0)
        n_dyn_clean = ep_m.get("n_dyn_imaged_clean", 0)
        n_static_clean = ep_m.get("n_static_imaged_clean", 0)
        n_rew = ep_m.get("total_reward",    0.0)

        dyn_rate = n_dyn_clean / max(1, n_det)
        cloud_err = (np.mean([c["abs_error"] for c in self._ep_clouds])
                     if self._ep_clouds else 0.0)

        # Write episode summary row
        self._ep_csv.writerow({
            "episode":         self._ep,
            "total_steps":     self._step_in_ep,
            "total_reward":    round(n_rew, 3),
            "n_static_imaged": n_static_clean,
            "n_dyn_detected":  n_det,
            "n_dyn_imaged":    n_dyn_clean,
            "n_dyn_missed":    n_det - n_dyn_clean,
            "dyn_success_rate": round(dyn_rate, 3),
            "mean_cloud_error": round(float(cloud_err), 4),
            "n_safety_vetoes": self._n_vetoes,
        })
        self._ep_csv_fh.flush()

        # Write cloud predictions for this episode
        self._flush_cloud_csv()

        # Every N episodes, save the detailed visualizations
        if self._ep % self.every_n == 0 and MPL_OK:
            self._save_cloud_patch_grid()
            self._save_algeria_map()
            self._save_decision_timeline()
            print(f"[ThesisLogger] Ep {self._ep}: saved cloud patches, map, timeline")

        # Reset episode state
        self._ep_actions  = []
        self._ep_events   = []
        self._ep_imaged   = []
        self._ep_missed   = []
        self._ep_clouds   = []
        self._n_vetoes    = 0
        self._step_in_ep  = 0

    # ââ Data collection 

    def _record_cloud_state(self, sat, now: float, action: int, reward: float):
        """Record CNN forecast vs ground truth for static targets this step."""
        try:
            scenario = sat.scenario
            date_str = getattr(scenario, "utc_init", "")[:10] or "unknown"
            for tid, tgt in enumerate(scenario.targets):
                truth    = float(getattr(tgt, "cloud_cover",          0.0))
                forecast = float(getattr(tgt, "cloud_cover_forecast",  truth))
                err      = abs(forecast - truth)
                self._ep_clouds.append({
                    "target_id":   tid,
                    "target_name": getattr(tgt, "name", f"T{tid:02d}"),
                    "cnn_forecast": round(forecast, 4),
                    "ground_truth": round(truth,    4),
                    "abs_error":    round(err,       4),
                    "date_str":     date_str,
                    "step":         self._step_in_ep,
                })
        except Exception:
            pass

    def _record_action(self, sat, now: float, action: int, reward: float, info: dict):
        """Record every agent decision with context."""
        try:
            N_STATIC = len(sat.scenario.targets)
            entry = {
                "sim_time_s": round(now, 0),
                "sim_time_h": round(now / 3600, 2),
                "action":     action,
                "reward":     round(reward, 4),
            }

            if action < N_STATIC:
                tgt = sat.scenario.targets[action]
                entry["type"]         = "static"
                entry["target_name"]  = getattr(tgt, "name", f"T{action:02d}")
                entry["cloud_truth"]  = round(float(getattr(tgt, "cloud_cover", 0)), 3)
                entry["cloud_fcst"]   = round(float(getattr(tgt, "cloud_cover_forecast", 0)), 3)
                entry["priority"]     = round(float(getattr(tgt, "priority", 0.5)), 3)
                entry["outcome"]      = "imaged" if reward > 0.01 else \
                                        ("cloudy" if getattr(tgt, "cloud_cover", 0) > 0.6
                                         else "no_access")

            elif action < N_STATIC + 3:
                slot = action - N_STATIC
                mgr  = getattr(sat, "_event_manager", None)
                evt  = None
                if mgr is not None:
                    slots = mgr.get_slots(sat, now)
                    evt   = slots[slot] if slot < len(slots) else None

                entry["type"]   = "dynamic"
                entry["slot"]   = slot
                if evt is not None:
                    entry["event_name"]   = evt.name
                    entry["event_type"]   = getattr(evt, "event_type",  "?")
                    entry["lat"]          = round(math.degrees(getattr(evt, "lat_rad", 0)), 3)
                    entry["lon"]          = round(math.degrees(getattr(evt, "lon_rad", 0)), 3)
                    entry["priority"]     = round(float(evt.priority), 3)
                    entry["cloud_truth"]  = round(float(evt.cloud_cover), 3)
                    entry["cloud_fcst"]   = round(float(evt.cloud_cover_forecast), 3)
                    entry["remaining_s"]  = round(max(0, evt.expiration_time - now), 0)
                    entry["outcome"]      = "imaged" if reward > 0.1 else \
                                           ("cloudy" if evt.cloud_cover > 0.6
                                            else "attempted_no_image")
                    # Track event appearance for map
                    self._ep_events.append({
                        "name":      evt.name,
                        "type":      entry["event_type"],
                        "lat":       entry["lat"],
                        "lon":       entry["lon"],
                        "priority":  entry["priority"],
                        "imaged":    reward > 0.1,
                        "cloud":     entry["cloud_truth"],
                        "appear_s":  float(evt.appearance_time),
                        "expire_s":  float(evt.expiration_time),
                        "imaged_at": now if reward > 0.1 else None,
                    })
                else:
                    entry["event_name"] = "empty_slot"
                    entry["outcome"]    = "empty_slot"

            else:
                entry["type"]    = "drift"
                entry["outcome"] = "drift"

            self._ep_actions.append(entry)

        except Exception as exc:
            logger.debug(f"[ThesisLogger] action record error: {exc}")

    # ââ Visualization âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

    def _save_cloud_patch_grid(self):
        """
        Save a PNG grid showing:
          - The actual MODIS patch for each target
          - CNN forecast vs ground truth overlaid
        """
        if not MPL_OK:
            return

        # Get last cloud state (most recent step's values)
        if not self._ep_clouds:
            return

        # Average forecast and truth per target over episode
        by_target = defaultdict(list)
        for c in self._ep_clouds:
            by_target[c["target_id"]].append(c)

        n_targets = len(by_target)
        if n_targets == 0:
            return

        ncols = 5
        nrows = math.ceil(n_targets / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 3.5, nrows * 3.5))
        fig.suptitle(f"Cloud Verification â Episode {self._ep}\n"
                     f"MODIS patch | CNN forecast vs ground truth",
                     fontsize=13, fontweight="bold")
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for tid in sorted(by_target.keys()):
            records = by_target[tid]
            name    = records[-1]["target_name"]
            fcst    = np.mean([r["cnn_forecast"]  for r in records])
            truth   = np.mean([r["ground_truth"]  for r in records])
            err     = abs(fcst - truth)

            ax = axes_flat[tid] if tid < len(axes_flat) else None
            if ax is None:
                continue

            # Try to load real patch
            patch_img = self._load_patch(name, records[-1].get("date_str"))
            if patch_img is not None:
                if patch_img.ndim == 2:
                    ax.imshow(patch_img, cmap="gray", vmin=0, vmax=1)
                elif patch_img.ndim == 3:
                    img_show = patch_img.transpose(1, 2, 0)
                    img_show = np.clip(img_show, 0, 1)
                    ax.imshow(img_show)
            else:
                # No patch - show synthetic cloud-fraction visualization
                cloud_vis = np.ones((64, 64, 3)) * 0.3  # dark background
                n_cloud = int(64 * 64 * truth)
                flat = cloud_vis.reshape(-1, 3)
                rng = np.random.default_rng(tid)
                idx = rng.choice(len(flat), n_cloud, replace=False)
                flat[idx] = [0.9, 0.9, 0.9]
                ax.imshow(cloud_vis)

            # Overlay text
            color_truth = "red"   if truth  > 0.6 else "green"
            color_fcst  = "red"   if fcst   > 0.6 else "lime"
            match = "â" if (truth > 0.6) == (fcst > 0.6) else "â"

            ax.set_title(f"{name}\nTruth: {truth:.2f}  CNN: {fcst:.2f}  err: {err:.2f} {match}",
                         fontsize=7,
                         color="green" if match == "â" else "red")
            # Truth bar on left edge
            ax.axhline(y=int((1 - truth) * 64), color=color_truth, linewidth=2,
                       label=f"truth={truth:.2f}")
            ax.axhline(y=int((1 - fcst) * 64), color=color_fcst, linewidth=2,
                       linestyle="--", label=f"cnn={fcst:.2f}")
            ax.axis("off")

        # Hide unused subplots
        for i in range(n_targets, len(axes_flat)):
            axes_flat[i].axis("off")

        plt.tight_layout()
        path = os.path.join(self.log_dir, "cloud_patches",
                            f"cloud_patch_ep{self._ep:04d}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    def _save_algeria_map(self):
        """
        Save an Algeria map showing:
          - Static target positions
          - Dynamic events (green=imaged, red=missed, size=priority)
          - Cloud cover color-coded
        """
        if not MPL_OK:
            return

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_facecolor("#E8F4FD")  # light blue (ocean/background)

        # Algeria bounding box
        ax.set_xlim(ALG_LON_MIN - 0.5, ALG_LON_MAX + 0.5)
        ax.set_ylim(ALG_LAT_MIN - 0.5, ALG_LAT_MAX + 0.5)
        ax.add_patch(plt.Rectangle(
            (ALG_LON_MIN, ALG_LAT_MIN),
            ALG_LON_MAX - ALG_LON_MIN, ALG_LAT_MAX - ALG_LAT_MIN,
            facecolor="#F5DEB3", edgecolor="brown", linewidth=1.5,
            alpha=0.6, zorder=0))

        # Static targets from satellite state
        sat, now = self._get_sat()
        if sat is not None:
            try:
                from bsk_rl.utils.orbital import lla2ecef
                import math as _m
                for tid, tgt in enumerate(sat.scenario.targets):
                    # Get lat/lon from r_LP_P
                    r = np.asarray(tgt.r_LP_P, dtype=float)
                    norm = np.linalg.norm(r)
                    if norm > 0:
                        lat = _m.degrees(_m.asin(r[2] / norm))
                        lon = _m.degrees(_m.atan2(r[1], r[0]))
                        cloud = float(getattr(tgt, "cloud_cover", 0.5))
                        cloud_color = plt.cm.RdYlGn(1 - cloud)
                        ax.scatter(lon, lat, s=100, marker="o",
                                   color=cloud_color, edgecolors="black",
                                   linewidths=0.8, zorder=3, alpha=0.85)
                        ax.annotate(getattr(tgt, "name", f"T{tid}")[:8],
                                    (lon, lat), fontsize=5,
                                    xytext=(3, 3), textcoords="offset points")
            except Exception:
                pass

        # Dynamic events
        seen_events = {}  # deduplicate
        for entry in self._ep_actions:
            if entry.get("type") == "dynamic" and "lat" in entry:
                key = entry.get("event_name", "?")
                if key not in seen_events:
                    seen_events[key] = entry

        for key, entry in seen_events.items():
            lat      = entry["lat"]
            lon      = entry["lon"]
            etype    = entry.get("event_type", "wildfire")
            imaged   = entry.get("outcome") == "imaged"
            priority = entry.get("priority", 0.8)
            cloud    = entry.get("cloud_truth", 0.0)

            marker  = EVENT_MARKERS.get(etype, "^")
            color   = "#00CC00" if imaged else "#CC0000"
            size    = 200 + priority * 300

            ax.scatter(lon, lat, s=size, marker=marker, color=color,
                       edgecolors="black", linewidths=1.5, zorder=5, alpha=0.9)
            ax.annotate(
                f"{etype[:4]}\ncld:{cloud:.2f}\n{'â' if imaged else 'â'}",
                (lon, lat), fontsize=6, ha="center",
                xytext=(0, 12), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="white", alpha=0.8, edgecolor=color))

        # Legend
        legend_elements = [
            mpatches.Patch(color="#00CC00", label="Event imaged â"),
            mpatches.Patch(color="#CC0000", label="Event missed â"),
            mpatches.Patch(color="#F5DEB3", edgecolor="brown", label="Algeria"),
            plt.scatter([], [], s=80, marker="o", color="green",
                        edgecolors="black", label="Target (clear)"),
            plt.scatter([], [], s=80, marker="o", color="red",
                        edgecolors="black", label="Target (cloudy)"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

        # Cloud colorbar
        sm = ScalarMappable(cmap="RdYlGn_r", norm=Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
        cbar.set_label("Cloud cover (static targets)", fontsize=8)

        ax.set_xlabel("Longitude (Â°E)", fontsize=10)
        ax.set_ylabel("Latitude (Â°N)", fontsize=10)
        ax.set_title(
            f"ALSAT-EO-1 Episode {self._ep} â Algeria Targeting Map\n"
            f"{len(seen_events)} dynamic events | "
            f"{sum(1 for e in seen_events.values() if e.get('outcome')=='imaged')} imaged",
            fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        path = os.path.join(self.log_dir, "maps",
                            f"algeria_map_ep{self._ep:04d}.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)

    def _save_decision_timeline(self):
        """
        Save a 3-panel timeline:
          - Top:    rewards per decision step
          - Middle: action type (static / dynamic / drift) colored bars
          - Bottom: cloud forecast accuracy (CNN vs truth) per step
        """
        if not MPL_OK or not self._ep_actions:
            return

        actions  = self._ep_actions
        times    = [a["sim_time_h"] for a in actions]
        rewards  = [a["reward"]     for a in actions]
        cum_r    = np.cumsum(rewards)
        types    = [a["type"]       for a in actions]

        # Cloud error per step (first target as proxy)
        cloud_steps = sorted(self._ep_clouds, key=lambda c: c["step"])
        # average across targets per step
        from collections import defaultdict as dd
        step_clouds = dd(list)
        for c in cloud_steps:
            step_clouds[c["step"]].append(c["abs_error"])
        step_nums   = sorted(step_clouds.keys())
        step_errors = [np.mean(step_clouds[s]) for s in step_nums]
        step_times  = [times[min(s - 1, len(times) - 1)] for s in step_nums]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10),
                                             gridspec_kw={"height_ratios": [2, 1, 1.5],
                                                          "hspace": 0.4})
        fig.suptitle(f"ALSAT-EO-1 Episode {self._ep} â Agent Decision Timeline\n"
                     f"total reward: {sum(rewards):+.2f}  |  "
                     f"steps: {len(actions)}  |  "
                     f"dynamic events imaged: "
                     f"{sum(1 for a in actions if a.get('outcome')=='imaged' and a['type']=='dynamic')}",
                     fontsize=12, fontweight="bold")

        # Panel 1: cumulative reward
        ax1.plot(times, cum_r, color="#2196F3", linewidth=2.5,
                 label=f"Cumulative reward = {cum_r[-1]:+.2f}")
        ax1.fill_between(times, cum_r, 0, alpha=0.1, color="#2196F3")
        ax1.axhline(0, color="black", linewidth=0.5, linestyle="--")
        # Mark DYN successes
        for a in actions:
            if a.get("outcome") == "imaged" and a["type"] == "dynamic":
                ax1.axvline(a["sim_time_h"], color="#4CAF50", alpha=0.7,
                            linewidth=1.5, linestyle=":")
        ax1.set_ylabel("Cumulative Reward")
        ax1.set_title("(a) Cumulative reward  [dotted line = DYN event imaged]")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Panel 2: action types
        color_map = {"static": "#4CAF50", "dynamic": "#FF9800", "drift": "#9E9E9E"}
        for a in actions:
            col = color_map.get(a["type"], "#9E9E9E")
            ax2.bar(a["sim_time_h"], 1, width=0.05,
                    color=col, alpha=0.85, linewidth=0)
        legend_h = [mpatches.Patch(color=v, label=k.capitalize())
                    for k, v in color_map.items()]
        ax2.legend(handles=legend_h, fontsize=8, ncol=3, loc="upper right")
        ax2.set_yticks([]); ax2.set_ylabel("Action type")
        ax2.set_title("(b) Action type per decision step")

        # Panel 3: cloud forecast error
        if step_times and step_errors:
            ax3.plot(step_times, step_errors, color="#F44336", linewidth=1.5,
                     label="Mean |CNN â truth|")
            ax3.fill_between(step_times, step_errors, 0,
                             alpha=0.2, color="#F44336")
            ax3.axhline(0.05, color="orange", linewidth=1, linestyle="--",
                        label="5% error threshold")
            ax3.set_ylabel("|CNN â truth|")
            ax3.set_title("(c) Cloud forecast error (CNN prediction vs MODIS ground truth)")
            ax3.legend(fontsize=9)
            ax3.set_ylim(0, max(0.3, max(step_errors) * 1.2))
            ax3.grid(True, alpha=0.3)

        ax3.set_xlabel("Simulation time (h)")

        path = os.path.join(self.log_dir, "timelines",
                            f"timeline_ep{self._ep:04d}.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)

    # ââ Helpers âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

    def _get_sat(self):
        """Walk the env stack to get the satellite and sim_time."""
        try:
            e = self.training_env
            while hasattr(e, "envs"):
                e = e.envs[0]
            while hasattr(e, "env"):
                e = e.env
            sat = getattr(e, "unwrapped", e).satellites[0]
            now = float(sat.simulator.sim_time)
            return sat, now
        except Exception:
            return None, 0.0

    def _load_patch(self, target_name: str, date_str: str) -> Optional[np.ndarray]:
        """Load a MODIS patch from the dated subdirectory."""
        if not date_str or date_str == "unknown":
            date_str = datetime.now().strftime("%Y-%m-%d")
        path = os.path.join(self.patches_dir, target_name, f"{date_str}.npy")
        if os.path.exists(path):
            try:
                return np.load(path).astype(np.float32)
            except Exception:
                pass
        # Try any date for this target
        tgt_dir = os.path.join(self.patches_dir, target_name)
        if os.path.isdir(tgt_dir):
            files = sorted(f for f in os.listdir(tgt_dir) if f.endswith(".npy"))
            if files:
                try:
                    return np.load(os.path.join(tgt_dir, files[-1])).astype(np.float32)
                except Exception:
                    pass
        return None

    def _flush_cloud_csv(self):
        """Write this episode's cloud readings to CSV."""
        # Average per target over episode
        by_target = defaultdict(list)
        for c in self._ep_clouds:
            by_target[c["target_id"]].append(c)
        for tid, records in by_target.items():
            name  = records[0]["target_name"]
            fcst  = float(np.mean([r["cnn_forecast"] for r in records]))
            truth = float(np.mean([r["ground_truth"] for r in records]))
            err   = abs(fcst - truth)
            date  = records[0].get("date_str", "?")
            self._cloud_csv.writerow({
                "episode":      self._ep,
                "date":         date,
                "target_id":    tid,
                "target_name":  name,
                "cnn_forecast": round(fcst,  4),
                "ground_truth": round(truth, 4),
                "abs_error":    round(err,   4),
                "cloudy_thresh": 0.6,
            })
        self._cloud_csv_fh.flush()

    def _open_csv(self, path: str, fieldnames: list):
        fh = open(path, "a", newline="", buffering=1)
        w  = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if os.path.getsize(path) == 0:
            w.writeheader()
        # Keep file handle to close later
        attr = "_cloud_csv_fh" if "cloud" in path else "_ep_csv_fh"
        setattr(self, attr, fh)
        return w

    def _on_training_end(self):
        for attr in ("_cloud_csv_fh", "_ep_csv_fh", "_event_log"):
            fh = getattr(self, attr, None)
            if fh:
                try:
                    fh.close()
                except Exception:
                    pass
        print(f"\n[ThesisLogger] All verification files saved â {self.log_dir}/")
        print(f"  cloud_predictions.csv â CNN vs truth per target per episode")
        print(f"  episode_summary.csv   â per-episode stats")
        print(f"  cloud_patches/        â MODIS patch grids with forecast overlay")
        print(f"  maps/                 â Algeria maps with events and decisions")
        print(f"  timelines/            â decision + reward + cloud error per episode")

