#!/usr/bin/env python3
"""
live_map_tracker.py  --  ALSAT-EO-1 Geographical Ground Tracker & Decision visualization
========================================================================================
Interactive map-based visualization using `cartopy` and `matplotlib` to plot the
satellite's real-time position, ground track, sensor footprint, and targets over
Algeria alongside the policy decisions and attributions.

Controls:
  - Press [SPACEBAR] in the plot window to advance the simulation by 1 step.
  - Press [C] to toggle auto-play mode.
  - Close the window or press [ESC] to exit.
"""
from __future__ import annotations
import os
import sys
import time
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.geodesic as cgeo

# ---- path-setup ----
_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT    = os.path.dirname(_SCRIPTS)
sys.path.insert(0, _SCRIPTS)
sys.path.insert(0, os.path.join(_SCRIPTS, "core"))
sys.path.insert(0, os.path.join(_SCRIPTS, "models"))
sys.path.insert(0, os.path.join(_SCRIPTS, "wrappers"))

from env_dynamic_factory import Config, make_env
from wrappers.action_mask_wrapper import make_masked_env

def eci_to_latlon(r_N, sim_time_s):
    """Converts Inertial ECI coordinates to Earth-Fixed Geodetic Lat/Lon."""
    omega_e = 7.2921150e-5  # Earth rotation rate (rad/s)
    theta = omega_e * sim_time_s
    
    X, Y, Z = r_N[0], r_N[1], r_N[2]
    
    # Rotate around Z axis by -theta
    x = X * math.cos(theta) + Y * math.sin(theta)
    y = -X * math.sin(theta) + Y * math.cos(theta)
    z = Z
    
    lon = math.degrees(math.atan2(y, x))
    lat = math.degrees(math.atan2(z, math.sqrt(x*x + y*y)))
    
    # Wrap longitude to [-180, 180]
    lon = (lon + 180) % 360 - 180
    return lat, lon

def propagate_orbit(r_0, v_0, start_time, duration=5700, step=120):
    """Propagates ECI two-body orbit equations and converts to Lat/Lon track."""
    mu = 3.986004418e14
    r = np.array(r_0, dtype=np.float64)
    v = np.array(v_0, dtype=np.float64)
    
    coords = []
    t = start_time
    for _ in range(int(duration / step)):
        # Runge-Kutta 4th order integration
        def derivatives(state):
            pos = state[:3]
            vel = state[3:]
            r_mag = np.linalg.norm(pos)
            acc = -mu * pos / (r_mag ** 3)
            return np.concatenate([vel, acc])
            
        state = np.concatenate([r, v])
        k1 = derivatives(state)
        k2 = derivatives(state + 0.5 * step * k1)
        k3 = derivatives(state + 0.5 * step * k2)
        k4 = derivatives(state + step * k3)
        state_next = state + (step / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        r = state_next[:3]
        v = state_next[3:]
        t += step
        
        lat, lon = eci_to_latlon(r, t)
        coords.append((lat, lon))
    return coords

class LiveMapTracker:
    def __init__(self, model_path: str, seed: int = 300, event_rate: float = 2.0):
        self.model_path = model_path
        self.seed = seed
        self.event_rate = event_rate
        
        # 1. Environment
        targets_path = os.path.join(_ROOT, "config/targets/global_45_targets.json")
        cloud_json_path = os.path.join(_ROOT, "config/cloud_reality/global_45_clouds.json")
        
        env = make_env(Config.DYN_MODIS, targets_path, cloud_json_path, event_rate=self.event_rate, seed=self.seed, with_safety=False)
        self.env = make_masked_env(env)
        
        # 2. Policy Model
        try:
            from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
            import inspect
            _orig_init = MaskableActorCriticPolicy.__init__
            def _patched_init(self, *args, **kwargs):
                sig = inspect.signature(_orig_init)
                if "use_sde" not in sig.parameters:
                    kwargs.pop("use_sde", None)
                _orig_init(self, *args, **kwargs)
            MaskableActorCriticPolicy.__init__ = _patched_init
        except Exception:
            pass

        import sys
        import numpy.core
        import numpy.core.numeric
        sys.modules["numpy._core"] = numpy.core
        sys.modules["numpy._core.numeric"] = numpy.core.numeric

        from sb3_contrib import MaskablePPO
        from attention_policy import SchedulerAttentionExtractor
        custom_objects = {
            "features_extractor_class": SchedulerAttentionExtractor,
            "action_space": self.env.action_space,
            "observation_space": self.env.observation_space,
        }
        self.model = MaskablePPO.load(self.model_path, env=self.env, custom_objects=custom_objects)
        
        self.obs, _ = self.env.reset()
        self.done = False
        self.step_count = 0
        self.total_reward = 0.0
        self.imaged_count = 0
        
        self.auto_play = False
        self.delay = 1.0
        
        # Track imaged targets
        self.imaged_targets = set()
        
        # Target coordinates lookup
        self.sat = self.env.unwrapped.satellites[0]
        self.all_targets = list(self.sat.scenario.targets)
        import json
        with open(targets_path) as f:
            targets_config = json.load(f)
        self.target_lats = [float(t.get("lat_deg", t.get("lat", 0.0))) for t in targets_config]
        self.target_lons = [float(t.get("lon_deg", t.get("lon", 0.0))) for t in targets_config]
        
        # Setup Matplotlib Interactive Figure
        plt.ion()
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.suptitle("🛰️ ALSAT-EO-1 Flight Tracker & Mission Control", fontsize=16, fontweight="bold")
        
        # Grid layout: Map on left, telemetry on right
        self.ax_map = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())
        self.ax_info = plt.subplot(1, 2, 2)
        self.ax_info.axis('off')
        
        # Set map boundaries over Algeria
        self.ax_map.set_extent([-10, 15, 18, 38], crs=ccrs.PlateCarree())
        
        # Add high-resolution geographical boundaries
        self.ax_map.add_feature(cfeature.LAND, facecolor='#f4f4f5')
        self.ax_map.add_feature(cfeature.OCEAN, facecolor='#e0f2fe')
        self.ax_map.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor='#1e293b')
        self.ax_map.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle=':', edgecolor='#475569')
        
        # Scatter handles
        self.sc_targets = self.ax_map.scatter(
            self.target_lons, self.target_lats,
            c='red', s=40, zorder=5, transform=ccrs.PlateCarree(),
            label='Targets (Not Imaged)'
        )
        
        self.sc_sat = self.ax_map.scatter(
            [], [],
            c='blue', marker='o', s=120, zorder=10, transform=ccrs.PlateCarree(),
            label='ALSAT-EO-1'
        )
        
        self.ln_orbit, = self.ax_map.plot(
            [], [],
            color='#3b82f6', linewidth=2.0, zorder=4, transform=ccrs.PlateCarree(),
            label='Orbit Ground Track'
        )
        
        self.swath_patch = None
        self.event_markers = []
        
        self.ax_map.legend(loc='lower left')
        
        # Connect key listener
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.advance_needed = False

    def _get_action_probabilities(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
        action_masks = self.env.action_masks()
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(obs_tensor, action_masks=action_masks)
            probs = distribution.distribution.probs.cpu().numpy()[0]
        return probs

    def _explain_action(self, obs: np.ndarray, action: int, probs: np.ndarray) -> list[str]:
        explanations = []
        base_prob = probs[action]
        
        if action < 45:
            tgt = self.all_targets[action]
            explanations.append(f"Action: Image static target [bold green]{tgt.name}[/bold green]")
            explanations.append(f"  - Target Priority: {tgt.priority:.2f}")
            # Locate slot in observation to inspect cloud
            found_slot = -1
            for slot_i in range(6):
                tid_val = obs[13 + slot_i*6 + 5]
                expected_tid = float(action) / 20.0
                if abs(tid_val - expected_tid) < 0.01:
                    found_slot = slot_i
                    break
            if found_slot != -1:
                idx_cloud = 13 + found_slot * 6 + 2
                cloud_fcst = obs[idx_cloud] * 100
                explanations.append(f"  - Cloud Cover Forecast: {cloud_fcst:.0f}%")
                if cloud_fcst > 50:
                    explanations.append(f"  - WARNING: high cloud probability! Selecting due to urgency or priority.")
                else:
                    explanations.append(f"  - Target region is forecasted clear (imaging rewarded).")
            else:
                explanations.append("  - Target is visible in current orbital pass.")
                
        elif 45 <= action <= 47:
            slot_idx = action - 45
            mgr = getattr(self.sat, "_event_manager", None)
            now = self.sat.simulator.sim_time
            slots = mgr.get_slots(self.sat, now)
            evt = slots[slot_idx]
            if evt is not None:
                explanations.append(f"Action: Image [bold magenta]Dynamic Event ({evt.name})[/bold magenta]")
                explanations.append(f"  - Event Priority: {evt.priority:.2f}")
                explanations.append(f"  - Cloud Cover Forecast: {evt.cloud_cover_forecast * 100:.0f}%")
                explanations.append(f"  - Urgent priority overrides static scheduling.")
            else:
                explanations.append(f"Action: Image Dynamic Slot {slot_idx} (Empty slot).")
                
        elif action == 48:
            explanations.append("Action: [bold yellow]Solar panel Drift (Recharge battery)[/bold yellow]")
            battery_soc = obs[0] * 100
            explanations.append(f"  - Battery SoC: {battery_soc:.1f}%")
            if battery_soc < 35.0:
                explanations.append(f"  - Enforced Drift: charging battery to prevent power depletion.")
            else:
                explanations.append(f"  - No clear targets available or desaturating reaction wheels.")
                
        # Top Alternatives
        top_indices = np.argsort(probs)[::-1]
        alts = []
        count = 0
        for idx in top_indices:
            if idx == action:
                continue
            if count >= 3:
                break
            prob_pct = probs[idx] * 100
            if prob_pct < 0.1:
                continue
            if idx < 45:
                alts.append(f"{self.all_targets[idx].name} ({prob_pct:.1f}%)")
            elif 45 <= idx <= 47:
                alts.append(f"Dyn Slot {idx-45} ({prob_pct:.1f}%)")
            else:
                alts.append(f"Drift ({prob_pct:.1f}%)")
            count += 1
        if alts:
            explanations.append(f"Rejected Alternatives: {', '.join(alts)}")
        return explanations

    def on_key(self, event):
        if event.key == ' ':
            self.advance_needed = True
        elif event.key in ('c', 'C'):
            self.auto_play = not self.auto_play
            print(f"[TRACKER] Auto-play toggled: {self.auto_play}")
        elif event.key == 'escape':
            plt.close()
            sys.exit(0)

    def draw_swath_circle(self, lat, lon):
        """Draws a geodetic circle representing the camera field of view swath (500km)."""
        if self.swath_patch:
            self.swath_patch.remove()
        
        try:
            geod = cgeo.Geodesic()
            # 500 km swath circle on ellipsoid
            circle_pts = geod.circle(lon=lon, lat=lat, radius=500000, n_samples=60)
            self.swath_patch = plt.Polygon(
                circle_pts,
                facecolor='#10b981', edgecolor='#059669',
                alpha=0.18, zorder=6, transform=ccrs.PlateCarree()
            )
            self.ax_map.add_patch(self.swath_patch)
        except Exception:
            pass

    def update_plot(self):
        # 1. Update satellite position and ground track
        sim_time_s = float(self.sat.simulator.sim_time)
        r_N = (getattr(self.sat.dynamics, "r_SC_N", None) or 
               getattr(self.sat.dynamics, "r_BN_N", None) or 
               getattr(self.sat.dynamics, "r_N", None))
        v_N = (getattr(self.sat.dynamics, "v_SC_N", None) or 
               getattr(self.sat.dynamics, "v_BN_N", None) or 
               getattr(self.sat.dynamics, "v_N", None))
        
        if r_N is None:
            return
            
        sat_lat, sat_lon = eci_to_latlon(r_N, sim_time_s)
        self.sc_sat.set_offsets([[sat_lon, sat_lat]])
        
        # Future ground track for 1 full orbit (5700s)
        orbit_track = propagate_orbit(r_N, v_N, sim_time_s, duration=5700, step=120)
        track_lons = [pt[1] for pt in orbit_track]
        track_lats = [pt[0] for pt in orbit_track]
        self.ln_orbit.set_data(track_lons, track_lats)
        
        # Center map on satellite
        self.ax_map.set_extent([sat_lon-12, sat_lon+12, sat_lat-10, sat_lat+10], crs=ccrs.PlateCarree())
        
        # Draw swath circle
        self.draw_swath_circle(sat_lat, sat_lon)
        
        # 2. Update Target Colors
        colors = []
        for t in self.all_targets:
            if t.name in self.imaged_targets:
                colors.append('#22c55e')  # Green (Imaged)
            else:
                colors.append('#ef4444')  # Red (Not Imaged)
        self.sc_targets.set_facecolors(colors)
        self.sc_targets.set_edgecolors(colors)
        
        # 3. Update Dynamic Event Markers
        for m in self.event_markers:
            m.remove()
        self.event_markers = []
        
        mgr = getattr(self.sat, "_event_manager", None)
        active_slots = mgr.get_slots(self.sat, sim_time_s)
        for slot in active_slots:
            if slot is not None:
                evt_lat = math.degrees(slot.lat_rad)
                evt_lon = math.degrees(slot.lon_rad)
                m = self.ax_map.scatter(
                    evt_lon, evt_lat,
                    c='#e11d48', marker='*', s=150, zorder=8,
                    transform=ccrs.PlateCarree(), label='Dynamic Event'
                )
                self.event_markers.append(m)
                
        # 4. Update Policy Decision & Telemetry panel
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        probs = self._get_action_probabilities(self.obs)
        action, _ = self.model.predict(self.obs, deterministic=True)
        action = int(action)
        
        explanations = self._explain_action(self.obs, action, probs)
        
        h = int(sim_time_s // 3600)
        m = int((sim_time_s % 3600) // 60)
        
        info_lines = [
            "🛰️ ALSAT-EO-1 TELEMETRY",
            "========================",
            f"• Mission Time : {h:02d}h {m:02d}m",
            f"• Battery SoC  : {self.obs[0]*100:.1f}%",
            f"• Total Steps  : {self.step_count}",
            f"• Total Reward : {self.total_reward:+.3f}",
            f"• Images Taken : {self.imaged_count}",
            "",
            "🧠 RL AGENT DECISION LOG",
            "========================"
        ] + explanations + [
            "",
            "---------------------------------------",
            "Controls:",
            "  - Press [SPACEBAR] in window to step",
            "  - Press [C] to toggle Auto-Play",
            "  - Press [ESC] to terminate"
        ]
        
        # Write text to subplot
        info_text = "\n".join(info_lines)
        self.ax_info.text(
            0.05, 0.95, info_text,
            transform=self.ax_info.transAxes,
            fontsize=10, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#fafafa', alpha=0.9, edgecolor='#cbd5e1')
        )
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def run(self):
        self.update_plot()
        
        while not self.done:
            plt.pause(0.01)
            
            # Step execution trigger
            if self.advance_needed or self.auto_play:
                self.advance_needed = False
                
                probs = self._get_action_probabilities(self.obs)
                action, _ = self.model.predict(self.obs, deterministic=True)
                action = int(action)
                
                # Take step
                next_obs, reward, term, trunc, info_step = self.env.step(action)
                self.total_reward += reward
                self.step_count += 1
                self.done = term or trunc
                
                # Image count checking
                if action < 45:
                    if info_step.get("imaging_occurred", False):
                        self.imaged_count += 1
                        self.imaged_targets.add(self.all_targets[action].name)
                elif 45 <= action <= 47:
                    if info_step.get("dynamic_imaging_occurred", False):
                        self.imaged_count += 1
                        mgr = getattr(self.sat, "_event_manager", None)
                        active_slots = mgr.get_slots(self.sat, float(self.sat.simulator.sim_time))
                        evt = active_slots[action - 45]
                        if evt:
                            self.imaged_targets.add(evt.name)
                            
                self.obs = next_obs
                self.update_plot()
                
                if self.auto_play:
                    time.sleep(self.delay)
                    
        print(f"\n[TRACKER] Simulation Finished! Total Steps: {self.step_count} | Total Images: {self.imaged_count}\n")
        plt.ioff()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Live geographical orbit tracker")
    parser.add_argument("--model", type=str, required=True, help="Path to trained PPO model")
    parser.add_argument("--seed", type=int, default=300, help="Random seed")
    parser.add_argument("--event-rate", type=float, default=2.0, help="Dynamic event rate")
    args = parser.parse_args()
    
    tracker = LiveMapTracker(model_path=args.model, seed=args.seed, event_rate=args.event_rate)
    tracker.run()

if __name__ == "__main__":
    main()
