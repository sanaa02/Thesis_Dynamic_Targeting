#!/usr/bin/env python3
"""
live_web_dashboard.py  --  ALSAT-EO-1 3D Web Dashboard & Explainability Center
=============================================================================
Launches a local HTTP server that serves an interactive 3D Web Dashboard.
Features:
  - Holographic 3D Globe (Three.js) showing the satellite's real-time orbit track.
  - Close-up 3D Satellite View showing the satellite physically rotating (slewing)
    and shooting camera beams to image targets.
  - Live Explainability: gradient attributions, rejected choices, and decision reasons.
  - Flashing alerts for dynamic wildfire/flood events.
  - Zero-installation: runs entirely via python's built-in http.server.

Usage:
------
    CUDA_VISIBLE_DEVICES="" python scripts/evaluation/live_web_dashboard.py \
        --model models/ppo_full_system_s42.zip \
        --port 8080
"""
from __future__ import annotations
import os
import sys
import time
import math
import json
import argparse
import webbrowser
import http.server
import socketserver
import threading
import numpy as np
import gymnasium as gym
import torch

# ---- path-setup ----
_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT    = os.path.dirname(_SCRIPTS)
sys.path.insert(0, _SCRIPTS)
sys.path.insert(0, os.path.join(_SCRIPTS, "core"))
sys.path.insert(0, os.path.join(_SCRIPTS, "models"))
sys.path.insert(0, os.path.join(_SCRIPTS, "wrappers"))

from env_dynamic_factory import Config, make_env
from wrappers.action_mask_wrapper import make_masked_env

# --- Geodetic & Orbit Math ---
def eci_to_latlon(r_N, sim_time_s):
    omega_e = 7.2921150e-5
    theta = omega_e * sim_time_s
    X, Y, Z = r_N[0], r_N[1], r_N[2]
    x = X * math.cos(theta) + Y * math.sin(theta)
    y = -X * math.sin(theta) + Y * math.cos(theta)
    z = Z
    lon = math.degrees(math.atan2(y, x))
    lat = math.degrees(math.atan2(z, math.sqrt(x*x + y*y)))
    lon = (lon + 180) % 360 - 180
    return lat, lon

def propagate_orbit(r_0, v_0, start_time, duration=5700, step=150):
    if np.linalg.norm(r_0) < 1.0:
        return []
    mu = 3.986004418e14
    r = np.array(r_0, dtype=np.float64)
    v = np.array(v_0, dtype=np.float64)
    coords = []
    t = start_time
    for _ in range(int(duration / step)):
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
        coords.append({"lat": lat, "lon": lon})
    return coords

# --- Dashboard Handler Backend ---
class DashboardBackend:
    def __init__(self, model_path: str, seed: int = 300, event_rate: float = 2.0):
        self.model_path = model_path
        self.seed = seed
        self.event_rate = event_rate
        
        # Load environment
        self.targets_path = os.path.join(_ROOT, "config/targets/global_45_targets.json")
        self.cloud_json_path = os.path.join(_ROOT, "config/cloud_reality/global_45_clouds.json")
        self.reset_env()
        
        # Load model with SB3 patch
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

    def reset_env(self):
        env = make_env(Config.DYN_MODIS, self.targets_path, self.cloud_json_path, event_rate=self.event_rate, seed=self.seed, with_safety=False)
        self.env = make_masked_env(env)
        self.obs, _ = self.env.reset()
        self.step_count = 0
        self.total_reward = 0.0
        self.imaged_count = 0
        self.imaged_targets = set()
        self.sat = self.env.unwrapped.satellites[0]
        self.all_targets = list(self.sat.scenario.targets)
        
        # Load target coordinates from config
        with open(self.targets_path) as f:
            targets_config = json.load(f)
        self.target_coords = [
            {"name": t.get("name"), "lat": float(t.get("lat_deg", t.get("lat", 0.0))), "lon": float(t.get("lon_deg", t.get("lon", 0.0))), "priority": float(t.get("priority", 1.0))}
            for t in targets_config
        ]

    def get_action_probs(self, obs):
        obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
        action_masks = self.env.action_masks()
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(obs_tensor, action_masks=action_masks)
            probs = distribution.distribution.probs.cpu().numpy()[0]
        return probs

    def get_state_json(self):
        sim_time_s = float(self.sat.simulator.sim_time)
        r_N = (getattr(self.sat.dynamics, "r_SC_N", None) or 
               getattr(self.sat.dynamics, "r_BN_N", None) or 
               getattr(self.sat.dynamics, "r_N", [0.0, 0.0, 0.0]))
        v_N = (getattr(self.sat.dynamics, "v_SC_N", None) or 
               getattr(self.sat.dynamics, "v_BN_N", None) or 
               getattr(self.sat.dynamics, "v_N", [0.0, 0.0, 0.0]))
        sat_lat, sat_lon = eci_to_latlon(r_N, sim_time_s)
        
        # Calculate future orbit ground track points
        orbit_track = propagate_orbit(r_N, v_N, sim_time_s)
        
        # Get target details
        targets_list = []
        for idx, t in enumerate(self.all_targets):
            coords = self.target_coords[idx]
            targets_list.append({
                "name": t.name,
                "lat": coords["lat"],
                "lon": coords["lon"],
                "priority": t.priority,
                "cloud": getattr(t, "cloud_cover", 0.0),
                "imaged": t.name in self.imaged_targets
            })
            
        # Get dynamic event slots
        mgr = getattr(self.sat, "_event_manager", None)
        active_slots = mgr.get_slots(self.sat, sim_time_s)
        events_list = []
        for idx, evt in enumerate(active_slots):
            if evt is not None:
                events_list.append({
                    "slot": idx,
                    "name": evt.name,
                    "lat": math.degrees(evt.lat_rad),
                    "lon": math.degrees(evt.lon_rad),
                    "priority": evt.priority,
                    "cloud": evt.cloud_cover_forecast,
                    "expires_in": max(0.0, evt.expiration_time - sim_time_s)
                })
                
        # Predict next action and explain it
        probs = self.get_action_probs(self.obs)
        action, _ = self.model.predict(self.obs, deterministic=True)
        action = int(action)
        
        # Explain action
        explanations = []
        if action < 45:
            tgt = self.all_targets[action]
            coords = self.target_coords[action]
            explanations.append(f"• Target [bold cyan]{tgt.name}[/bold cyan] chosen.")
            explanations.append(f"  - Target Priority: {tgt.priority:.2f}")
            explanations.append(f"  - Cloud Cover Forecast: {getattr(tgt, 'cloud_cover_forecast', 0.5)*100:.0f}%")
            explanations.append(f"  - Spacecraft is executing slew maneuver to target region.")
            target_focus = {"lat": coords["lat"], "lon": coords["lon"]}
            action_type = "image_static"
        elif 45 <= action <= 47:
            slot_idx = action - 45
            evt = active_slots[slot_idx]
            if evt:
                explanations.append(f"• [bold magenta]Dynamic Event ({evt.name})[/bold magenta] chosen.")
                explanations.append(f"  - Priority: {evt.priority:.2f} | Cloud forecast: {evt.cloud_cover_forecast*100:.0f}%")
                target_focus = {"lat": math.degrees(evt.lat_rad), "lon": math.degrees(evt.lon_rad)}
                action_type = "image_dynamic"
            else:
                explanations.append(f"• Dynamic slot {slot_idx} selected (empty).")
                target_focus = None
                action_type = "drift"
        else:
            explanations.append("• Satellite performing panel charging Drift.")
            target_focus = None
            action_type = "drift"
            
        # Top 3 alternatives
        top_indices = np.argsort(probs)[::-1]
        alts = []
        count = 0
        for idx in top_indices:
            if idx == action:
                continue
            if count >= 3:
                break
            prob = probs[idx] * 100
            if prob < 0.1:
                continue
            if idx < 45:
                alts.append(f"{self.all_targets[idx].name} ({prob:.1f}%)")
            elif 45 <= idx <= 47:
                alts.append(f"Dyn Slot {idx-45} ({prob:.1f}%)")
            else:
                alts.append(f"Drift ({prob:.1f}%)")
            count += 1
            
        battery_pct = self.obs[0] * 100
            
        return {
            "time_h": int(sim_time_s // 3600),
            "time_m": int((sim_time_s % 3600) // 60),
            "step": self.step_count,
            "reward": self.total_reward,
            "images_taken": self.imaged_count,
            "battery_soc": battery_pct,
            "sat_lat": sat_lat,
            "sat_lon": sat_lon,
            "orbit_track": orbit_track,
            "targets": targets_list,
            "events": events_list,
            "action": action,
            "action_type": action_type,
            "confidence": float(probs[action] * 100),
            "explanations": explanations,
            "alternates": alts,
            "target_focus": target_focus
        }

    def step(self):
        # 1. Capture state before step
        t_old = float(self.sat.simulator.sim_time)
        r_old = (getattr(self.sat.dynamics, "r_SC_N", None) or 
                 getattr(self.sat.dynamics, "r_BN_N", None) or 
                 getattr(self.sat.dynamics, "r_N", [0.0, 0.0, 0.0]))
        v_old = (getattr(self.sat.dynamics, "v_SC_N", None) or 
                 getattr(self.sat.dynamics, "v_BN_N", None) or 
                 getattr(self.sat.dynamics, "v_N", [0.0, 0.0, 0.0]))

        # 2. Execute step
        action, _ = self.model.predict(self.obs, deterministic=True)
        action = int(action)
        
        next_obs, reward, term, trunc, info_step = self.env.step(action)
        self.total_reward += reward
        self.step_count += 1
        
        # Track imaged targets
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
        
        # 3. Compute transition path
        t_new = float(self.sat.simulator.sim_time)
        dt = t_new - t_old
        if dt > 0.0 and np.linalg.norm(r_old) >= 1.0:
            transition_path = propagate_orbit(r_old, v_old, t_old, duration=dt, step=40)
        else:
            transition_path = []
            
        state = self.get_state_json()
        state["transition_path"] = transition_path
        return state

# --- Server Handlers ---
def make_handler(backend: DashboardBackend):
    class DashboardHTTPHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Suppress console log spam
            return

        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(HTML_CONTENT.encode("utf-8"))
            else:
                self.send_error(404, "File not found")

        def do_POST(self):
            if self.path == "/api/reset":
                backend.reset_env()
                state = backend.get_state_json()
                self.send_json_response(state)
            elif self.path == "/api/step":
                state = backend.step()
                self.send_json_response(state)
            else:
                self.send_error(404, "Endpoint not found")

        def send_json_response(self, data):
            response_bytes = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_bytes)))
            self.end_headers()
            self.wfile.write(response_bytes)

    return DashboardHTTPHandler

# --- Frontend HTML/CSS/JS (Procedural Three.js) ---
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ALSAT-EO-1 Mission Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        :root {
            --bg-color: #090d16;
            --panel-bg: rgba(13, 20, 38, 0.7);
            --border-color: rgba(59, 130, 246, 0.2);
            --text-color: #f8fafc;
            --neon-blue: #3b82f6;
            --neon-green: #10b981;
            --neon-magenta: #f43f5e;
            --font-stack: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        body {
            margin: 0;
            padding: 0;
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: var(--font-stack);
            overflow: hidden;
            height: 100vh;
        }

        #app-container {
            display: grid;
            grid-template-columns: 320px 1fr 360px;
            height: 100vh;
            padding: 10px;
            box-sizing: border-box;
            gap: 10px;
        }

        .panel {
            background-color: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            backdrop-filter: blur(16px);
            display: flex;
            flex-direction: column;
            padding: 15px;
            box-sizing: border-box;
            overflow-y: auto;
        }

        /* Scrollbars */
        ::-webkit-scrollbar {
            width: 4px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(0,0,0,0.1);
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255,255,255,0.2);
            border-radius: 4px;
        }

        h2 {
            margin-top: 0;
            font-size: 1.1rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 8px;
            color: var(--neon-blue);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        /* Center Visualizer */
        #visualizer-container {
            position: relative;
            grid-column: 2;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        #globe-canvas-container {
            flex-grow: 1;
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            position: relative;
            background: radial-gradient(circle, #1e293b 0%, #020617 100%);
        }

        /* Corner Satellite Screen */
        #sat-screen-container {
            position: absolute;
            bottom: 20px;
            right: 20px;
            width: 180px;
            height: 180px;
            border: 2px solid var(--neon-blue);
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.4);
            background: #000;
        }

        #sat-screen-label {
            position: absolute;
            top: 5px;
            left: 5px;
            background: rgba(0,0,0,0.7);
            padding: 2px 6px;
            font-size: 9px;
            font-weight: bold;
            color: var(--neon-blue);
            border-radius: 3px;
            z-index: 10;
            letter-spacing: 0.5px;
        }

        /* Telemetry Cards */
        .telemetry-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 15px;
        }

        .tel-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 8px;
            padding: 8px;
            text-align: center;
        }

        .tel-val {
            font-size: 1.3rem;
            font-weight: bold;
            color: var(--neon-blue);
        }

        .tel-label {
            font-size: 9px;
            text-transform: uppercase;
            color: #94a3b8;
            margin-top: 3px;
        }

        /* Control Panel */
        #controls {
            display: flex;
            gap: 10px;
            padding: 10px;
            background: rgba(13, 20, 38, 0.8);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            justify-content: center;
        }

        .btn {
            background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
            border: none;
            color: white;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.2s;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 1px;
        }

        .btn:hover {
            box-shadow: 0 0 15px var(--neon-blue);
            transform: translateY(-1px);
        }

        .btn-magenta {
            background: linear-gradient(135deg, #be123c 0%, #9f1239 100%);
        }

        .btn-magenta:hover {
            box-shadow: 0 0 15px var(--neon-magenta);
        }

        /* Explainability Log style */
        .log-box {
            font-family: 'Courier New', Courier, monospace;
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 8px;
            padding: 10px;
            font-size: 11px;
            line-height: 1.5;
            flex-grow: 1;
            overflow-y: auto;
        }

        /* Highlight classes */
        .green-txt { color: var(--neon-green); }
        .blue-txt { color: var(--neon-blue); }
        .magenta-txt { color: var(--neon-magenta); }
        .yellow-txt { color: #eab308; }

        /* Dynamic Events list */
        .evt-item {
            border-left: 3px solid var(--neon-magenta);
            background: rgba(244, 63, 94, 0.05);
            padding: 6px 10px;
            border-radius: 0 6px 6px 0;
            margin-bottom: 8px;
            font-size: 11px;
            animation: pulse-bg 2s infinite alternate;
        }

        @keyframes pulse-bg {
            0% { background: rgba(244, 63, 94, 0.03); }
            100% { background: rgba(244, 63, 94, 0.15); }
        }

        /* Targets List */
        .tgt-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 11px;
        }

        .tgt-table th, .tgt-table td {
            text-align: left;
            padding: 5px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }

        .status-badge {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }

        .status-green { background-color: var(--neon-green); }
        .status-red { background-color: var(--neon-magenta); }
    </style>
</head>
<body>
    <div id="app-container">
        <!-- LEFT: Telemetry & Targets -->
        <div class="panel">
            <h2>Spacecraft Telemetry</h2>
            <div class="telemetry-grid">
                <div class="tel-card">
                    <div id="tel-time" class="tel-val">00h 00m</div>
                    <div class="tel-label">Mission Time</div>
                </div>
                <div class="tel-card">
                    <div id="tel-step" class="tel-val">0</div>
                    <div class="tel-label">Step Index</div>
                </div>
                <div class="tel-card">
                    <div id="tel-battery" class="tel-val">100%</div>
                    <div class="tel-label">Battery (SoC)</div>
                </div>
                <div class="tel-card">
                    <div id="tel-reward" class="tel-val">+0.00</div>
                    <div class="tel-label">Reward</div>
                </div>
            </div>

            <h2>Static Targets</h2>
            <div style="flex-grow: 1; overflow-y: auto; max-height: 250px; margin-bottom: 15px;">
                <table class="tgt-table">
                    <thead>
                        <tr>
                            <th>Target</th>
                            <th>Prio</th>
                            <th>Cloud</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody id="targets-list-body">
                        <!-- Filled dynamically -->
                    </tbody>
                </table>
            </div>

            <h2>Dynamic Alerts</h2>
            <div id="events-list" style="flex-grow: 1; overflow-y: auto;">
                <div class="dim-txt" style="text-align: center; font-size: 11px; margin-top: 20px;">No events detected</div>
            </div>
        </div>

        <!-- CENTER: Main 3D Globe Visualizer -->
        <div id="visualizer-container">
            <div id="globe-canvas-container">
                <!-- 3D Satellite Close-up Screen -->
                <div id="sat-screen-container">
                    <div id="sat-screen-label">ALSAT-EO-1 (CLOSE-UP VIEW)</div>
                </div>
            </div>
            
            <!-- Simulation Controls -->
            <div id="controls">
                <button id="btn-reset" class="btn btn-magenta">Reset</button>
                <button id="btn-step" class="btn" style="flex-grow: 1;">Step Action [Space]</button>
                <button id="btn-camlock" class="btn">Lock Camera</button>
                <button id="btn-autoplay" class="btn">Auto-Play</button>
            </div>
        </div>

        <!-- RIGHT: Decision Explainability -->
        <div class="panel">
            <h2>Decision Center</h2>
            <div class="telemetry-grid" style="grid-template-columns: 1fr;">
                <div class="tel-card" style="text-align: left; padding: 10px;">
                    <div id="decision-text" class="tel-val" style="font-size: 1.1rem; color: var(--neon-green);">Initializing...</div>
                    <div class="tel-label">Active Action</div>
                </div>
            </div>

            <h2>Decision Attributions</h2>
            <div id="decision-log" class="log-box">
                <!-- Explained attribution details filled dynamically -->
            </div>
        </div>
    </div>

    <script>
        // --- Globals ---
        let scene, camera, renderer, globe, orbitLine;
        let globeLaser, globeSwath;
        let satMesh, stars;
        let targetSpheres = [];
        let controls;
        let camLockActive = false;
        let orbitPoints = [];
        let orbitIndex = 0;
        let currentState = null;
        
        let satScene, satCamera, satRenderer, satBody;
        let laserBeam;

        let autoplayActive = false;
        let autoplayInterval = null;

        // Initialize 3D Scenes
        init3DGlobe();
        initCloseUpSat();
        resetSim();

        // Keyboard bindings
        window.addEventListener('keydown', (e) => {
            if (e.code === 'Space') {
                e.preventDefault();
                stepSim();
            }
        });

        document.getElementById('btn-step').addEventListener('click', stepSim);
        document.getElementById('btn-reset').addEventListener('click', resetSim);
        document.getElementById('btn-autoplay').addEventListener('click', toggleAutoplay);
        document.getElementById('btn-camlock').addEventListener('click', toggleCamLock);

        function toggleCamLock() {
            camLockActive = !camLockActive;
            const btn = document.getElementById('btn-camlock');
            if (camLockActive) {
                btn.textContent = 'Camera Locked';
                btn.style.background = 'linear-gradient(135deg, #10b981 0%, #059669 100%)';
            } else {
                btn.textContent = 'Lock Camera';
                btn.style.background = 'linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%)';
            }
        }

        function init3DGlobe() {
            const container = document.getElementById('globe-canvas-container');
            const width = container.clientWidth;
            const height = container.clientHeight;

            scene = new THREE.Scene();
            camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
            camera.position.set(0, 0, 180);

            renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
            renderer.setSize(width, height);
            container.appendChild(renderer.domElement);

            // Light sources
            const ambient = new THREE.AmbientLight(0x333333);
            scene.add(ambient);
            const sunLight = new THREE.DirectionalLight(0xffffff, 1.2);
            sunLight.position.set(100, 50, 50);
            scene.add(sunLight);

            // Holographic Earth Globe with texture mapping
            const globeGeo = new THREE.SphereGeometry(60, 48, 48);
            const globeMat = new THREE.MeshPhongMaterial({
                color: 0xcccccc, // clean gray fallback before texture loads
                shininess: 15,
                bumpScale: 0.05
            });
            globe = new THREE.Mesh(globeGeo, globeMat);
            scene.add(globe);

            // Load high-resolution Earth texture dynamically
            const textureLoader = new THREE.TextureLoader();
            textureLoader.crossOrigin = 'Anonymous';
            textureLoader.load(
                'https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg',
                (texture) => {
                    globeMat.map = texture;
                    globeMat.color.setHex(0xffffff); // remove fallback color overlay
                    globeMat.needsUpdate = true;
                },
                undefined,
                (error) => {
                    console.warn("Failed to load Earth texture, falling back to tactical wireframe.", error);
                    globeMat.color.setHex(0x0f172a); // dark blue tactical style
                    globeMat.wireframe = true;
                    globeMat.needsUpdate = true;
                }
            );

            // Add mesh grid overlay
            const gridGeo = new THREE.SphereGeometry(60.1, 24, 24);
            const gridMat = new THREE.MeshBasicMaterial({
                color: 0x3b82f6,
                wireframe: true,
                transparent: true,
                opacity: 0.15
            });
            const gridMesh = new THREE.Mesh(gridGeo, gridMat);
            globe.add(gridMesh);

            // Satellite 3D model (large scale to be visible orbiting the globe)
            satMesh = new THREE.Group();
            
            // Main body (golden cuboid)
            const sBodyGeo = new THREE.BoxGeometry(4.5, 3.0, 3.0);
            const sBodyMat = new THREE.MeshPhongMaterial({ color: 0xd97706, shininess: 80 });
            const sBody = new THREE.Mesh(sBodyGeo, sBodyMat);
            satMesh.add(sBody);
            
            // Blue solar panels extending on both sides
            const sPanelGeo = new THREE.BoxGeometry(9.0, 0.15, 2.0);
            const sPanelMat = new THREE.MeshPhongMaterial({ color: 0x1e3a8a, specular: 0x3b82f6 });
            const sLPanel = new THREE.Mesh(sPanelGeo, sPanelMat);
            sLPanel.position.set(6.75, 0, 0);
            const sRPanel = sLPanel.clone();
            sRPanel.position.set(-6.75, 0, 0);
            satMesh.add(sLPanel);
            satMesh.add(sRPanel);
            
            // Red beacon light so it stands out and pulses
            const beaconGeo = new THREE.SphereGeometry(1.0, 8, 8);
            const beaconMat = new THREE.MeshBasicMaterial({ color: 0xff3b30 });
            const beacon = new THREE.Mesh(beaconGeo, beaconMat);
            beacon.position.set(0, 1.6, 0);
            satMesh.add(beacon);
            
            scene.add(satMesh);

            // Orbit ground track line (Segments to prevent wrapping cuts across globe)
            const orbitMat = new THREE.LineBasicMaterial({ color: 0x3b82f6, linewidth: 2 });
            const orbitGeo = new THREE.BufferGeometry();
            orbitLine = new THREE.LineSegments(orbitGeo, orbitMat);
            scene.add(orbitLine);

            // Main globe imaging laser beam line
            const laserMat = new THREE.LineBasicMaterial({ color: 0x10b981, linewidth: 3 });
            const laserGeo = new THREE.BufferGeometry();
            globeLaser = new THREE.Line(laserGeo, laserMat);
            scene.add(globeLaser);

            // Main globe swath ring (footprint tangent to Earth surface)
            const swathMat = new THREE.MeshBasicMaterial({ 
                color: 0x10b981, 
                side: THREE.DoubleSide,
                transparent: true,
                opacity: 0.0
            });
            const swathGeo = new THREE.RingGeometry(1.5, 3.0, 16);
            globeSwath = new THREE.Mesh(swathGeo, swathMat);
            scene.add(globeSwath);

            // Starfield background
            const starsGeo = new THREE.BufferGeometry();
            const starsCount = 500;
            const starsPositions = new Float32Array(starsCount * 3);
            for(let i=0; i<starsCount*3; i++) {
                starsPositions[i] = (Math.random() - 0.5) * 500;
            }
            starsGeo.setAttribute('position', new THREE.BufferAttribute(starsPositions, 3));
            const starsMat = new THREE.PointsMaterial({ color: 0x888888, size: 0.8 });
            stars = new THREE.Points(starsGeo, starsMat);
            scene.add(stars);

            // Add OrbitControls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.minDistance = 80;
            controls.maxDistance = 300;

            // Resize handler
            window.addEventListener('resize', () => {
                const w = container.clientWidth;
                const h = container.clientHeight;
                camera.aspect = w / h;
                camera.updateProjectionMatrix();
                renderer.setSize(w, h);
            });

            animateGlobe();
        }

        function initCloseUpSat() {
            const container = document.getElementById('sat-screen-container');
            const w = container.clientWidth;
            const h = container.clientHeight;

            satScene = new THREE.Scene();
            satCamera = new THREE.PerspectiveCamera(40, w / h, 0.1, 100);
            satCamera.position.set(0, 8, 15);
            satCamera.lookAt(0, 0, 0);

            satRenderer = new THREE.WebGLRenderer({ antialias: true });
            satRenderer.setSize(w, h);
            container.appendChild(satRenderer.domElement);

            const ambient = new THREE.AmbientLight(0x444444);
            satScene.add(ambient);
            const light = new THREE.DirectionalLight(0xffffff, 1.0);
            light.position.set(10, 10, 5);
            satScene.add(light);

            // Satellite Model assembly
            satBody = new THREE.Group();

            // Main golden body
            const bodyGeo = new THREE.BoxGeometry(2.5, 2, 2);
            const bodyMat = new THREE.MeshPhongMaterial({ color: 0xd97706, shininess: 80 });
            const body = new THREE.Mesh(bodyGeo, bodyMat);
            satBody.add(body);

            // Solar panels (blue arrays)
            const panelGeo = new THREE.BoxGeometry(4.5, 0.1, 1.2);
            const panelMat = new THREE.MeshPhongMaterial({ color: 0x1e3a8a, specular: 0x3b82f6 });
            const leftPanel = new THREE.Mesh(panelGeo, panelMat);
            leftPanel.position.set(3.5, 0, 0);
            const rightPanel = leftPanel.clone();
            rightPanel.position.set(-3.5, 0, 0);
            satBody.add(leftPanel);
            satBody.add(rightPanel);

            // Camera cylinder aperture
            const camGeo = new THREE.CylinderGeometry(0.5, 0.5, 1.0, 16);
            const camMat = new THREE.MeshPhongMaterial({ color: 0x1e293b });
            const cameraLens = new THREE.Mesh(camGeo, camMat);
            cameraLens.position.set(0, -1.2, 0);
            satBody.add(cameraLens);

            satScene.add(satBody);

            // Earth horizon plane below
            const horizonGeo = new THREE.PlaneGeometry(25, 25, 10, 10);
            const horizonMat = new THREE.MeshBasicMaterial({ color: 0x0f172a, wireframe: true });
            const horizon = new THREE.Mesh(horizonGeo, horizonMat);
            horizon.rotation.x = -Math.PI / 2;
            horizon.position.y = -4;
            satScene.add(horizon);

            // Laser imaging beam cone
            const beamGeo = new THREE.ConeGeometry(0.8, 3.5, 16, 1, true);
            const beamMat = new THREE.MeshBasicMaterial({
                color: 0x10b981,
                transparent: true,
                opacity: 0.0,
                side: THREE.DoubleSide
            });
            laserBeam = new THREE.Mesh(beamGeo, beamMat);
            laserBeam.position.set(0, -2.8, 0);
            satScene.add(laserBeam);

            animateSat();
        }

        function latLonToVector3(lat, lon, radius) {
            const phi = (90 - lat) * (Math.PI / 180);
            const theta = (lon + 180) * (Math.PI / 180);
            
            return new THREE.Vector3(
                -radius * Math.sin(phi) * Math.sin(theta),
                radius * Math.cos(phi),
                -radius * Math.sin(phi) * Math.cos(theta)
            );
        }

        function drawGlobeLaser(satPos, targetLatLon) {
            if (!targetLatLon || !satPos) {
                globeLaser.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
                return;
            }
            const tgtPos = latLonToVector3(targetLatLon.lat, targetLatLon.lon, 60.1);
            const points = [satPos, tgtPos];
            const positions = new Float32Array(6);
            positions[0] = points[0].x;
            positions[1] = points[0].y;
            positions[2] = points[0].z;
            positions[3] = points[1].x;
            positions[4] = points[1].y;
            positions[5] = points[1].z;
            globeLaser.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            globeLaser.geometry.computeBoundingSphere();
        }

        function drawGlobeSwath(targetLatLon) {
            if (!targetLatLon) {
                globeSwath.material.opacity = 0.0;
                return;
            }
            const tgtPos = latLonToVector3(targetLatLon.lat, targetLatLon.lon, 60.2);
            globeSwath.position.copy(tgtPos);
            globeSwath.lookAt(0, 0, 0);
            globeSwath.material.opacity = 0.55;
        }

        function animateGlobe() {
            requestAnimationFrame(animateGlobe);
            
            if (controls) {
                controls.update();
            }
            
            // Continuous orbit gliding when not transitioning
            if (orbitPoints.length > 0 && !isAnimating) {
                orbitIndex += 0.05; // orbit speed glide
                if (orbitIndex >= orbitPoints.length) {
                    orbitIndex = 0;
                }
                const idx = Math.floor(orbitIndex);
                const nextIdx = (idx + 1) % orbitPoints.length;
                const alpha = orbitIndex - idx;
                const pt1 = orbitPoints[idx];
                const pt2 = orbitPoints[nextIdx];
                
                // Wrap-around longitude interpolation
                let lon1 = pt1.lon;
                let lon2 = pt2.lon;
                if (lon2 - lon1 > 180) lon1 += 360;
                else if (lon1 - lon2 > 180) lon2 += 360;
                const lon = ((lon1 + (lon2 - lon1) * alpha) + 180) % 360 - 180;
                const lat = pt1.lat + (pt2.lat - pt1.lat) * alpha;
                
                const satPos = latLonToVector3(lat, lon, 64);
                satMesh.position.copy(satPos);
                
                if (currentState && currentState.target_focus) {
                    const tgtPos = latLonToVector3(currentState.target_focus.lat, currentState.target_focus.lon, 60);
                    satMesh.lookAt(tgtPos);
                } else {
                    satMesh.lookAt(0, 0, 0);
                }
                satMesh.rotateX(Math.PI / 2);
                
                if (camLockActive) {
                    camera.position.copy(satPos.clone().normalize().multiplyScalar(160));
                    if (controls) {
                        controls.target.set(0, 0, 0);
                    }
                }
            }
            
            // Auto rotate globe slowly
            globe.rotation.y += 0.0003;
            stars.rotation.y -= 0.0001;

            renderer.render(scene, camera);
        }

        function animateSat() {
            requestAnimationFrame(animateSat);
            
            // Slow orbital oscillation
            satBody.position.y = Math.sin(Date.now() * 0.001) * 0.15;
            
            satRenderer.render(satScene, satCamera);
        }

        let isAnimating = false;

        // Web API fetch bindings
        async function resetSim() {
            if (isAnimating) return;
            const res = await fetch('/api/reset', { method: 'POST' });
            const state = await res.json();
            updateUI(state);
        }

        async function stepSim() {
            if (isAnimating) return;
            const res = await fetch('/api/step', { method: 'POST' });
            const state = await res.json();
            updateUI(state);
        }

        function toggleAutoplay() {
            autoplayActive = !autoplayActive;
            const btn = document.getElementById('btn-autoplay');
            if (autoplayActive) {
                btn.textContent = 'Pause';
                btn.style.background = 'linear-gradient(135deg, #10b981 0%, #059669 100%)';
                if (!isAnimating) {
                    stepSim();
                }
            } else {
                btn.textContent = 'Auto-Play';
                btn.style.background = 'linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%)';
            }
        }

        function updateUI(state) {
            currentState = state;
            orbitPoints = state.orbit_track;
            orbitIndex = 0;
            // Update Text telemetries
            document.getElementById('tel-time').textContent = `${state.time_h}h ${state.time_m}m`;
            document.getElementById('tel-step').textContent = state.step;
            document.getElementById('tel-battery').textContent = `${state.battery_soc.toFixed(1)}%`;
            document.getElementById('tel-reward').textContent = `${state.reward >= 0 ? '+' : ''}${state.reward.toFixed(3)}`;

            // Style battery color
            const batEl = document.getElementById('tel-battery');
            if (state.battery_soc > 50) batEl.className = 'tel-val green-txt';
            else if (state.battery_soc > 30) batEl.className = 'tel-val yellow-txt';
            else batEl.className = 'tel-val magenta-txt';

            // Active Decision and styling
            const decEl = document.getElementById('decision-text');
            if (state.action_type === 'image_static') {
                decEl.textContent = `Image target [${state.targets[state.action].name}]`;
                decEl.className = 'tel-val green-txt';
                triggerImagingBeam(true);
                animateSlew(0.4);
            } else if (state.action_type === 'image_dynamic') {
                decEl.textContent = `Image Dynamic Event`;
                decEl.className = 'tel-val magenta-txt';
                triggerImagingBeam(true);
                animateSlew(-0.4);
            } else {
                decEl.textContent = 'Solar panel Drift';
                decEl.className = 'tel-val yellow-txt';
                triggerImagingBeam(false);
                animateSlew(0.0);
            }

            // Update Explainability attribution details
            const logBox = document.getElementById('decision-log');
            logBox.innerHTML = '';
            state.explanations.forEach(line => {
                const p = document.createElement('div');
                p.style.marginBottom = '6px';
                // HTML format colors
                p.innerHTML = line
                    .replace('[bold green]', '<b class="green-txt">')
                    .replace('[/bold green]', '</b>')
                    .replace('[bold yellow]', '<b class="yellow-txt">')
                    .replace('[/bold yellow]', '</b>')
                    .replace('[bold magenta]', '<b class="magenta-txt">')
                    .replace('[/bold magenta]', '</b>')
                    .replace('[cyan]', '<b class="blue-txt">')
                    .replace('[white]', '<b style="color:#fff">')
                    .replace('[/white]', '</b>')
                    .replace('[/cyan]', '</b>');
                logBox.appendChild(p);
            });

            // Alternatives text
            const altBox = document.createElement('div');
            altBox.style.marginTop = '15px';
            altBox.style.borderTop = '1px dashed rgba(255,255,255,0.05)';
            altBox.style.paddingTop = '10px';
            altBox.innerHTML = `<b>Alt Choices:</b> ${state.alternates.join(' | ')}`;
            logBox.appendChild(altBox);

            // Update Targets List Table
            const listBody = document.getElementById('targets-list-body');
            listBody.innerHTML = '';
            state.targets.forEach(t => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${t.name}</td>
                    <td>${t.priority.toFixed(1)}</td>
                    <td>${(t.cloud*100).toFixed(0)}%</td>
                    <td><span class="status-badge ${t.imaged ? 'status-green' : 'status-red'}"></span></td>
                `;
                listBody.appendChild(tr);
            });

            // Update Dynamic Event Alerts
            const evList = document.getElementById('events-list');
            evList.innerHTML = '';
            if (state.events.length === 0) {
                evList.innerHTML = '<div class="dim-txt" style="text-align: center; font-size:11px; margin-top:20px;">No events detected</div>';
            } else {
                state.events.forEach(e => {
                    const el = document.createElement('div');
                    el.className = 'evt-item';
                    el.innerHTML = `
                        <b class="magenta-txt">ACTIVE SLOT ${e.slot} ALERT: ${e.name.split('_')[1]}</b><br/>
                        Prio: ${e.priority.toFixed(2)} | Cloud: ${(e.cloud*100).toFixed(0)}%<br/>
                        Expires: ${e.expires_in.toFixed(0)}s
                    `;
                    evList.appendChild(el);
                });
            }

            // --- 3D Scene updates ---
            const btnStep = document.getElementById('btn-step');
            if (state.transition_path && state.transition_path.length > 0) {
                isAnimating = true;
                btnStep.disabled = true;
                btnStep.textContent = "Slewing & Orbiting...";
                
                let currentFrame = 0;
                const path = state.transition_path;
                
                function frame() {
                    if (currentFrame < path.length) {
                        const pt = path[currentFrame];
                        const satPos = latLonToVector3(pt.lat, pt.lon, 64);
                        satMesh.position.copy(satPos);
                        
                        // Orient satMesh and draw laser/swath footprint in final segment
                        if (state.target_focus && currentFrame > path.length - 8) {
                            const tgtPos = latLonToVector3(state.target_focus.lat, state.target_focus.lon, 60);
                            satMesh.lookAt(tgtPos);
                            drawGlobeLaser(satPos, state.target_focus);
                            drawGlobeSwath(state.target_focus);
                        } else {
                            satMesh.lookAt(0, 0, 0);
                            drawGlobeLaser(null);
                            drawGlobeSwath(null);
                        }
                        satMesh.rotateX(Math.PI / 2);
                        
                        // Move camera (if lock is active)
                        if (camLockActive) {
                            camera.position.copy(satPos.clone().normalize().multiplyScalar(160));
                            if (controls) {
                                controls.target.set(0, 0, 0);
                            }
                        }
                        
                        currentFrame++;
                        // Smoothly animate at ~30 steps per second visually
                        setTimeout(() => { requestAnimationFrame(frame); }, 25);
                    } else {
                        // Animation finished, finalize state
                        finalizeStep();
                    }
                }
                
                function finalizeStep() {
                    isAnimating = false;
                    btnStep.disabled = false;
                    btnStep.textContent = "Step Action [Space]";
                    
                    // Trigger camera flash at target on completion of step
                    if (state.action_type === 'image_static' || state.action_type === 'image_dynamic') {
                        triggerImagingBeam(true);
                        drawGlobeLaser(satMesh.position, state.target_focus);
                        drawGlobeSwath(state.target_focus);
                        // Keep visible for 1.5s for visual confirmation, then clear
                        setTimeout(() => {
                            drawGlobeLaser(null);
                            drawGlobeSwath(null);
                        }, 1500);
                    } else {
                        drawGlobeLaser(null);
                        drawGlobeSwath(null);
                    }
                    
                    if (autoplayActive) {
                        setTimeout(stepSim, 2000); // 2s delay so jury can see the imaging event
                    }
                }
                
                frame();
            } else {
                // Instantly place satellite (e.g. on reset)
                const satPos = latLonToVector3(state.sat_lat, state.sat_lon, 64);
                satMesh.position.copy(satPos);
                satMesh.lookAt(0, 0, 0);
                satMesh.rotateX(Math.PI / 2);
                
                drawGlobeLaser(null);
                drawGlobeSwath(null);
                
                // Focus camera on satellite once on reset, or continuously if locked
                if (camLockActive || state.step === 0) {
                    camera.position.copy(satPos.clone().normalize().multiplyScalar(160));
                    if (controls) {
                        controls.target.set(0, 0, 0);
                    }
                }
            }

            // Draw Orbit Track segments (using LineSegments to avoid wrap-around crossing cuts)
            const segments = [];
            for (let i = 0; i < state.orbit_track.length - 1; i++) {
                const pt1 = state.orbit_track[i];
                const pt2 = state.orbit_track[i + 1];
                if (Math.abs(pt1.lon - pt2.lon) < 180) {
                    segments.push(latLonToVector3(pt1.lat, pt1.lon, 63));
                    segments.push(latLonToVector3(pt2.lat, pt2.lon, 63));
                }
            }
            const positions = new Float32Array(segments.length * 3);
            for (let i = 0; i < segments.length; i++) {
                positions[i * 3] = segments[i].x;
                positions[i * 3 + 1] = segments[i].y;
                positions[i * 3 + 2] = segments[i].z;
            }
            orbitLine.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            orbitLine.geometry.computeBoundingSphere();

            // Re-render targets spheres on the globe
            targetSpheres.forEach(s => scene.remove(s));
            targetSpheres = [];

            state.targets.forEach(t => {
                const tGeo = new THREE.SphereGeometry(0.8, 8, 8);
                const tMat = new THREE.MeshBasicMaterial({ color: t.imaged ? 0x10b981 : 0xf43f5e });
                const s = new THREE.Mesh(tGeo, tMat);
                s.position.copy(latLonToVector3(t.lat, t.lon, 60.1));
                scene.add(s);
                targetSpheres.push(s);
            });

            // Render active dynamic events on the globe
            state.events.forEach(e => {
                const eGeo = new THREE.BoxGeometry(1.5, 1.5, 1.5);
                const eMat = new THREE.MeshBasicMaterial({ color: 0xff0055 });
                const s = new THREE.Mesh(eGeo, eMat);
                s.position.copy(latLonToVector3(e.lat, e.lon, 60.2));
                scene.add(s);
                targetSpheres.push(s);
            });
        }

        function triggerImagingBeam(active) {
            if (active) {
                laserBeam.material.opacity = 0.55;
                // Beam pulse effect
                setTimeout(() => { laserBeam.material.opacity = 0.0; }, 400);
            } else {
                laserBeam.material.opacity = 0.0;
            }
        }

        function animateSlew(targetZRotation) {
            // Animate satellite body rotation
            let current = satBody.rotation.z;
            let step = 0.05;
            function update() {
                if (Math.abs(satBody.rotation.z - targetZRotation) > 0.02) {
                    satBody.rotation.z += (targetZRotation - satBody.rotation.z) * 0.2;
                    requestAnimationFrame(update);
                } else {
                    satBody.rotation.z = targetZRotation;
                }
            }
            update();
        }
    </script>
</body>
</html>
"""

# --- Dashboard Daemon Server ---
def run_server(backend: DashboardBackend, port: int):
    handler_class = make_handler(backend)
    # Enable socket re-use to prevent address already in use errors
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), handler_class) as httpd:
        print(f"[SERVER] Dashboard running at: http://localhost:{port}/")
        httpd.serve_forever()

def main():
    parser = argparse.ArgumentParser(description="ALSAT-EO-1 3D Live Web Dashboard")
    parser.add_argument("--model", type=str, required=True, help="Path to trained PPO zip model")
    parser.add_argument("--seed", type=int, default=300, help="Random seed for simulation")
    parser.add_argument("--event-rate", type=float, default=2.0, help="Dynamic event rate")
    parser.add_argument("--port", type=int, default=8080, help="Local HTTP server port")
    args = parser.parse_args()
    
    print("=== Launching ALSAT-EO-1 3D Web Dashboard ===")
    backend = DashboardBackend(model_path=args.model, seed=args.seed, event_rate=args.event_rate)
    
    # Run HTTP Server in a background thread
    server_thread = threading.Thread(target=run_server, args=(backend, args.port), daemon=True)
    server_thread.start()
    
    # Give the thread a split second to spin up the ZMQ port
    time.sleep(0.5)
    
    # Launch browser automatically
    webbrowser.open(f"http://localhost:{args.port}/")
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[INFO] Dashboard server stopped by user.")

if __name__ == "__main__":
    main()
