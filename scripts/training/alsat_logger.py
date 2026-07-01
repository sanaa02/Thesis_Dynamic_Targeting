#!/usr/bin/env python3
"""
alsat_logger.py  --  ALSAT-EO-1  Organised Training Logger
===========================================================
Provides a clean, structured view of everything happening during training
without flooding the terminal or creating huge log files.

OUTPUT
------
  TERMINAL:      One compact line per iteration (episode, reward, loss, entropy, ETA)
  logs/training.csv        Machine-readable per-iteration metrics
  logs/episodes.jsonl      Per-episode summary (reward, imaged, battery, etc.)
  logs/decisions.log       Per-step decisions (target chosen, event chosen, reward)
  logs/orbit.log           Satellite position every ORBIT_LOG_EVERY steps

USAGE
-----
  from alsat_logger import ALSATLogger, make_loggers
  callbacks = make_loggers(total_steps=1_000_000)
  model.learn(..., callback=callbacks, verbose=0)
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import sys
import time
from typing import Optional

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
for _d in ["scripts/core", "scripts/training", "scripts"]:
    _p = os.path.join(_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from stable_baselines3.common.callbacks import BaseCallback
    _HAS_SB3 = True
except ImportError:
    _HAS_SB3 = False
    class BaseCallback:   # type: ignore
        def __init__(self, verbose=0): self.verbose = verbose
        def _on_step(self): return True

# ── Algeria geography ─────────────────────────────────────────────────────────
try:
    from algerian_geography import lookup_location, ecef_to_latlon
    _HAS_GEO = True
except ImportError:
    _HAS_GEO = False

# ── Target names from config ──────────────────────────────────────────────────
_TARGET_NAMES: list[str] = []

def _load_target_names(targets_path: str) -> None:
    global _TARGET_NAMES
    try:
        with open(targets_path) as f:
            data = json.load(f)
        if isinstance(data, list):
            _TARGET_NAMES = [t.get("name", f"T{i:02d}") for i, t in enumerate(data)]
        elif isinstance(data, dict) and "targets" in data:
            _TARGET_NAMES = [t.get("name", f"T{i:02d}") for i, t in enumerate(data["targets"])]
    except Exception:
        _TARGET_NAMES = [f"Target-{i:02d}" for i in range(24)]

_TARGETS_JSON = os.path.join(_ROOT, "config/targets/global_45_targets.json")
_load_target_names(_TARGETS_JSON)

N_STATIC = 45
N_DYN    = 3

# ── Config ────────────────────────────────────────────────────────────────────
ORBIT_LOG_EVERY  = 20    # write orbit position every N env steps
LOGS_DIR         = os.path.join(_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)


def _action_label(action: int, env = None) -> str:
    n_static = N_STATIC
    n_dyn = N_DYN
    target_names = _TARGET_NAMES
    
    if env is not None:
        try:
            unwrapped = _unwrap_vecenv(env)
            if hasattr(unwrapped, "satellites") and unwrapped.satellites:
                sat = unwrapped.satellites[0]
                n_static = len(sat.scenario.targets)
                target_names = [t.name for t in sat.scenario.targets]
        except Exception:
            pass

    if action < n_static:
        name = target_names[action] if action < len(target_names) else f"T{action:02d}"
        return f"static   target={name} (idx={action})"
    elif action < n_static + n_dyn:
        return f"dynamic  slot={action - n_static}"
    else:
        return "drift"


def _unwrap_vecenv(env):
    """
    Unwrap a VecEnv + gymnasium wrapper stack to find the raw base env.

    SB3 wraps the user's env in DummyVecEnv (or SubprocVecEnv).
    DummyVecEnv stores actual envs in .envs[0]  — NOT .env.
    VecNormalize (optional) adds another layer via .venv.
    Gymnasium wrappers chain via .env.
    """
    raw = env

    # ── Layer 1: VecEnv shell ─────────────────────────────────────────────────
    # VecNormalize wraps a VecEnv in .venv
    for _ in range(4):
        if hasattr(raw, "venv"):
            raw = raw.venv
        else:
            break

    # DummyVecEnv / SubprocVecEnv stores actual envs in .envs list
    if hasattr(raw, "envs") and raw.envs:
        raw = raw.envs[0]

    # ── Layer 2: gymnasium wrapper chain (.env) ────────────────────────────────
    for _ in range(20):
        if hasattr(raw, "satellites"):
            return raw
        nxt = getattr(raw, "env", None)
        if nxt is None or nxt is raw:
            # Try .unwrapped as last resort
            uw = getattr(raw, "unwrapped", None)
            if uw is not None and uw is not raw:
                return uw
            break
        raw = nxt

    return raw


def _get_sat_state(env, info: Optional[dict] = None) -> dict:
    """
    Extract satellite physical state from anywhere in the env stack or info.
    Returns an empty dict on any failure — caller must handle defaults.

    Keys returned (all optional):
      battery_pct  : float  0-100
      sim_time_s   : float  seconds since episode start
      lat, lon     : float  degrees
      alt_km       : float
      location     : str    wilaya / country / sea name
    """
    try:
        # Check if we have pre-recorded raw state in info to bypass post-reset state
        if info is not None and "sat_state_raw" in info:
            raw = info["sat_state_raw"]
            state: dict = {}
            batt = raw.get("battery_charge_fraction")
            if batt is not None and not math.isnan(float(batt)):
                state["battery_pct"] = round(float(batt) * 100, 1)
            sim_time = raw.get("sim_time")
            if sim_time is not None:
                state["sim_time_s"] = float(sim_time)
            
            if _HAS_GEO and "r_SC_N" in raw:
                try:
                    r_ecef = np.asarray(raw["r_SC_N"], dtype=float).flatten()
                    lat, lon, alt_km = ecef_to_latlon(r_ecef)
                    state["lat"]      = round(lat, 3)
                    state["lon"]      = round(lon, 3)
                    state["alt_km"]   = round(alt_km, 1)
                    loc = lookup_location(lat, lon)
                    state["location"] = loc["label"]
                    state["country"]  = loc["country"]
                    state["wilaya"]   = loc.get("wilaya")
                except Exception:
                    pass
            return state

        unwrapped = _unwrap_vecenv(env)

        if not hasattr(unwrapped, "satellites"):
            return {}

        sat = unwrapped.satellites[0]
        state: dict = {}

        # ── Battery ──────────────────────────────────────────────────────────
        # Canonical path confirmed: sat.dynamics.battery_charge_fraction
        # (matches callbacks.py and safety_monitor.py in this codebase)
        batt = None
        try:
            batt = float(sat.dynamics.battery_charge_fraction)
        except Exception:
            pass
        if batt is None:
            try:
                batt = float(sat.battery_charge_fraction)   # fallback: satellite shortcut
            except Exception:
                pass
        if batt is not None and not math.isnan(batt):
            state["battery_pct"] = round(batt * 100, 1)

        # ── Simulation time ───────────────────────────────────────────────────
        sim = getattr(sat, "simulator", None)
        if sim is not None:
            state["sim_time_s"] = float(getattr(sim, "sim_time", 0.0))

        # ── Position (ECEF → lat/lon/alt) ─────────────────────────────────────
        if _HAS_GEO:
            try:
                dyn = getattr(sat, "dynamics", None)
                if dyn is not None:
                    r_ecef = np.asarray(
                        getattr(dyn, "r_SC_N", None) or
                        getattr(dyn, "r_BN_N", None) or
                        getattr(dyn, "r_N",    None),
                        dtype=float,
                    ).flatten()
                    lat, lon, alt_km = ecef_to_latlon(r_ecef)
                    state["lat"]      = round(lat, 3)
                    state["lon"]      = round(lon, 3)
                    state["alt_km"]   = round(alt_km, 1)
                    loc = lookup_location(lat, lon)
                    state["location"] = loc["label"]
                    state["country"]  = loc["country"]
                    state["wilaya"]   = loc.get("wilaya")
            except Exception:
                pass

        return state

    except Exception:
        return {}


class _CSVWriter:
    def __init__(self, path: str, fieldnames: list[str]):
        self._path   = path
        self._fields = fieldnames
        self._file   = None
        self._writer = None

    def _open(self):
        if self._writer is None:
            exists = os.path.exists(self._path)
            self._file   = open(self._path, "a", newline="", buffering=1)
            self._writer = csv.DictWriter(self._file, fieldnames=self._fields,
                                          extrasaction="ignore")
            if not exists:
                self._writer.writeheader()

    def write(self, row: dict):
        self._open()
        self._writer.writerow(row)

    def close(self):
        if self._file:
            self._file.close()


class _JSONLWriter:
    def __init__(self, path: str):
        self._path = path
        self._file = None

    def write(self, obj: dict):
        if self._file is None:
            self._file = open(self._path, "a", buffering=1)
        self._file.write(json.dumps(obj, default=float) + "\n")

    def close(self):
        if self._file:
            self._file.close()


class _LogWriter:
    def __init__(self, path: str):
        self._path = path
        self._file = None

    def write(self, line: str):
        if self._file is None:
            self._file = open(self._path, "a", buffering=1)
        self._file.write(line + "\n")
        self._file.flush()

    def close(self):
        if self._file:
            self._file.close()


class ALSATLogger(BaseCallback):
    """
    SB3 Callback: clean terminal output + organised log files.

    Parameters
    ----------
    total_timesteps : int   total training steps (for ETA calculation)
    log_dir         : str   directory for log files (default: logs/)
    orbit_every     : int   write orbit position every N env steps
    stage_label     : str   curriculum stage label shown in terminal
    """

    def __init__(
        self,
        total_timesteps: int  = 1_000_000,
        log_dir:         str  = LOGS_DIR,
        orbit_every:     int  = ORBIT_LOG_EVERY,
        stage_label:     str  = "",
        fresh_logs:      bool = False,
        start_ep:        int  = 0,
        start_step:      int  = 0,
        verbose:         int  = 0,
    ):
        super().__init__(verbose=verbose)
        self.total_timesteps = total_timesteps
        self.log_dir         = log_dir
        self.orbit_every     = orbit_every
        self.stage_label     = stage_label

        os.makedirs(log_dir, exist_ok=True)

        # ── Optionally clear old log files ────────────────────────────────────
        if fresh_logs:
            for fname in ("training.csv", "episodes.jsonl",
                          "decisions.log", "orbit.log"):
                p = os.path.join(log_dir, fname)
                if os.path.exists(p):
                    os.remove(p)

        # ── Writers ──────────────────────────────────────────────────────────
        self._train_csv  = _CSVWriter(
            os.path.join(log_dir, "training.csv"),
            ["step", "iteration", "ep", "ep_len", "reward",
             "loss", "pg_loss", "vf_loss", "entropy", "kl",
             "explained_var", "clip_frac", "lr", "fps",
             "n_dyn_imaged_ep", "n_dyn_detected_ep", "battery_end_pct"],
        )
        self._ep_log   = _JSONLWriter(os.path.join(log_dir, "episodes.jsonl"))
        self._dec_log  = _LogWriter(os.path.join(log_dir, "decisions.log"))
        self._orb_log  = _LogWriter(os.path.join(log_dir, "orbit.log"))

        # ── Run-start separator (so multiple appended runs are clearly split) ─
        import datetime
        _ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _hdr = (f"\n{'='*72}\n"
                f"  RUN START  {_ts}  stage={stage_label or 'all'}  "
                f"total_steps={total_timesteps:,}\n"
                f"{'='*72}")
        self._dec_log.write(_hdr)
        self._orb_log.write(_hdr)

        # ── Internal state ───────────────────────────────────────────────────
        self._t0              = time.time()
        self._last_iter_t     = time.time()   # for self-timed FPS
        self._last_iter_step  = 0             # step count at last iteration
        self._ep_num          = start_ep
        self._step_in_ep      = 0  # sub-step counter (SB3 on_step calls)
        self._smdp_dec_in_ep  = 0  # SMDP scheduler decisions (what ep_len means)
        self._log_seq         = 0  # monotone counter for ordering episodes.jsonl
        self._global_step     = start_step
        self._iteration       = 0
        self._ep_r            = 0.0
        self._ep_actions: list[int] = []
        self._last_ep_metrics: dict = {}
        self._prev_ep_len     = 0.0
        self._prev_ep_rew     = 0.0
        self._last_batt_pct: Optional[float] = 100.0
        self._last_batt_end:  Optional[float] = None

        # Load last episode metrics if episodes.jsonl exists and is not empty
        ep_file = os.path.join(log_dir, "episodes.jsonl")
        if not fresh_logs and os.path.exists(ep_file) and os.path.getsize(ep_file) > 0:
            try:
                import json
                with open(ep_file, "r") as f:
                    lines = f.readlines()
                    if lines:
                        last_line = ""
                        for line in reversed(lines):
                            if line.strip():
                                last_line = line.strip()
                                break
                        if last_line:
                            last_ep = json.loads(last_line)
                            self._prev_ep_len = float(last_ep.get("ep_len", 0.0))
                            self._prev_ep_rew = float(last_ep.get("total_reward", 0.0))
                            self._last_ep_metrics = {
                                "n_dyn_imaged": last_ep.get("n_dyn_imaged", 0),
                                "n_dyn_detected": last_ep.get("n_dyn_detected", 1),
                                "n_static_imaged_clean": last_ep.get("n_static_imaged", 0),
                            }
                            self._last_batt_end = last_ep.get("battery_end_pct", 100.0)
                            self._last_batt_pct = self._last_batt_end
                            self._log_seq = int(last_ep.get("log_seq", 0))
            except Exception:
                pass

        # ── Header banner ─────────────────────────────────────────────────────
        label = f"  [{self.stage_label}]" if self.stage_label else ""
        print(f"\n{'─'*72}")
        print(f"  ALSAT-EO-1 Training{label}")
        print(f"  Logs → {log_dir}/")
        print(f"  {'training.csv':20s} — per-iteration metrics")
        print(f"  {'episodes.jsonl':20s} — per-episode summary")
        print(f"  {'decisions.log':20s} — per-step action log")
        print(f"  {'orbit.log':20s} — satellite position (every {orbit_every} steps)")
        print(f"{'─'*72}")
        print(f"  {'Step':>10}  {'Ep':>5}  {'EpLen':>6}  {'Reward':>8}  "
              f"{'Loss':>7}  {'Entropy':>8}  {'DynSuc':>7}  {'FPS':>5}  ETA")
        print(f"  {'─'*10}  {'─'*5}  {'─'*6}  {'─'*8}  "
              f"{'─'*7}  {'─'*8}  {'─'*7}  {'─'*5}  {'─'*8}")

    # ── SB3 Callback interface ────────────────────────────────────────────────

    def _on_step(self) -> bool:
        self._global_step += 1
        self._step_in_ep  += 1

        # ── Extract step info ─────────────────────────────────────────────────
        try:
            info   = self.locals.get("infos", [{}])[0]
            action = int(self.locals.get("actions", [23])[0])
            # FIX-LOG-RAW: VecNormalize normalizes rewards before storing in locals['rewards'].
            # Early in training the running std is tiny → -0.10 penalty ≈ 0.000 after norm.
            # Read raw reward from VecNormalize.old_reward (unnormalized) if available.
            _raw_reward = None
            try:
                _vn = self.training_env
                if hasattr(_vn, 'old_reward') and len(_vn.old_reward) > 0:
                    _raw_reward = float(_vn.old_reward[0])
            except Exception:
                pass
            if _raw_reward is None:
                _raw_reward = float(info.get('raw_reward', self.locals.get('rewards', [0.0])[0]))
            reward = _raw_reward
            done   = bool(self.locals.get("dones", [False])[0])
        except Exception:
            info = {}; action = 23; reward = 0.0; done = False

        # Count actual SMDP scheduler decisions (smdp_tau_s present = real decision)
        if info.get("smdp_tau_s") is not None:
            self._smdp_dec_in_ep += 1

        self._ep_r += reward
        self._ep_actions.append(action)

        # ── decisions.log: every step ─────────────────────────────────────────
        try:
            sat_state = _get_sat_state(self.training_env, info)
            sim_t     = sat_state.get("sim_time_s", None)
            batt_pct  = sat_state.get("battery_pct", None)

            if batt_pct is not None:
                self._last_batt_pct = batt_pct

            tau_s     = info.get("smdp_tau_s", info.get("tau", 0.0))
            dyn_r     = info.get("dyn_event_imaged", {})

            # FIX-5: count valid (unmasked) static targets for this step
            _n_valid_static = 0
            try:
                _amask = info.get("action_masks", None)
                if _amask is None:
                    _amask = self.locals.get("action_masks", None)
                if _amask is not None:
                    _mask_arr = (
                        _amask[0] if hasattr(_amask, "__len__") and
                        hasattr(_amask[0], "__len__") else _amask
                    )
                    _n_static_local = N_STATIC
                    try:
                        _unwrapped_local = _unwrap_vecenv(self.training_env)
                        if hasattr(_unwrapped_local, "satellites") and _unwrapped_local.satellites:
                            _n_static_local = len(_unwrapped_local.satellites[0].scenario.targets)
                    except Exception:
                        pass
                    _n_valid_static = int(sum(
                        bool(_mask_arr[i]) for i in range(min(len(_mask_arr), _n_static_local))
                    ))
            except Exception:
                _n_valid_static = -1

            sim_t_str  = f"{sim_t:7.0f}" if sim_t is not None else "      ?"
            batt_val   = batt_pct if batt_pct is not None else self._last_batt_pct
            batt_str   = f"{batt_val:.1f}" if batt_val is not None else "?"
            n_static = N_STATIC
            n_dyn = N_DYN
            try:
                unwrapped = _unwrap_vecenv(self.training_env)
                if hasattr(unwrapped, "satellites") and unwrapped.satellites:
                    sat = unwrapped.satellites[0]
                    n_static = len(sat.scenario.targets)
            except Exception:
                pass

            if action >= n_static and action < n_static + n_dyn:
                if dyn_r:
                    act_label = f"dynamic  slot={action - n_static} event={dyn_r.get('name', 'unknown')} (cloud={dyn_r.get('cloud', 0.0)*100:.0f}%)"
                else:
                    act_label = f"dynamic  slot={action - n_static} (no image / failed)"
            elif action < n_static:
                # FIX-LOG-1: distinguish successful static imaging from failed attempts
                _base_act_label = _action_label(action, self.training_env)
                # reward > 0.05 means image was actually taken and credited
                if abs(reward) < 0.001:
                    act_label = _base_act_label + " (failed/slewing)"
                elif reward < 0:
                    act_label = _base_act_label + " (cloudy/penalty)"
                else:
                    act_label = _base_act_label + " ✓"
            else:
                act_label  = _action_label(action, self.training_env)
            # FIX-LOG-2: normalize -0.000 / +0.000 display (floating-point artefact)
            _r_display = 0.0 if abs(reward) < 5e-4 else reward
            _valid_str = f"{_n_valid_static:2d}" if _n_valid_static >= 0 else " ?"
            dec_line   = (
                f"[ep={self._ep_num:4d} step={self._step_in_ep:3d} "
                f"t={sim_t_str}s τ={tau_s:5.0f}s] "
                f"{act_label:<52s} "
                f"r={_r_display:+.3f}  batt={batt_str}%  valid_static={_valid_str}"
            )
            if dyn_r:
                dec_line += (
                    f"  ★DYN lat={dyn_r.get('lat',0):+.2f} "
                    f"lon={dyn_r.get('lon',0):+.2f}"
                )
            self._dec_log.write(dec_line)
        except Exception:
            pass

        # ── orbit.log: every orbit_every steps ───────────────────────────────
        if self._step_in_ep % self.orbit_every == 0:
            try:
                sat_state = _get_sat_state(self.training_env, info)
                sim_t     = sat_state.get("sim_time_s", None)
                lat       = sat_state.get("lat",       None)
                lon       = sat_state.get("lon",       None)
                alt       = sat_state.get("alt_km",    None)
                loc       = sat_state.get("location",  "unknown")
                batt_pct  = sat_state.get("battery_pct", None)

                sim_t_str = f"{sim_t/3600:5.2f}h" if sim_t is not None else "   ?h"
                lat_str   = f"{lat:+7.3f}°" if lat is not None else "      ?°"
                lon_str   = f"{lon:+8.3f}°" if lon is not None else "       ?°"
                alt_str   = f"{alt:5.1f}km" if alt is not None else "    ?km"
                bat_str   = f"{batt_pct:5.1f}%" if batt_pct is not None else "    ?%"

                orb_line  = (
                    f"[ep={self._ep_num:4d} step={self._step_in_ep:3d} "
                    f"t={sim_t_str}]  "
                    f"lat={lat_str} lon={lon_str} alt={alt_str}  "
                    f"→  {loc:<30s}  batt={bat_str}"
                )
                self._orb_log.write(orb_line)
            except Exception:
                pass

        # ── Episode end ───────────────────────────────────────────────────────
        if done:
            # Skip logging transient or uninitialized resets (length <= 1)
            if self._step_in_ep <= 1:
                self._prev_ep_len = self._smdp_dec_in_ep
                self._prev_ep_rew = self._ep_r
                self._step_in_ep  = 0
                self._smdp_dec_in_ep = 0
                self._ep_r        = 0.0
                self._ep_actions  = []
                self._last_batt_pct = 100.0
                return True

            self._ep_num += 1
            self._log_seq += 1
            ep_metrics = info.get("episode_metrics", {})
            self._last_ep_metrics = ep_metrics
            batt_end = self._last_batt_pct
            self._last_batt_end = batt_end

            n_dyn_det = ep_metrics.get("n_dyn_detected", 0)
            n_dyn_clean = ep_metrics.get("n_dyn_imaged_clean", 0)
            n_static_clean = ep_metrics.get("n_static_imaged_clean", 0)
            dyn_suc_rate = float(n_dyn_clean / n_dyn_det) if n_dyn_det > 0 else 0.0
            batt_end_val = float(batt_end) if (batt_end is not None and not math.isnan(float(batt_end))) else 100.0

            # ep_len = SMDP scheduler decisions (not sub-steps)
            # sub_steps = self._step_in_ep  (raw SB3 on_step calls, = sub-steps)
            ep_decisions = self._smdp_dec_in_ep if self._smdp_dec_in_ep > 0 else self._step_in_ep

            ep_summary = {
                "log_seq":               self._log_seq,       # monotone ordering key
                "ep":                    self._ep_num,
                "global_step":           self._global_step,
                "ep_len":                ep_decisions,        # SMDP decisions, not sub-steps
                "ep_substeps":           self._step_in_ep,    # raw sub-steps (30s each)
                "total_reward":          round(self._ep_r, 4),
                "n_imaged":              ep_metrics.get("n_imaged",       0),
                "n_static_imaged":       n_static_clean,
                "n_static_total":        45,                  # total static targets
                "static_completion_pct": round(n_static_clean / 45.0 * 100, 1),  # % of 45 targets
                "static_coverage_bonus": round(ep_metrics.get("static_coverage_bonus_total", 0.0), 2),
                "n_dyn_imaged":          n_dyn_clean,
                "n_dyn_detected":        n_dyn_det,
                "dyn_suc":               round(dyn_suc_rate, 4),  # n_dyn_imaged / n_dyn_detected
                "n_missed_events":       ep_metrics.get("n_missed_events",0),
                "n_cloud_free":          ep_metrics.get("n_cloud_free",   0),
                "n_cloudy":              ep_metrics.get("n_cloudy",       0),
                "total_slew_deg":        round(ep_metrics.get("total_slew_angle_deg",  0.0), 1),
                "total_slew_wh":         round(ep_metrics.get("total_slew_energy_wh",  0.0), 3),
                "battery_end_pct":       round(batt_end_val, 1),
                "action_counts":         _action_histogram(self._ep_actions, n_static, n_dyn),
            }
            self._ep_log.write(ep_summary)

            # Log episode end summary (use scheduler decisions for readability)
            dyn_suc = dyn_suc_rate * 100
            static_pct = n_static_clean / 45.0 * 100
            warn = "  ⚠ SHORT EP" if ep_decisions < 10 else ""
            warn += "  ⚠ PURE DRIFT" if (n_dyn_clean == 0 and n_static_clean == 0 and ep_decisions > 50) else ""
            self._dec_log.write(
                f"  ↳ EPISODE END  decisions={ep_decisions} substeps={self._step_in_ep}  "
                f"reward={self._ep_r:+.3f}  "
                f"static={n_static_clean}/45 ({static_pct:.0f}%)  "
                f"dyn_suc={dyn_suc:.0f}% ({n_dyn_clean}/{n_dyn_det})  "
                f"batt={batt_end_val:.1f}%{warn}"
            )

            self._prev_ep_len = ep_decisions
            self._prev_ep_rew = self._ep_r
            self._step_in_ep  = 0
            self._smdp_dec_in_ep = 0
            self._ep_r        = 0.0
            self._ep_actions  = []
            self._last_batt_pct = 100.0

        return True

    def _on_rollout_end(self) -> None:
        self._iteration += 1

        # ── FPS: timed by us, not from SB3 logger ────────────────────────────
        # (SB3's time/fps is not reliably populated at _on_rollout_end time)
        now   = time.time()
        dt    = now - self._last_iter_t
        steps = self._global_step - self._last_iter_step
        fps   = int(steps / dt) if dt > 0.001 else 0
        self._last_iter_t    = now
        self._last_iter_step = self._global_step

        # ── Pull train/* from SB3 logger (available after policy update) ──────
        try:
            sb3_log = self.logger.name_to_value
        except Exception:
            sb3_log = {}

        loss      = sb3_log.get("train/loss",                 float("nan"))
        pg_loss   = sb3_log.get("train/policy_gradient_loss", float("nan"))
        vf_loss   = sb3_log.get("train/value_loss",           float("nan"))
        entropy   = sb3_log.get("train/entropy_loss",         float("nan"))
        kl        = sb3_log.get("train/approx_kl",            float("nan"))
        expl_var  = sb3_log.get("train/explained_variance",   float("nan"))
        clip_frac = sb3_log.get("train/clip_fraction",        float("nan"))
        lr        = sb3_log.get("train/learning_rate",        float("nan"))
        ep_rew    = sb3_log.get("rollout/ep_rew_mean",        self._prev_ep_rew)
        ep_len    = sb3_log.get("rollout/ep_len_mean",        self._prev_ep_len)

        em          = self._last_ep_metrics
        n_dyn_img   = em.get("n_dyn_imaged",   0)
        n_dyn_det   = em.get("n_dyn_detected", 1)
        batt_end    = self._last_batt_end
        dyn_suc_pct = n_dyn_img / max(1, n_dyn_det) * 100

        # ── ETA ───────────────────────────────────────────────────────────────
        if self.total_timesteps > 0 and fps > 0:
            remain_s = (self.total_timesteps - self._global_step) / fps
            if remain_s > 3600:
                eta_str = f"{remain_s/3600:.1f}h"
            elif remain_s > 60:
                eta_str = f"{remain_s/60:.0f}m"
            else:
                eta_str = f"{remain_s:.0f}s"
        else:
            eta_str = "..."

        # ── CSV row ───────────────────────────────────────────────────────────
        self._train_csv.write({
            "step":              self._global_step,
            "iteration":         self._iteration,
            "ep":                self._ep_num,
            "ep_len":            round(ep_len, 1) if not math.isnan(float(ep_len)) else "",
            "reward":            round(ep_rew, 4) if not math.isnan(float(ep_rew)) else "",
            "loss":              round(loss,     5) if not math.isnan(loss)     else "",
            "pg_loss":           round(pg_loss,  5) if not math.isnan(pg_loss)  else "",
            "vf_loss":           round(vf_loss,  5) if not math.isnan(vf_loss)  else "",
            "entropy":           round(entropy,  4) if not math.isnan(entropy)  else "",
            "kl":                round(kl,       6) if not math.isnan(kl)       else "",
            "explained_var":     round(expl_var, 4) if not math.isnan(expl_var) else "",
            "clip_frac":         round(clip_frac,5) if not math.isnan(clip_frac)else "",
            "lr":                lr if not math.isnan(float(lr if lr else float("nan"))) else "",
            "fps":               fps,
            "n_dyn_imaged_ep":   n_dyn_img,
            "n_dyn_detected_ep": n_dyn_det,
            "battery_end_pct":   batt_end if batt_end is not None else "",
        })

        # ── Terminal line ─────────────────────────────────────────────────────
        loss_str    = f"{loss:+7.3f}"    if not math.isnan(loss)    else "       ?"
        entropy_str = f"{entropy:+8.3f}" if not math.isnan(entropy) else "        ?"
        dyn_str     = f"{dyn_suc_pct:5.0f}%"
        stage       = f"[{self.stage_label}] " if self.stage_label else ""

        print(
            f"  {self._global_step:>10,}  "
            f"{self._ep_num:>5}  "
            f"{ep_len:>6.1f}  "
            f"{ep_rew:>+8.3f}  "
            f"{loss_str}  "
            f"{entropy_str}  "
            f"{dyn_str:>7s}  "
            f"{fps:>5}  "
            f"{stage}{eta_str}"
        )

    def _on_training_end(self) -> None:
        elapsed = time.time() - self._t0
        print(f"\n{'─'*72}")
        print(f"  Training complete.  "
              f"steps={self._global_step:,}  eps={self._ep_num}  "
              f"elapsed={elapsed/60:.1f}min")
        print(f"  Logs written to {self.log_dir}/")
        print(f"{'─'*72}\n")
        self._train_csv.close()
        self._ep_log.close()
        self._dec_log.close()
        self._orb_log.close()


def _action_histogram(actions: list[int], n_static: int = N_STATIC, n_dyn: int = N_DYN) -> dict:
    """Summarise episode actions as {static: N, dynamic: N, drift: N}."""
    h = {"static": 0, "dynamic": 0, "drift": 0}
    for a in actions:
        if a < n_static:
            h["static"] += 1
        elif a < n_static + n_dyn:
            h["dynamic"] += 1
        else:
            h["drift"] += 1
    return h


def make_loggers(
    total_timesteps: int  = 1_000_000,
    stage_label:     str  = "",
    log_dir:         str  = LOGS_DIR,
    orbit_every:     int  = ORBIT_LOG_EVERY,
    fresh_logs:      bool = False,
    start_ep:        int  = 0,
    start_step:      int  = 0,
) -> "BaseCallback":
    """
    Factory: returns the ALSATLogger callback.
    Pass the result directly to model.learn(callback=..., verbose=0).

    Parameters
    ----------
    fresh_logs : bool
        If True, deletes existing log files before starting. Use for a clean run.
        Typically pass True for the first stage only.

    Example
    -------
        from alsat_logger import make_loggers
        cb = make_loggers(total_timesteps=1_000_000, stage_label="dense",
                          fresh_logs=True)
        model.learn(total_timesteps=1_000_000, callback=cb, verbose=0)
    """
    from stable_baselines3.common.callbacks import CallbackList
    logger_cb = ALSATLogger(
        total_timesteps=total_timesteps,
        log_dir=log_dir,
        orbit_every=orbit_every,
        stage_label=stage_label,
        fresh_logs=fresh_logs,
        start_ep=start_ep,
        start_step=start_step,
    )
    return CallbackList([logger_cb])
