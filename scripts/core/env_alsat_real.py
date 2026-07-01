
#!/usr/bin/env python3
"""
env_alsat_real.py  --  Real-Data Drop-in Wrapper for ALSAT-EO-1
================================================================
A lightweight gymnasium Wrapper that enriches the existing
DynamicObsWrapper / AlsatScenario environment with:

  1. Real TLE orbit     — sets Basilisk r_CN_NInit / v_CN_NInit from SGP4
  2. Real FIRMS/GDACS   — seeds DynamicEvent pool with real geo-located hazards
  3. Real ERA5 clouds   — overrides target.cloud_cover with real daily values
  4. Real MODIS patches — feeds real 32×32 reflectance patches to the cloud CNN

The wrapper is ADDITIVE — it never modifies the underlying env class files.
All logic lives here.  To disable, simply don't apply the wrapper.

Integration (4 lines in train_ppo_smdp_full_fixed.py)
------------------------------------------------------
See INTEGRATION_INSTRUCTIONS at the bottom of this file.

Quick start
-----------
    from env_alsat_real import RealDataWrapper, RealDataConfig

    cfg = RealDataConfig(
        tle_path       = "config/orbit/alsat2a.tle",
        events_path    = "data/real_events/firms_gdacs_algeria.json",
        cloud_json     = "config/cloud_reality/era5_clouds_algeria.json",
        modis_dir      = "data/modis_patches",
    )
    env = make_env(...)                 # your existing factory call
    env = RealDataWrapper(env, cfg)     # <-- add this line

The wrapper is compatible with:
  - gymnasium.Env  (base)
  - DummyVecEnv / SubprocVecEnv  (apply before wrapping with Monitor)
  - MaskablePPO  (ActionMasker must be outermost, so apply before ActionMasker)
  - DynamicRewardShaper  (apply after RealDataWrapper, before Monitor)

Dependency on sgp4 (optional)
------------------------------
    pip install sgp4

If sgp4 is not installed, TLE injection is silently skipped (synthetic orbit
from bsk_rl is used instead — still valid, just not real TLE).

Observation space
-----------------
The wrapper does NOT change the obs shape or the action space.
It only changes the *values* inside the env (cloud fractions, event locations,
initial orbit state).  The obs tensor dimensions stay at 56.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Optional

import numpy as np
import gymnasium as gym

logger = logging.getLogger(__name__)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RealDataConfig:
    """
    Paths to real data files.  Each field can be None to disable that source.

    Parameters
    ----------
    tle_path    : Path to alsat2a.tle  (3-line format: name, L1, L2)
    events_path : Path to firms_gdacs_algeria.json
    cloud_json  : Path to era5_clouds_algeria.json
    modis_dir   : Root directory for data/modis_patches/<target>/<date>.npy
    episode_date: Date string "YYYY-MM-DD" to use for cloud lookup.
                  If None, uses today's date each episode.
    event_max   : Maximum real events to inject per episode (default 10)
    verbose     : Print injection details at each reset
    """
    tle_path:    Optional[str] = None
    events_path: Optional[str] = None
    cloud_json:  Optional[str] = None
    modis_dir:   Optional[str] = None
    episode_date: Optional[str] = None   # if None, uses today
    event_max:   int           = 10
    event_rate:  Optional[float] = None   # curriculum events/hr; None = legacy fixed event_max
    verbose:     bool          = False

    @classmethod
    def auto(cls, root: str = ROOT, verbose: bool = False) -> "RealDataConfig":
        """
        Build a config by auto-detecting which real-data files exist.
        Any missing file is silently disabled (falls back to synthetic).
        """
        tle     = os.path.join(root, "config", "orbit",        "alsat2a.tle")
        events  = os.path.join(root, "data",   "real_events",  "firms_gdacs_algeria.json")
        cloud   = os.path.join(root, "config", "cloud_reality","era5_clouds_algeria.json")
        modis   = os.path.join(root, "data",   "modis_patches")
        return cls(
            tle_path    = tle    if os.path.exists(tle)    else None,
            events_path = events if os.path.exists(events) else None,
            cloud_json  = cloud  if os.path.exists(cloud)  else None,
            modis_dir   = modis  if os.path.isdir(modis)   else None,
            verbose     = verbose,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: TLE → Basilisk state injection
# ─────────────────────────────────────────────────────────────────────────────

def _load_tle(path: str) -> tuple[str, str, str]:
    with open(path) as f:
        lines = [l.rstrip() for l in f.readlines() if l.strip()]
    if len(lines) < 3:
        raise ValueError(f"TLE file {path} must have 3 lines (name, L1, L2)")
    return lines[0], lines[1], lines[2]


def _sgp4_state(l1: str, l2: str, dt_utc: datetime) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Propagate TLE to *dt_utc* using SGP4.
    Returns (r_ECI_m, v_ECI_ms) or None if sgp4 is not available / fails.
    """
    try:
        from sgp4.api import Satrec, jday
    except ImportError:
        return None   # sgp4 not installed — skip injection silently

    try:
        sat = Satrec.twoline2rv(l1, l2)
        jd, fr = jday(dt_utc.year, dt_utc.month, dt_utc.day,
                      dt_utc.hour, dt_utc.minute,
                      dt_utc.second + dt_utc.microsecond * 1e-6)
        err, r_km, v_kms = sat.sgp4(jd, fr)
        if err != 0:
            return None
        r_m  = np.array(r_km,  dtype=float) * 1e3
        v_ms = np.array(v_kms, dtype=float) * 1e3
        return r_m, v_ms
    except Exception as exc:
        logger.debug(f"[RealDataWrapper] SGP4 propagation failed: {exc}")
        return None


def _inject_tle_into_scenario(scenario, r_m: np.ndarray, v_ms: np.ndarray) -> bool:
    """
    Set the initial orbit state on the Basilisk satellite object.
    Tries both mHub and scObject attribute paths seen in the codebase.
    Returns True if injection succeeded.
    """
    try:
        sat = scenario.satellites[0]
        hub = sat.scObject.hub

        # Path A: mHub (used in env_alsat_debug.py AlsatSatellite)
        if hasattr(hub, "mHub"):
            hub.mHub.r_CN_NInit = r_m.tolist()
            hub.mHub.v_CN_NInit = v_ms.tolist()
            return True

        # Path B: direct hub attributes (bsk_rl GeneralSatelliteTasking base)
        if hasattr(hub, "r_CN_NInit"):
            hub.r_CN_NInit = r_m.tolist()
            hub.v_CN_NInit = v_ms.tolist()
            return True

    except Exception as exc:
        logger.debug(f"[RealDataWrapper] TLE injection failed: {exc}")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: real cloud injection
# ─────────────────────────────────────────────────────────────────────────────

def _load_cloud_json(path: str) -> dict[str, dict[str, float]]:
    with open(path) as f:
        raw = json.load(f)
    # Strip metadata keys
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _get_cloud_for_target(cloud_db: dict, name: str, day_str: str) -> float | None:
    """
    Look up cloud fraction for *name* on *day_str*.
    Searches:  exact name, partial name, date ± 1 day (in case of gaps).
    Returns None if not found.
    """
    # Exact match
    if name in cloud_db:
        day_data = cloud_db[name]
        if day_str in day_data:
            return float(day_data[day_str])
        # Try nearby dates (ERA5 archive has a ~5-day lag)
        for delta in range(1, 6):
            from datetime import timedelta
            alt_date = (date.fromisoformat(day_str) - timedelta(days=delta)).isoformat()
            if alt_date in day_data:
                return float(day_data[alt_date])

    # Partial match (e.g., name="algiers_port" matches "algiers")
    for db_name, day_data in cloud_db.items():
        if db_name in name or name in db_name:
            if day_str in day_data:
                return float(day_data[day_str])

    return None


def _inject_clouds_into_scenario(scenario, cloud_db: dict, day_str: str,
                                  verbose: bool = False) -> int:
    """
    Override target.cloud_cover for all targets using real ERA5 data.
    Returns number of targets successfully updated.
    """
    n_updated = 0
    try:
        for tgt in scenario.targets:
            name = getattr(tgt, "name", None) or str(id(tgt))
            real_cloud = _get_cloud_for_target(cloud_db, name, day_str)
            if real_cloud is not None:
                tgt.cloud_cover = float(np.clip(real_cloud, 0.0, 1.0))
                n_updated += 1
                if verbose:
                    logger.debug(f"  [CLOUD] {name}: cloud_cover = {real_cloud:.3f}")
    except Exception as exc:
        logger.warning(f"[RealDataWrapper] Cloud injection failed: {exc}")
    return n_updated


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: real event injection
# ─────────────────────────────────────────────────────────────────────────────

def _load_events_json(path: str) -> list[dict]:
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        return raw.get("events", [])
    return raw


def _latlon_to_ecef(lat_deg: float, lon_deg: float,
                    alt_m: float = 0.0) -> np.ndarray:
    """WGS-84 lat/lon/alt → ECEF XYZ in metres."""
    a    = 6_378_137.0
    f    = 1 / 298.257223563
    e2   = 2 * f - f * f
    lat  = math.radians(lat_deg)
    lon  = math.radians(lon_deg)
    N    = a / math.sqrt(1 - e2 * math.sin(lat)**2)
    x    = (N + alt_m) * math.cos(lat) * math.cos(lon)
    y    = (N + alt_m) * math.cos(lat) * math.sin(lon)
    z    = (N * (1 - e2) + alt_m) * math.sin(lat)
    return np.array([x, y, z], dtype=float)


def _inject_events_into_env(env, events: list[dict], max_events: int,
                             verbose: bool = False) -> int:
    """
    Inject real FIRMS/GDACS events into the DynamicEvent pool of the env.
    Traverses the wrapper stack to find the EventManager.

    Strategy:
      - Find the EventManager on the satellite object
      - Build DynamicEvent-like objects from the JSON events
      - Append them to the manager's pending queue

    Returns number of events injected.
    """
    # Find the innermost unwrapped env and its satellite
    inner = env
    while hasattr(inner, "env"):
        inner = inner.env
    sat = None
    try:
        sat = getattr(inner, "unwrapped", inner).satellites[0]
    except Exception:
        pass

    if sat is None:
        return 0

    mgr = getattr(sat, "_event_manager", None)
    if mgr is None:
        return 0
    
    gen = None
    obj = env
    while obj is not None:
        g = getattr(obj, "_gen", None)
        if g is not None and hasattr(g, "add_scripted"):
            gen = g
            break
        obj = getattr(obj, "env", None)

    # Determine simulation duration for event timing
    try:
        from scripts.core.env_alsat_debug import SIM_DURATION_S
        sim_dur = float(SIM_DURATION_S)
    except ImportError:
        sim_dur = 172800.0   # 48h default

    # Sample up to max_events from the real event list (shuffle for variety)
    rng_seed = int(abs(hash(str(events[:3]))) % (2**31))
    rng = np.random.default_rng(rng_seed)
    selected = list(rng.choice(len(events), size=min(max_events, len(events)),
                               replace=False))

    n_injected = 0
    for idx in selected:
        ev_dict = events[int(idx)]
        try:
            lat  = float(ev_dict["lat_deg"])
            lon  = float(ev_dict["lon_deg"])
            prio = float(ev_dict.get("priority", 0.7))
            name = ev_dict.get("name", f"real_event_{idx}")
            etype = ev_dict.get("event_type", "wildfire")

            # Random appearance time within first 80% of episode
            t_appear = float(rng.uniform(sim_dur * 0.05, sim_dur * 0.85))
            t_expire = t_appear + float(rng.uniform(3600.0, 14400.0))  # 1-4 h window

            r_ecef = _latlon_to_ecef(lat, lon)

            # DynamicEvent-compatible object
            ev = _RealEventProxy(
                name            = name,
                event_type      = etype,
                lat_rad         = math.radians(lat),
                lon_rad         = math.radians(lon),
                r_LP_P          = r_ecef,
                priority        = prio,
                appearance_time = t_appear,
                expiration_time = t_expire,
                cloud_cover     = float(rng.uniform(0.0, 0.5)),   # overridden later
            )

            if gen is not None:
                gen.add_scripted([ev]); n_injected += 1   # progressive release + add_events()
            else:
                # fallback: still respect appearance_time via get_slots gate (FIX 2)
                mgr._events.append(ev); n_injected += 1

        except Exception as exc:
            logger.debug(f"[RealDataWrapper] Event injection error for {ev_dict}: {exc}")

    if verbose and n_injected > 0:
        logger.info(f"[RealDataWrapper] Injected {n_injected} real FIRMS/GDACS events")
    return n_injected


class _RealEventProxy:
    """
    Minimal DynamicEvent-compatible proxy for a real FIRMS/GDACS event.
    Implements the same attribute interface used by EventManager and
    DynamicObsWrapper.
    """
    __slots__ = ("name", "event_type", "lat_rad", "lon_rad", "r_LP_P",
                 "priority", "appearance_time", "expiration_time",
                 "cloud_cover", "cloud_cover_forecast",
                 "imaged", "_accessed")

    def __init__(self, name, event_type, lat_rad, lon_rad, r_LP_P,
                 priority, appearance_time, expiration_time, cloud_cover):
        self.name             = name
        self.event_type       = event_type
        self.lat_rad          = lat_rad
        self.lon_rad          = lon_rad
        self.r_LP_P           = r_LP_P
        self.priority         = priority
        self.appearance_time  = appearance_time
        self.expiration_time  = expiration_time
        self.cloud_cover      = cloud_cover
        self.cloud_cover_forecast = cloud_cover
        self.imaged           = False
        self._accessed        = False

    def mark_accessed(self) -> None:
        self.imaged    = True
        self._accessed = True

    def is_active(self, sim_time: float) -> bool:
        return (self.appearance_time <= sim_time < self.expiration_time
                and not self.imaged)

    def __repr__(self) -> str:
        lat = math.degrees(self.lat_rad)
        lon = math.degrees(self.lon_rad)
        return (f"<RealEvent {self.name}  {self.event_type}  "
                f"({lat:+.2f}°, {lon:+.2f}°)  prio={self.priority:.2f}>")


# ─────────────────────────────────────────────────────────────────────────────
# MODIS patch override for cloud CNN
# ─────────────────────────────────────────────────────────────────────────────

def _inject_modis_into_env(env, modis_dir: str, day_str: str,
                            verbose: bool = False) -> int:
    """
    Override the patch provider in the environment's cloud model so that
    it reads real MODIS .npy patches instead of generating synthetic ones.

    Traverses the wrapper stack to find _cloud_model with a _provider attribute,
    then monkey-patches its _get_patch method.
    """
    from data_fetchers.fetch_modis_patches import load_patch

    def _real_get_patch(target_name: str, date_str: str = day_str) -> np.ndarray:
        return load_patch(target_name, date_str, patch_dir=modis_dir)

    n_patched = 0
    obj = env
    while obj is not None:
        cm = getattr(obj, "_cloud_model", None) or getattr(obj, "cloud_model", None)
        if cm is not None:
            prov = getattr(cm, "_provider", None)
            if prov is not None:
                prov._get_patch = _real_get_patch
                n_patched += 1
                if verbose:
                    logger.info(f"[RealDataWrapper] MODIS patch provider replaced "
                                f"({prov.__class__.__name__})")
            # Also try direct _get_patch on cloud model
            if hasattr(cm, "_get_patch"):
                cm._get_patch = _real_get_patch
                n_patched += 1
        obj = getattr(obj, "env", None)

    return n_patched


# ─────────────────────────────────────────────────────────────────────────────
# The Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class RealDataWrapper(gym.Wrapper):
    """
    Drop-in wrapper that injects real TLE, FIRMS/GDACS, ERA5, and MODIS data
    into the existing AlsatScenario / DynamicObsWrapper env at each reset.

    This wrapper does NOT change obs_space, action_space, or reward structure.
    It only changes the underlying environment's state at episode boundaries.

    Parameters
    ----------
    env : gymnasium.Env
        The existing ALSAT env (after make_env() but before Monitor).
    cfg : RealDataConfig
        Paths to real data files.  Use RealDataConfig.auto() for auto-detection.
    """

    def __init__(self, env: gym.Env, cfg: RealDataConfig):
        super().__init__(env)
        self.cfg   = cfg
        self._day  = None   # cached today's date string
        self._event_rate = cfg.event_rate   # curriculum rate (events/hr); None = legacy
        self._episode    = 0

        # Pre-load data files (fail gracefully if missing)
        self._tle:        tuple | None = None
        self._cloud_db:   dict  | None = None
        self._events:     list  | None = None

        if cfg.tle_path and os.path.exists(cfg.tle_path):
            try:
                self._tle = _load_tle(cfg.tle_path)
                logger.info(f"[RealDataWrapper] TLE loaded: {cfg.tle_path}")
            except Exception as exc:
                logger.warning(f"[RealDataWrapper] TLE load failed: {exc}")

        if cfg.cloud_json and os.path.exists(cfg.cloud_json):
            try:
                self._cloud_db = _load_cloud_json(cfg.cloud_json)
                logger.info(f"[RealDataWrapper] Cloud DB loaded: "
                            f"{len(self._cloud_db)} targets, {cfg.cloud_json}")
            except Exception as exc:
                logger.warning(f"[RealDataWrapper] Cloud JSON load failed: {exc}")

        if cfg.events_path and os.path.exists(cfg.events_path):
            try:
                self._events = _load_events_json(cfg.events_path)
                logger.info(f"[RealDataWrapper] Events loaded: "
                            f"{len(self._events)} events, {cfg.events_path}")
            except Exception as exc:
                logger.warning(f"[RealDataWrapper] Events load failed: {exc}")

    # ── Public API for VecEnv compatibility ───────────────────────────────────

    def _set_inner_event_rate(self, rate: float) -> None:
        inner = self.env
        while inner is not None:
            if hasattr(inner, "set_event_rate"):
                inner.set_event_rate(rate)
                return
            inner = getattr(inner, "env", None)

    def set_event_rate(self, rate: float) -> None:
        """Remember the curriculum rate AND forward it to the inner env."""
        self._event_rate = float(rate)
        self._set_inner_event_rate(rate)

    def _curriculum_event_count(self) -> int:
        """How many real events to inject this episode, gated by curriculum rate."""
        rate = self._event_rate
        if rate is None:
            return self.cfg.event_max          # legacy behaviour
        if rate <= 0.0:
            return 0                           # static stage: NO dynamic events
        try:
            from env_alsat_debug import SIM_DURATION_S
            sim_h = float(SIM_DURATION_S) / 3600.0
        except Exception:
            sim_h = 48.0
        rng = np.random.default_rng(1234 + self._episode)
        n   = int(rng.poisson(rate * sim_h))
        return int(min(n, len(self._events), 200))  # cap by pool, not the tiny event_max

    # ── Reset hook ────────────────────────────────────────────────────────────

    def reset(self, **kwargs) -> tuple:
        obs, info = self.env.reset(**kwargs)

        # Determine episode date for cloud/MODIS lookup
        self._day = self.cfg.episode_date or date.today().isoformat()

        # ADD THIS - tell the cloud model which date we're in
        self._set_cloud_model_date(self._day)

        # Find the inner Basilisk scenario
        scenario = self._find_scenario()

        # 1. Inject TLE orbit state
        if self._tle is not None and scenario is not None:
            self._inject_tle(scenario)

        # 2. Inject ERA5 real cloud fractions
        if self._cloud_db is not None and scenario is not None:
            n = _inject_clouds_into_scenario(
                scenario, self._cloud_db, self._day, verbose=self.cfg.verbose
            )
            if self.cfg.verbose:
                logger.info(f"[RealDataWrapper] Clouds injected: {n} targets updated")

        # 3. Inject real events (FIRMS / GDACS) — CURRICULUM-GATED
        self._episode += 1
        if self._events is not None:
            n_events = self._curriculum_event_count()
            # Real events are the SOLE dynamic source: silence the synthetic
            # Poisson generator so densities don't double-count.
            self._set_inner_event_rate(0.0)
            if n_events > 0:
                _inject_events_into_env(
                    self.env, self._events,
                    max_events = n_events,
                    verbose    = self.cfg.verbose,
                )

        # 4. Inject MODIS patch provider
        if self.cfg.modis_dir and os.path.isdir(self.cfg.modis_dir):
            _inject_modis_into_env(self.env, self.cfg.modis_dir,
                                   self._day, verbose=self.cfg.verbose)

        return obs, info

    # ── Internals ─────────────────────────────────────────────────────────────

    def _find_scenario(self):
        """Walk the wrapper stack to find the Basilisk scenario object."""
        obj = self.env
        while obj is not None:
            # Try via satellites[0].scenario
            try:
                sat = getattr(obj, "unwrapped", obj).satellites[0]
                sc  = getattr(sat, "scenario", None)
                if sc is not None:
                    return sc
            except Exception:
                pass
            # Also try direct scenario attribute
            sc = getattr(obj, "scenario", getattr(getattr(obj, "unwrapped", obj),
                                                   "scenario", None))
            if sc is not None:
                return sc
            obj = getattr(obj, "env", None)
        return None

    def _inject_tle(self, scenario) -> None:
        """Propagate TLE to episode date and inject into Basilisk."""
        _name, l1, l2 = self._tle
        try:
            dt = datetime.fromisoformat(self._day).replace(tzinfo=timezone.utc)
        except ValueError:
            dt = datetime.now(timezone.utc)

        result = _sgp4_state(l1, l2, dt)
        if result is None:
            logger.debug("[RealDataWrapper] SGP4 skipped (sgp4 not installed or error)")
            return

        r_m, v_ms = result
        ok = _inject_tle_into_scenario(scenario, r_m, v_ms)
        if ok and self.cfg.verbose:
            alt_km = (np.linalg.norm(r_m) - 6_378_137.0) / 1e3
            logger.info(f"[RealDataWrapper] TLE injected: "
                        f"|r|={np.linalg.norm(r_m)/1e3:.1f} km  "
                        f"alt={alt_km:.1f} km")

    def _set_cloud_model_date(self, date_str: str) -> None:
        """Walk wrapper stack to find and update cloud model date."""
        obj = self.env
        while obj is not None:
            try:
                base = getattr(obj, "unwrapped", obj)
                for sat in getattr(base, "satellites", []):
                    sc = getattr(sat, "scenario", None)
                    for attr in ("_cloud_model", "cloud_model"):
                        cm = getattr(sc, attr, None)
                        if cm is not None and hasattr(cm, "set_episode_date"):
                            cm.set_episode_date(date_str)
                            return
            except Exception:
                pass
            obj = getattr(obj, "env", None)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def wrap_with_real_data(env: gym.Env,
                        root: str   = ROOT,
                        verbose: bool = False) -> gym.Env:
    """
    One-liner: auto-detect all real data files and apply RealDataWrapper.

    Usage:
        env = make_env(cfg, ...)
        env = wrap_with_real_data(env)    # ← single new line
        env = DynamicRewardShaper(env)
        env = Monitor(env)
    """
    cfg = RealDataConfig.auto(root=root, verbose=verbose)
    active = []
    if cfg.tle_path:    active.append("TLE")
    if cfg.events_path: active.append("events")
    if cfg.cloud_json:  active.append("ERA5-clouds")
    if cfg.modis_dir:   active.append("MODIS-patches")

    if active:
        logger.info(f"[RealDataWrapper] Active sources: {', '.join(active)}")
        return RealDataWrapper(env, cfg)
    else:
        logger.info("[RealDataWrapper] No real data files found — using synthetic sources.")
        return env


# ─────────────────────────────────────────────────────────────────────────────
# Integration instructions
# ─────────────────────────────────────────────────────────────────────────────

INTEGRATION_INSTRUCTIONS = """
=======================================================================
HOW TO INTEGRATE env_alsat_real.py INTO train_ppo_smdp_full_fixed.py
=======================================================================

Make EXACTLY 4 additions to train_ppo_smdp_full_fixed.py.
Do NOT change any other line.

─────────────────────────────────────────────────────────────────────
CHANGE 1 of 4  — Import (add after line "from reward_shaping import ...")
─────────────────────────────────────────────────────────────────────

    from env_alsat_real import wrap_with_real_data   # REAL-DATA


─────────────────────────────────────────────────────────────────────
CHANGE 2 of 4  — Inject wrapper in _make_env_with_fixes()
  Find: the line "_patch_cloud(env)   # SPEED-1: batch+cache CNN"
  Add ONE line immediately after it:
─────────────────────────────────────────────────────────────────────

    env = wrap_with_real_data(env)   # REAL-DATA: TLE+events+ERA5+MODIS


─────────────────────────────────────────────────────────────────────
CHANGE 3 of 4  — Pass --real-data flag in argparse (add to main())
  Find: ap.add_argument("--show-drift", ...)
  Add ONE line after it:
─────────────────────────────────────────────────────────────────────

    ap.add_argument("--real-data", action="store_true",     # REAL-DATA
                    help="Use real TLE/FIRMS/ERA5/MODIS data sources")


─────────────────────────────────────────────────────────────────────
CHANGE 4 of 4  — Conditionally enable wrapper (add in main())
  Find: the line "cfg = Config.DYN_REAL_VISION if use_vision else Config.DYN_MODIS"
  Add ONE line after it:
─────────────────────────────────────────────────────────────────────

    if getattr(args, 'real_data', False):               # REAL-DATA
        wrap_with_real_data.__defaults__ = (ROOT, True) # enable verbose

─────────────────────────────────────────────────────────────────────
USAGE after integration
─────────────────────────────────────────────────────────────────────

    # Without real data (unchanged behaviour):
    python -m scripts.training.train_ppo_smdp_full_fixed --episodes 500

    # With real data sources:
    python -m scripts.training.train_ppo_smdp_full_fixed \\
        --episodes 500 \\
        --real-data \\
        --cloud config/cloud_reality/era5_clouds_algeria.json

=======================================================================
"""

if __name__ == "__main__":
    print(INTEGRATION_INSTRUCTIONS)
