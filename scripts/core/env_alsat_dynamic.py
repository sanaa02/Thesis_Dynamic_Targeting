
#!/usr/bin/env python3
# ---- ALSAT path-setup -------------------------------------------
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import path_setup  # noqa
# -----------------------------------------------------------------
"""
env_alsat_dynamic.py  --  ALSAT-EO-1  Phase 3  SMDP Dynamic Environment
=========================================================================
COMPLETE REWRITE  --  Four principal upgrades vs. prior version:

[SMDP]  Native variable-duration step  (replaces SMDPDynamicWrapper)
        step() computes tau = slew_time + IMAGING_DUR_S, runs ceil(tau/BASE_STEP_S)
        Basilisk sub-steps, and accumulates reward with per-sub-step discount
        gamma_sub = gamma^(BASE_STEP_S / STEP_REF_S).
        Obs gains sojourn_time_norm as the 56th element (was 55).

[TTA]   Continuous Keplerian time-to-access
        EventManager.time_to_access() calls keplerian_tta() from wrappers/
        instead of returning the binary 0 / INACCESSIBLE_TIME_S placeholder.
        The tta_norm features in obs[43:55] now carry real predicted access
        times, giving the policy meaningful look-ahead.

[URGENCY] Deadline-pressure dynamic event reward
        urgency(t) = 1.0 + 0.5*(1 - remaining/total_lifetime)  ∈ [1.0, 1.5]
        reward = DYN_MULTIPLIER*priority*(1-cloud)*urgency - SLEW_ENERGY_ALPHA*slew_wh
        Missed events apply -0.5*priority*(1-cloud) at expiry (see DynamicObsWrapper).

[SAFE]  Safety monitor hook
        DynamicImageTargetAction calls satellite._safety_monitor.check()
        if the attribute exists, vetoing the action (->DRIFT) if it would
        violate battery/slew/storage constraints.

Observation space : Box(-inf, inf, (56,))
  [0:43]  base satellite obs (Phase 2 unchanged)
  [43:55] 3 dynamic-event slots x [priority, cloud_fcst, tta_norm, slew_norm]
  [55]    sojourn_time_norm = tau / MAX_ACTION_DUR_S        <- NEW

Action space : Discrete(24)
  0-19  static targets
  20-22 dynamic event slots 0/1/2
  23    DRIFT

smdp_dynamic.py is now DEPRECATED: make_dynamic_env() already returns
a full SMDP environment with obs (56,). Existing code that called
make_smdp_dynamic_env() should switch to make_dynamic_env().
"""
import logging
import math
from typing import List, Optional

import gymnasium as gym
import numpy as np
from bsk_rl.sim.world import WorldModel, EclipseWorldModel
from env_alsat_debug import (
    AlsatSatellite, AlsatScenario, AlsatTarget, ScienceData,
    ScienceDataStore, ScienceReward, ImageTargetAction, ModisCloudModel,
    TorqueLimitedDynamics, calculate_slew_angle_to_target,
    calculate_slew_time, calculate_slew_energy_wh, load_targets_config,
    CLOUD_THRESH, SLEW_ENERGY_ALPHA, SCHED_STEP_S, SIM_DURATION_S,
    BSK_SIM_RATE_S, CNN_NOISE_STD, SMA_M,
    BATTERY_WH,
)
from dynamic_event import (
    DynamicEvent, EventGenerator, EventManager,
    N_DYN_SLOTS, MAX_OFFNADIR_RAD, INACCESSIBLE_TIME_S, DYNAMIC_BONUS, DYN_MULTIPLIER,
)
from bsk_rl.act import Action
from bsk_rl.act.discrete_actions import DiscreteActionBuilder
from bsk_rl.gym import GeneralSatelliteTasking
from bsk_rl.data.base import DataStore
from bsk_rl.data import GlobalReward
from bsk_rl.sim import fsw

# Optional: Keplerian TTA solver from wrappers/
try:
    from env_alsat_dynamic_tta_patch import keplerian_tta as _keplerian_tta
    _HAS_KEPLERIAN = True
except ImportError:
    _keplerian_tta   = None
    _HAS_KEPLERIAN   = False

logger = logging.getLogger(__name__)


from bsk_rl.sim.world import AtmosphereWorldModel, EclipseWorldModel

class AtmosphereEclipseWorldModel(AtmosphereWorldModel, EclipseWorldModel):
    """Combined world model: exponential atmosphere + eclipse detection.

    Explicitly calls both parent setup methods to work around non-cooperative
    super() chains in BSK_RL world model base classes.
    """

    def _init_world(self, **kwargs) -> None:
        # Call both parents explicitly — don't rely on super() chain
        try:
            AtmosphereWorldModel._init_world(self, **kwargs)
        except Exception as e:
            logger.warning("[AEWorld] AtmosphereWorldModel._init_world: %s", e)
        try:
            EclipseWorldModel._init_world(self, **kwargs)
        except Exception as e:
            logger.warning("[AEWorld] EclipseWorldModel._init_world: %s", e)

    def _setup_eclipse_model(self, **kwargs) -> None:
        # If BSK_RL uses _setup_eclipse_model instead of _init_world,
        # this ensures it's called regardless of super() chain
        try:
            EclipseWorldModel._setup_eclipse_model(self, **kwargs)
        except (AttributeError, Exception) as e:
            logger.debug("[AEWorld] _setup_eclipse_model: %s", e)

# ---- constants ---------------------------------------------------------------
# N_STATIC_TARGETS = 20
# N_DYN_SLOTS = 3
N_STATIC_TARGETS  = 45
N_TOTAL_ACTIONS   = N_STATIC_TARGETS + N_DYN_SLOTS + 1     # 24
OBS_BASE_DIM      = 43
OBS_DYN_DIM       = N_DYN_SLOTS * 4                         # 12
OBS_SOJOURN_DIM   = 1
OBS_TARGET_ID_DIM = 6                                          # one ID per static slot
# +1 for static_coverage_frac (fraction of 45 targets imaged so far)
OBS_COVERAGE_DIM  = 1
OBS_TOTAL_DIM     = OBS_BASE_DIM + OBS_DYN_DIM + OBS_SOJOURN_DIM + OBS_TARGET_ID_DIM + OBS_COVERAGE_DIM  # 63

# ── Multi-objective reward balance ───────────────────────────────────────────
# STATIC_MULTIPLIER: scales per-image static reward so it competes with dynamic.
# Dynamic reward ≈ DYN_MULTIPLIER(2.0) × priority × urgency(1.5) + DYNAMIC_BONUS(1.5) ≈ 4.5
# Static reward (before fix) ≈ priority × (1-cloud) ≈ 0.5-1.0  → 4-9× weaker!
# Setting STATIC_MULTIPLIER=3.0 makes static ≈ 3×priority×(1-cloud) ≈ 1.5-3.0,
# still slightly less than dynamic (time-sensitive) but much more competitive.
STATIC_MULTIPLIER = 5.0

# Coverage completion bonuses (added once per episode at end-of-episode).
# Keys = fraction threshold, value = bonus reward.
# E.g. imaging 25% of 45 targets = 11/45 → +3 bonus.
STATIC_COVERAGE_BONUSES = {
    0.25: 3.0,   # image ≥ 11/45 targets → +3
    0.50: 6.0,   # image ≥ 23/45 targets → +6
    0.75: 10.0,  # image ≥ 34/45 targets → +10
    1.00: 15.0,  # image all 45 targets  → +15
}

# SMDP timing
BASE_STEP_S       = 30.0
STEP_REF_S        = SCHED_STEP_S        # 1200 s reference for discounting
MAX_ACTION_DUR_S  = 200.0               # normalisation cap for sojourn feature
MAX_SUB_STEPS     = 20                  # safety cap on sub-steps per action
DEFAULT_GAMMA     = 0.99

# TTA normalisation  (same scale as opportunity_open)
ORBITAL_PERIOD_S  = 5900.0             # T = 2π√(a³/μ) at 686 km altitude
TIME_NORM_S       = ORBITAL_PERIOD_S   

# [DECAY] urgency decay time-constant (1 hour)
EVENT_DECAY_TAU_S = 3600.0      # 1-hour exponential time constant

# ── Eclipse-aware battery power injection ─────────────────────────────────────
# CONVERGENCE-FIX-6: raise battery safety threshold 0.20→0.25.
# Training CSV shows ~5-10% of episodes die at battery_end_pct=0.4%
# despite the 0.20 threshold, polluting training batches with large
# negative rewards.  0.25 provides more margin without over-restricting
# imaging in the second half of the 48-h episode.
MIN_BATTERY_SAFE_SOC = 0.30

_MAX_OFFNADIR = math.radians(45.0)
_orig_check = AlsatSatellite.was_image_taken_since_last_check
def _patched_check(self, _o=_orig_check, _m=_MAX_OFFNADIR):
    if getattr(self, 'current_action_is_dynamic', False):
        slew = getattr(self, '_min_dyn_slew',
               getattr(self, 'last_slew_angle', float('inf')))
        if slew <= _m and not getattr(self, '_dyn_img_fired', False):
            self._dyn_img_fired = True
            return True
        if slew <= _m:
            return False  # already fired this action
    return _o(self)
AlsatSatellite.was_image_taken_since_last_check = _patched_check

# =============================================================================
#  Keplerian TTA wrapper (with binary fallback)
# =============================================================================

def _compute_tta(satellite, event, sim_time: float) -> float:
    if _HAS_KEPLERIAN:
        try:
            return float(_keplerian_tta(satellite, event, sim_time))
        except Exception:
            pass
    # Binary fallback
    slew = _slew_safe(satellite, event)
    return 0.0 if slew <= MAX_OFFNADIR_RAD else INACCESSIBLE_TIME_S


def _slew_safe(satellite, target) -> float:
    try:
        val = float(calculate_slew_angle_to_target(satellite, target))
        # Sanity check: a zero slew is only valid if satellite is pointing
        # almost exactly at the target. A suspiciously-zero value from an
        # uninitialized satellite should be treated as unknown → use pi/2.
        if val == 0.0:
            # verify by checking c_hat_P norm
            try:
                c_hat = np.asarray(satellite.fsw.c_hat_P, dtype=float).ravel()
                if np.linalg.norm(c_hat) < 1e-6:
                    return math.pi / 2  # uninitialized pointing → treat as inaccessible
            except Exception:
                return math.pi / 2
        return val
    except Exception:
        return math.pi / 2  # on error, assume inaccessible (large slew)


# =============================================================================
#  [SMDP] Action-duration helper
# =============================================================================

def _action_duration(satellite, action: int) -> float:
    drift = N_STATIC_TARGETS + N_DYN_SLOTS
    if action >= drift:
        return SCHED_STEP_S
    if action < N_STATIC_TARGETS:
        target = satellite.scenario.targets[action]
    else:
        slot   = action - N_STATIC_TARGETS
        mgr    = getattr(satellite, '_event_manager', None)
        if mgr is None:
            return BASE_STEP_S
        now    = float(satellite.simulator.sim_time)
        slots  = mgr.get_slots(satellite, now)
        target = slots[slot] if slot < len(slots) else None
        if target is None:
            return BASE_STEP_S
    slew = _slew_safe(satellite, target)
    tau  = calculate_slew_time(slew) + 20.0
    result = float(np.clip(tau, BASE_STEP_S, MAX_ACTION_DUR_S))
    logger.debug(f"_action_duration: action={action} slew={math.degrees(slew):.1f}° tau={tau:.0f}s -> {result:.0f}s")
    return result


# =============================================================================
#  [DYN-1] Extended action handler
# =============================================================================

class DynamicImageTargetAction(ImageTargetAction):
    """Handles actions 0-22 (static + dynamic) and 23 (DRIFT)."""

    @property
    def n_actions(self) -> int:
        if hasattr(self, 'satellite') and hasattr(self.satellite, 'scenario'):
            return len(self.satellite.scenario.targets) + N_DYN_SLOTS + 1
        return 1

    def set_action(self, action: int, prev_action_key=None) -> None:
        n_static = len(self.satellite.scenario.targets)
        now      = float(self.satellite.simulator.sim_time)

        if self.satellite.scenario is not None:
            self.satellite.scenario.update_cloud(now)

        # DRIFT — clear any locked DYN event and activate sun-pointing charge mode
        if action >= n_static + N_DYN_SLOTS:
            self.satellite.last_slew_angle         = 0.0
            self.satellite.current_action_is_dynamic = False
            self.satellite._locked_dyn_slot  = None
            self.satellite._locked_dyn_event = None
            self.satellite._dyn_img_fired    = False
            self.satellite._dyn_tasked       = False
            self.satellite._last_tasked_target = None
            try:
                self.satellite.fsw.action_charge()
            except Exception as e:
                logger.warning(f"Failed to call action_charge: {e}")
            return

        # Static target — clear DYN lock
        # In DynamicImageTargetAction.set_action(), static target branch:
        if action < n_static:
            # ── Check safety monitor for static targets ──
            monitor = getattr(self.satellite, '_safety_monitor', None)
            if monitor is not None:
                try:
                    _tgt = self.satellite.scenario.targets[action]
                    _chk = monitor.check(self.satellite, action, _tgt, now)
                    safe = _chk[0] if isinstance(_chk, tuple) else bool(_chk)
                    if not safe:
                        logger.info(f"Safety veto: static target {_tgt.name} (action={action}) is unsafe -> redirecting to DRIFT")
                        self.set_action(n_static + N_DYN_SLOTS, prev_action_key)
                        return
                except Exception as _e_safe:
                    logger.warning(f"Error checking safety for static target: {_e_safe}")

            self.satellite.current_action_is_dynamic = False
            self.satellite._locked_dyn_slot  = None
            self.satellite._locked_dyn_event = None
            self.satellite._dyn_tasked       = False
            self.satellite._last_tasked_target = None
            # ── Record last static target for monitor logging ─────────────
            try:
                _tgt = self.satellite.scenario.targets[action]   # safe: action < n_static
                self.satellite._last_static_log = {
                    "name":     getattr(_tgt, "name",        f"Target-{action:02d}"),
                    "cloud":    float(getattr(_tgt, "cloud_cover",  0.0)),
                    "priority": float(getattr(_tgt, "priority",     0.5)),
                }
            except Exception:
                pass
            # ────────────────────────────────────────────────────────────────
            super().set_action(action, prev_action_key)
            return

        # Dynamic event
        slot      = action - n_static
        logger.debug(
            f"[ACT-DYN] action={action}  slot={slot}  "
            f"locked_slot={getattr(self.satellite,'_locked_dyn_slot',None)}  "
            f"locked_event={getattr(self.satellite,'_locked_dyn_event',None)}  "
            f"t={now:.0f}s"
        )
        event_mgr = getattr(self.satellite, '_event_manager', None)
        if event_mgr is None:
            self.satellite.last_slew_angle = 0.0
            return

        # ── SMDP sub-step locking fix ────────────────────────────────────
        # PROBLEM: set_action(22) is called at EVERY SMDP sub-step (t, t+30,
        # t+60, ...).  Each call re-runs get_slots(sat, now_updated) — the
        # event ranking changes as the satellite moves, so slot 2 gets a
        # DIFFERENT event (or None) on each sub-step.  The satellite keeps
        # re-tasking before imaging completes → was_image_taken() = False →
        # mark_imaged() never called → n_dyn_imaged = 0 forever.
        #
        # FIX: lock the chosen event when first selected for this slot.
        # Reuse the locked event for all subsequent sub-steps of the SAME
        # action.  Clear the lock when action changes to drift or static.
        _LOCK_SLOT  = '_locked_dyn_slot'
        _LOCK_EVT   = '_locked_dyn_event'

        locked_slot  = getattr(self.satellite, _LOCK_SLOT,  None)
        locked_event = getattr(self.satellite, _LOCK_EVT,   None)

        if locked_slot == slot and locked_event is not None:
            # Same DYN slot — reuse the locked event (avoids re-tasking)
            event = locked_event
        else:
            # New DYN action or first sub-step — query and lock
            slots = event_mgr.get_slots(self.satellite, now)
            logger.debug(
                f"[ACT-DYN-SLOTS] mgr_id={id(event_mgr)}  sat_mgr_id={id(getattr(self.satellite,'_event_manager',None))}  "
                f"n_events={len(getattr(event_mgr,'_events',[]))}  slots={[s.name if s else None for s in slots]}"
            )
            event = slots[slot] if slot < len(slots) else None
            setattr(self.satellite, _LOCK_SLOT,  slot)
            setattr(self.satellite, _LOCK_EVT,   event)
            self.satellite._dyn_tasked = False

        if event is None:
            logger.debug(f"[ACT-DYN] slot={slot} → no event available at t={now:.0f}s")
            self.satellite._locked_dyn_event = None
            self.satellite._locked_dyn_slot = None
            self.satellite._dyn_img_fired    = False
            self.satellite._dyn_tasked       = False
            self.satellite._last_tasked_target = None
            self.satellite.last_slew_angle         = 0.0
            self.satellite.current_action_is_dynamic = False
            return

        slew = _slew_safe(self.satellite, event)
        self.satellite.last_slew_angle = float(slew)
        if slew < getattr(self.satellite, '_min_dyn_slew', float('inf')):
            self.satellite._min_dyn_slew = slew

        # [SAFE] optional safety monitor veto
        monitor = getattr(self.satellite, '_safety_monitor', None)
        if monitor is not None:
            _chk = monitor.check(self.satellite, action, event, now)
            safe, reason = _chk if isinstance(_chk, tuple) else (bool(_chk), 'safety')
            if not safe:
                logger.info(f"Safety veto: {reason}  action={action} -> redirecting to DRIFT")
                self.set_action(n_static + N_DYN_SLOTS, prev_action_key)
                return

        # Always record target regardless of slew — P4 (bsk_patches) reads
        # current_action_target after _orig returns to set _locked_dyn_event.
        self.satellite.current_action_target    = event
        self.satellite.current_action_is_dynamic = True

        # After successfully imaging a static target, add:



        # ────────────────────────────────────────────────────────────────

        # Compute off-nadir angle of target to verify orbital accessibility
        try:
            r_sat = np.asarray(self.satellite.dynamics.r_BN_P, dtype=float).ravel()
            r_tgt = np.asarray(event.r_LP_P, dtype=float).ravel()
            los = r_tgt - r_sat
            los_n = np.linalg.norm(los)
            sat_n = np.linalg.norm(r_sat)
            if los_n < 1.0 or sat_n < 1.0:
                off_nadir = 0.0
            else:
                nadir = -r_sat / sat_n
                los_unit = los / los_n
                dot = float(np.clip(np.dot(nadir, los_unit), -1.0, 1.0))
                off_nadir = float(math.acos(dot))
        except Exception:
            off_nadir = slew
            
        if off_nadir <= MAX_OFFNADIR_RAD:
            try:
                # Synthetic window so bsk_rl's task_target_for_imaging doesn't
                # crash with 'next_window' UnboundLocalError on DynamicEvents.
                try:
                    _now_s = float(self.satellite.simulator.sim_time)
                    _fake  = {"object": event,
                              "window": (_now_s - 30.0, float(event.expiration_time)),
                              "type": "target", "r_LP_P": getattr(event, 'r_LP_P', None)}
                    if not hasattr(self.satellite, "opportunities"):
                        self.satellite.opportunities = []
                    self.satellite.opportunities = [o for o in self.satellite.opportunities if o.get("object") is not event]
                    import bisect
                    bisect.insort(
                        self.satellite.opportunities,
                        _fake,
                        key=lambda x: x["window"][1]
                    )
                except Exception:
                    pass
                if not getattr(self.satellite, "_dyn_tasked", False):
                    self.satellite.task_target_for_imaging(event)
                logger.debug(
                    f"[ACT-DYN] tasked event={event.name}  "
                    f"slew_deg={math.degrees(slew):.1f}  "
                    f"cloud_fcst={event.cloud_cover_forecast:.2f}"
                )
            except Exception as exc:
                logger.debug(f"task_target_for_imaging (dynamic): {exc}")


# =============================================================================
#  [DYN-2] Extended reward with [DECAY]
# =============================================================================

class DynamicScienceDataStore(ScienceDataStore):
    data_type = ScienceData

    def compare_log_states(self, old_state, new_state) -> ScienceData:
        sat = self.satellite

        # ── DYN event imaging bypass ──────────────────────────────────────
        # bsk_rl's was_image_taken_since_last_check() only returns True for
        # targets with precomputed access windows (upcoming_opportunities).
        # DynamicEvents have NO precomputed windows → always returns False.
        # Fix: directly confirm imaging when satellite correctly pointed at
        # the DYN event (slew <= MAX_OFFNADIR_RAD) and imaging not yet fired.
        _locked = getattr(sat, '_locked_dyn_event', None)
        if _locked is not None and getattr(_locked, 'imaged', False):
            sat.current_action_target = None
            sat.current_action_is_dynamic = False
            sat._locked_dyn_event = None
            sat._locked_dyn_slot = None
            sat.was_image_taken_since_last_check()  # drain buffer
            return ScienceData(0.0)

        is_dyn_action = getattr(sat, 'current_action_is_dynamic', False)
        slew_angle    = getattr(sat, 'last_slew_angle', float('inf'))
        already_fired = getattr(sat, '_dyn_img_fired', False)


        
        if is_dyn_action:
            sat.was_image_taken_since_last_check()  # drain bsk_rl image buffer
            return ScienceData(0.0)                 # reward injected by wrapper

        # Static target: use bsk_rl's standard imaging check
        # Static target: use bsk_rl's standard imaging check
        image_taken = sat.was_image_taken_since_last_check()
        if not image_taken:
            logger.debug(f"[STATIC] was_image_taken=False  target={getattr(sat,'current_action_target',None)}")
            return ScienceData(0.0)

        target = getattr(sat, 'current_action_target', None)
        if target is None:
            return ScienceData(0.0)

        is_dynamic  = getattr(sat, 'current_action_is_dynamic', False)
        cloud_truth = float(target.cloud_cover)
        priority    = float(target.priority)
        slew_angle  = getattr(sat, 'last_slew_angle', 0.0)
        _slew_mult  = getattr(sat, '_slew_energy_multiplier', 1.0)
        slew_energy = calculate_slew_energy_wh(slew_angle, _slew_mult)

        if is_dynamic:
            # [DECAY] urgency factor
            try:
                now       = float(sat.simulator.sim_time)
                elapsed   = now - float(target.appearance_time)
                remaining = max(0.0, float(target.expiration_time) - now)
                total_dur  = max(1.0, float(target.expiration_time) - float(target.appearance_time))
                frac_remaining = min(1.0, max(0.0, remaining / total_dur))  # 1 fresh → 0 expiry
                urgency = 1.0 + 0.5 * frac_remaining  # linearly decays from 1.5 to 1.0 as event approaches expiry
            except Exception:
                urgency = 1.0

            if cloud_truth < CLOUD_THRESH:
                reward = (DYN_MULTIPLIER * priority * (1.0 - cloud_truth) * urgency
                         - SLEW_ENERGY_ALPHA * slew_energy + DYNAMIC_BONUS)
                sat._metrics['n_cloud_free'] += 1
            else:
                # FIX-10: unified cloud penalty — same scale as static (-0.1 * priority)
                # (was hard-coded -0.3 regardless of priority; now consistent)
                reward = -0.1 * priority
                sat._metrics['n_cloudy'] += 1


            event_mgr = getattr(sat, '_event_manager', None)
            if event_mgr is not None and isinstance(target, DynamicEvent):
                event_mgr.mark_imaged(target, float(sat.simulator.sim_time), reward)
                sat._metrics['n_dyn_imaged'] = event_mgr._metrics['n_imaged']

        else:
            # FIX-3: suppress duplicate reward for already-imaged static targets
            _imaged_set = getattr(sat, '_imaged_static_set', None)
            if _imaged_set is None:
                sat._imaged_static_set = set()
                _imaged_set = sat._imaged_static_set
            _tgt_id = getattr(target, 'name', id(target))
            if _tgt_id in _imaged_set:
                logger.debug(
                    f"[STATIC-DUP] target={_tgt_id} already imaged this episode — reward=0"
                )
                sat.current_action_target     = None
                sat.current_action_is_dynamic = False
                return ScienceData(0.0)
            _imaged_set.add(_tgt_id)

            if cloud_truth < CLOUD_THRESH:
                # STATIC_MULTIPLIER makes static imaging competitive with dynamic events.
                # Without this, dynamic (≈4.5 reward) dominates static (≈0.5-1.0).
                _base_r = STATIC_MULTIPLIER * priority * (1.0 - cloud_truth)
                _cost   = SLEW_ENERGY_ALPHA * slew_energy
                # Wasted slew energy is fully penalized without capping
                reward  = _base_r - _cost
                sat._metrics['n_cloud_free'] += 1
                sat._metrics['n_static_imaged_clean'] = sat._metrics.get('n_static_imaged_clean', 0) + 1
                # Track cumulative coverage for completion bonuses
                _n_imaged_static = sat._metrics.get('n_static_imaged_clean', 0)
                _coverage_frac = _n_imaged_static / float(N_STATIC_TARGETS)
                sat._metrics['static_coverage_frac'] = _coverage_frac
            else:
                reward = -0.1 * priority
                sat._metrics['n_cloudy'] += 1

        logger.debug(
            f"[STATIC] image taken: target={target.name}  "
            f"cloud_truth={cloud_truth:.2f}  priority={priority:.2f}  "
            f"slew_deg={math.degrees(slew_angle):.1f}  reward={reward:+.4f}"
        )
        # ── FIX-08: Zero-reward static action diagnostic ──────────────────
        if reward <= 0.0 and not is_dynamic:
            _r08 = (
                'CLOUDY'        if cloud_truth >= CLOUD_THRESH
                else 'SLEW_LIMIT' if math.degrees(slew_angle) > 45.0
                else 'NEGATIVE_SLEW_COST'
            )
            logger.debug(
                f'[STATIC-ZERO] target={target.name}  reason={_r08}  '
                f'cloud={cloud_truth:.2f}  slew={math.degrees(slew_angle):.1f}deg  '
                f'reward={reward:+.4f}'
            )
        # ──────────────────────────────────────────────────────────────────
     
        sat._metrics['n_imaged']             += 1
        sat._metrics['total_slew_angle_deg'] += math.degrees(slew_angle)
        sat._metrics['total_slew_energy_wh'] += slew_energy
        sat._metrics['total_reward']         += reward



        sat.current_action_target    = None
        sat.current_action_is_dynamic = False
        return ScienceData(reward)


class DynamicScienceReward(GlobalReward):
    data_store_type = DynamicScienceDataStore

    def __init__(self, reward_scale: float = 1.0):
        super().__init__()
        self.reward_scale = reward_scale

    def calculate_reward(self, new_data_dict: dict) -> dict:
        return {k: v.value * self.reward_scale for k, v in new_data_dict.items()}


# =============================================================================
#  Satellite with event manager + extended metrics
# =============================================================================

class DynamicAlsatSatellite(AlsatSatellite):
    action_spec = [DynamicImageTargetAction()]

    def __init__(self, name='ALSAT-1', sat_args=None, scenario=None,
                 event_manager: Optional[EventManager] = None,
                 safety_monitor=None, **kwargs):
        self._event_manager  = event_manager
        self._safety_monitor = safety_monitor
        super().__init__(name=name, sat_args=sat_args, scenario=scenario, **kwargs)

    def reset_post_sim_init(self) -> None:
        self._dyn_img_fired    = False
        self._locked_dyn_event = None
        self._locked_dyn_slot  = None
        self._dyn_reward_given = False

        super().reset_post_sim_init()
        # print(f"Initial battery SOC: {self.dynamics.battery_charge_fraction:.2f}")
        self.current_action_is_dynamic = False
        self._metrics.update({
            'n_dyn_detected': 0, 'n_dyn_imaged': 0,
            'n_static_imaged_clean': 0, 'n_dyn_imaged_clean': 0,
            'static_coverage_frac': 0.0,
            'static_coverage_bonus_total': 0.0,
            'n_static_available_windows': 0,   # how many unique static windows were seen
            'n_missed_events': 0,               # track missed dynamic events
        })
        # FIX-3: track which static targets were already imaged this episode
        self._imaged_static_set: set = set()
        # Track coverage bonus milestones already awarded this episode
        self._coverage_bonus_awarded: set = set()
        if self._event_manager is not None:
            self._event_manager.reset()






# =============================================================================
#  Flat single-satellite wrapper (for SB3 compatibility)
# =============================================================================

class SingleSatelliteEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space      = self.env.action_space[0]
        self.observation_space = self.env.observation_space[0]

    def reset(self, **kwargs):
        obs_tuple, info = self.env.reset(**kwargs)
        return obs_tuple[0], info

    def step(self, action):
        obs_tuple, r, term, trunc, info = self.env.step((action,))
        try:
            sat = self.env.unwrapped.satellites[0]
            info['episode_metrics'] = dict(sat._metrics)
        except Exception:
            pass
        return obs_tuple[0], r, term, trunc, info


# =============================================================================
#  [SMDP + TTA + DECAY] DynamicObsWrapper  --  the core Phase 3 env
# =============================================================================

def _clean_array(val, prev_val=None, default=None):
    if val is None:
        return np.array(default) if prev_val is None else prev_val
    try:
        arr = np.asarray(val, dtype=float).ravel()
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            if prev_val is not None:
                return prev_val
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr
    except Exception:
        if prev_val is not None:
            return prev_val
        return np.array(default if default is not None else [0.0, 0.0, 0.0])

def _clean_scalar(val, prev_val=None, default=0.0):
    try:
        fval = float(val)
        if math.isnan(fval) or math.isinf(fval):
            return default if prev_val is None else prev_val
        return fval
    except Exception:
        return default if prev_val is None else prev_val

class DynamicObsWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that delivers the full Phase 3 feature set:

      Obs (56,): base(43) + dyn_events(12) + sojourn(1)
      SMDP step: variable duration tau = slew + imaging,
                 discount gamma_sub = gamma^(BASE_STEP_S/STEP_REF_S)
      TTA features: Keplerian-predicted (continuous, not binary)
      Safety: EventManager slots ranked by accessibility

    Parameters
    ----------
    env    : SingleSatelliteEnv (wrapping bsk_rl GeneralSatelliteTasking)
    gen    : EventGenerator (Poisson arrivals)
    mgr    : EventManager   (shared with satellite)
    gamma  : discount factor per STEP_REF_S  (default 0.99)
    """

    def __init__(self, env: gym.Env, gen: EventGenerator, mgr: EventManager,
                 gamma: float = DEFAULT_GAMMA, seed: int = 42):
        super().__init__(env)
        self._gen       = gen
        self._mgr       = mgr
        self._gamma_sub = gamma ** (BASE_STEP_S / STEP_REF_S)
        self._prev_time = 0.0
        self._batt_time = 0.0
        self.seed       = seed

        self.action_space = gym.spaces.Discrete(N_TOTAL_ACTIONS)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_TOTAL_DIM,), dtype=np.float32)

        # ── 1-second simulation logger state ──
        self._sim_log_file = None
        self._log_t_prev = None
        self._log_r_prev = None
        self._log_v_prev = None
        self._log_sig_prev = None
        self._log_c_prev = None
        self._log_batt_prev = None

    # ---- reset / step -------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # ── 1-second simulation logger reset ──
        if hasattr(self, "_sim_log_file") and self._sim_log_file is not None:
            try:
                self._sim_log_file.close()
            except Exception:
                pass
            self._sim_log_file = None

        if getattr(self, "seed", 42) == 42:
            try:
                log_dir = "/home/sanaa/Videos/Thesis_Dynamic_Targeting_copy2/logs"
                _os.makedirs(log_dir, exist_ok=True)
                self._sim_log_path = _os.path.join(log_dir, "satellite_simulation.csv")
                self._sim_log_file = open(self._sim_log_path, "w")
                self._sim_log_file.write("time_s,x_m,y_m,z_m,vx_ms,vy_ms,vz_ms,sigma0,sigma1,sigma2,cx,cy,cz,battery_pct,action_name,action_idx\n")
                self._sim_log_file.flush()
            except Exception as e:
                logger.warning(f"Failed to open simulation log: {e}")

        try:
            sat = self.env.unwrapped.satellites[0]
            self._log_t_prev = float(sat.simulator.sim_time)
            self._log_r_prev = _clean_array(sat.dynamics.r_BN_N, default=[0.0, 0.0, 0.0])
            self._log_v_prev = _clean_array(sat.dynamics.v_BN_N, default=[0.0, 0.0, 0.0])
            self._log_sig_prev = _clean_array(sat.dynamics.sigma_BN, default=[0.0, 0.0, 0.0])
            
            c_raw = getattr(sat.fsw, "c_hat_P", None)
            self._log_c_prev = _clean_array(c_raw, default=[0.0, 0.0, 0.0])
            
            self._log_batt_prev = _clean_scalar(sat.dynamics.battery_charge_fraction, default=1.0) * 100.0

            if self._sim_log_file is not None:
                self._sim_log_file.write(
                    f"{self._log_t_prev:.1f},{self._log_r_prev[0]:.3f},{self._log_r_prev[1]:.3f},{self._log_r_prev[2]:.3f},"
                    f"{self._log_v_prev[0]:.3f},{self._log_v_prev[1]:.3f},{self._log_v_prev[2]:.3f},"
                    f"{self._log_sig_prev[0]:.6f},{self._log_sig_prev[1]:.6f},{self._log_sig_prev[2]:.6f},"
                    f"{self._log_c_prev[0]:.6f},{self._log_c_prev[1]:.6f},{self._log_c_prev[2]:.6f},"
                    f"{self._log_batt_prev:.2f},RESET,48\n"
                )
                self._sim_log_file.flush()
        except Exception as e:
            logger.debug(f"[SimulationLogger] Error during reset log: {e}")
            self._log_t_prev = None

        # ═══════════════════════════════════════════════════════════════════════════
        # SNIPPET 2: BATTERY INIT (paste in env_alsat_dynamic.py reset() method)
        # ═══════════════════════════════════════════════════════════════════════════

        # ╔═══════════════════════════════════════════════════════════════════════╗
        # BATTERY INIT DEBUG - SNIPPET 2
        # ╚═══════════════════════════════════════════════════════════════════════╝
        try:
            _sat = self.env.unwrapped.satellites[0]
            _batt = _sat.dynamics.battery_charge_fraction
            
            # Try to get solar power if available
            _solar = getattr(_sat.dynamics, 'solar_power', None)
            _solar_str = f"{_solar:.2f}W" if _solar is not None else "N/A"
            
            # logger.info(
            #     f"\n{'='*70}\n"
            #     f"  EPISODE RESET\n"
            #     f"  Initial Battery SOC:     {_batt:.3f}\n"
            #     f"  Solar Power:              {_solar_str}\n"
            #     f"  Safety Threshold:         0.20\n"
            #     f"  Critical Threshold:       0.15\n"
            #     f"{'='*70}\n"
            # )
        except Exception as e:
            logger.warning(f"[BATT-INIT] Could not log: {e}")
        


        self._prev_time = 0.0
        self._batt_time = 0.0
        self._prev_batt_time = 0.0
        self._gen.reset(seed=kwargs.get('seed'))
        self._mgr.reset()

        # print(f"[EVT-DBG] reset: gen.rate_hz={self._gen.rate_hz:.6f} "
        #     f"gen._countdown={self._gen._countdown} "
        #     f"mgr._events={len(self._mgr._events)}", flush=True)
        # ── CRITICAL: attach self._mgr to satellite so set_action() sees events
        try:
            for _sat in self.env.unwrapped.satellites:
                _sat._event_manager = self._mgr
        except Exception:
            pass
        self._n_static_actions_ep = 0
        self._penalized_static_windows = set()  # FIX-MISS-STATIC: de-dup per window

        # ── FIX-06: Access window coverage check ──────────────────────────
        try:
            _sat_wr = self.env.unwrapped.satellites[0]
            _opps   = getattr(_sat_wr, 'upcoming_opportunities', [])
            _tgts   = list(_sat_wr.scenario.targets)
            _dur    = float(getattr(
                self.env.unwrapped, 'time_limit',
                getattr(self.env, 'time_limit', SIM_DURATION_S)
            ))
            # print(f'[WINDOWS] {len(_tgts)} targets  episode={_dur:.0f}s')
            _any_no_win = False
            for _tgt in _tgts:
                _wins = []
                for _opp in _opps:
                    try:
                        _o = (_opp.get('object') if isinstance(_opp, dict)
                              else getattr(_opp, 'object', None))
                        _t = (_opp.get('type', '') if isinstance(_opp, dict)
                              else getattr(_opp, 'type', ''))
                        _w = (_opp.get('window', [0,1]) if isinstance(_opp, dict)
                              else getattr(_opp, 'window', [0, 1]))
                        if _o is _tgt and _t == 'target':
                            _wins.append(_w)
                    except Exception:
                        pass
                _name = getattr(_tgt, 'name', str(_tgt))
                if _wins:
                    _first = min(w[0] for w in _wins)
                    _last  = max(w[1] for w in _wins)
                    # print(f'  {_name:20s}: first={_first:7.0f}s  last={_last:7.0f}s  count={len(_wins)}')
                else:
                    _any_no_win = True
                    # print(f'  {_name:20s}: *** NO WINDOWS ***')
            # if _any_no_win:
                # print('[WINDOWS] WARNING: some targets have no windows — '
                #       'check initial_generation_duration in make_dynamic_env()')
        except Exception as _wc_exc:
            logger.debug(f'[WINDOWS] coverage check failed: {_wc_exc}')
        # ──────────────────────────────────────────────────────────────────


        return self._build_obs(obs, tau_norm=0.0), info

    def _log_state_1s(self, action: int):
        if getattr(self, "seed", 42) != 42 or self._sim_log_file is None:
            return

        try:
            sat = self.env.unwrapped.satellites[0]
            t_curr = float(sat.simulator.sim_time)
            t_prev = self._log_t_prev

            if t_prev is None or t_curr <= t_prev:
                self._log_t_prev = t_curr
                self._log_r_prev = _clean_array(sat.dynamics.r_BN_N, default=[0.0, 0.0, 0.0])
                self._log_v_prev = _clean_array(sat.dynamics.v_BN_N, default=[0.0, 0.0, 0.0])
                self._log_sig_prev = _clean_array(sat.dynamics.sigma_BN, default=[0.0, 0.0, 0.0])
                
                c_raw = getattr(sat.fsw, "c_hat_P", None)
                self._log_c_prev = _clean_array(c_raw, default=[0.0, 0.0, 0.0])
                
                self._log_batt_prev = _clean_scalar(sat.dynamics.battery_charge_fraction, default=1.0) * 100.0
                return

            r_curr = _clean_array(sat.dynamics.r_BN_N, prev_val=self._log_r_prev, default=[0.0, 0.0, 0.0])
            v_curr = _clean_array(sat.dynamics.v_BN_N, prev_val=self._log_v_prev, default=[0.0, 0.0, 0.0])
            sig_curr = _clean_array(sat.dynamics.sigma_BN, prev_val=self._log_sig_prev, default=[0.0, 0.0, 0.0])
            
            c_raw = getattr(sat.fsw, "c_hat_P", None)
            c_curr = _clean_array(c_raw, prev_val=self._log_c_prev, default=[0.0, 0.0, 0.0])
                
            batt_curr = _clean_scalar(sat.dynamics.battery_charge_fraction, prev_val=self._log_batt_prev/100.0, default=1.0) * 100.0

            r_prev = self._log_r_prev
            v_prev = self._log_v_prev
            sig_prev = self._log_sig_prev
            c_prev = self._log_c_prev
            batt_prev = self._log_batt_prev

            n_static = N_STATIC_TARGETS
            if action < n_static:
                try:
                    aname = sat.scenario.targets[action].name
                except Exception:
                    aname = f"STATIC_{action}"
            elif action < n_static + N_DYN_SLOTS:
                aname = f"DYNAMIC_{action - n_static}"
            else:
                aname = "DRIFT"

            start_s = int(math.ceil(t_prev))
            end_s = int(math.floor(t_curr))

            lines = []
            for t in range(start_s, end_s + 1):
                frac = (t - t_prev) / (t_curr - t_prev)
                frac = max(0.0, min(1.0, frac))

                r_t = r_prev + frac * (r_curr - r_prev)
                v_t = v_prev + frac * (v_curr - v_prev)
                sig_t = sig_prev + frac * (sig_curr - sig_prev)
                
                c_t = c_prev + frac * (c_curr - c_prev)
                c_norm = np.linalg.norm(c_t)
                if c_norm > 1e-8:
                    c_t = c_t / c_norm
                
                batt_t = batt_prev + frac * (batt_curr - batt_prev)

                lines.append(
                    f"{t:.1f},{r_t[0]:.3f},{r_t[1]:.3f},{r_t[2]:.3f},"
                    f"{v_t[0]:.3f},{v_t[1]:.3f},{v_t[2]:.3f},"
                    f"{sig_t[0]:.6f},{sig_t[1]:.6f},{sig_t[2]:.6f},"
                    f"{c_t[0]:.6f},{c_t[1]:.6f},{c_t[2]:.6f},"
                    f"{batt_t:.2f},{aname},{action}\n"
                )

            if lines:
                self._sim_log_file.writelines(lines)
                self._sim_log_file.flush()

            self._log_t_prev = t_curr
            self._log_r_prev = r_curr
            self._log_v_prev = v_curr
            self._log_sig_prev = sig_curr
            self._log_c_prev = c_curr
            self._log_batt_prev = batt_curr

        except Exception as e:
            logger.debug(f"[SimulationLogger] Error writing log: {e}")

    def set_event_rate(self, rate: float) -> None:
        try:
            self._gen.rate_hz = float(rate) / 3600.0   # fix: use rate_hz, not rate
            logger.debug(f"[DynamicObsWrapper] event_rate set: rate_hz={self._gen.rate_hz:.6f} ({rate:.2f}/hr)")
        except Exception as exc:
            logger.debug(f"[DynamicObsWrapper] set_event_rate failed: {exc}")
    def step(self, action: int):
        # Dynamically adjust Basilisk task periods based on the action:
        # 0.5s for dynamic targeting actions to settle pointing; 10.0s otherwise for speed.
        try:
            _sat = self.env.unwrapped.satellites[0]
            _sim = _sat.simulator
            _N_STATIC = N_STATIC_TARGETS
            _is_imaging_action = int(action) < (_N_STATIC + N_DYN_SLOTS)
            for _proc in _sim.procList:
                _proc_name = _proc.processData.processName
                for _entry in _proc.processData.processTasks:
                    _task_name = _entry.TaskPtr.TaskName
                    if "FSW" in _proc_name:
                        _fsw_rate_s = 0.5 if _is_imaging_action else 10.0
                        _proc.processData.changeTaskPeriod(_task_name, int(_fsw_rate_s * 1e9))
                    elif "Dynamics" in _proc_name:
                        # Dynamics process integrates physical equations. Using 1.0s instead of 10.0s during drift
                        # ensures numerical integration stability (no NaN blow-up due to high wheel speed).
                        _dyn_rate_s = 0.5 if _is_imaging_action else 1.0
                        _proc.processData.changeTaskPeriod(_task_name, int(_dyn_rate_s * 1e9))
        except Exception as _e_rate:
            logger.debug(f"[RateAdjustment] Error adjusting task periods: {_e_rate}")

        # ──────────────────────────────────────────────────────────────────────────────
        # [LOG1] WHAT ACTION ARE WE EXECUTING AND WHAT'S THE BATTERY STATE?
        # ──────────────────────────────────────────────────────────────────────────────
        try:
            _sat = self.env.unwrapped.satellites[0]
            _soc = float(_sat.dynamics.battery_charge_fraction)
            _t = float(_sat.simulator.sim_time)
            _n_static = N_STATIC_TARGETS
            if action < _n_static:
                _aname = f"IMG[{action}]"
            elif action < _n_static + N_DYN_SLOTS:
                _aname = f"DYN[{action - _n_static}]"
            else:
                _aname = "DRIFT"
            # print(f"[LOG1] t={_t:>8.0f}s | action={action:>2d}({_aname:>8s}) | SOC_start={_soc:.4f}")
        except Exception as e:
            print(f"[LOG1] ERROR: {e}")
 
        # ─────────────────────────────────────────────────────────────────────────────

        


        _N_STATIC = N_STATIC_TARGETS
        _is_dyn_action = _N_STATIC <= int(action) < _N_STATIC + N_DYN_SLOTS
        if _is_dyn_action:
            try:
                _sat_pre = self.env.unwrapped.satellites[0]
                _sat_pre.was_image_taken_since_last_check()  # drain before ANY sub-step
                _sat_pre._dyn_img_fired = False
                _sat_pre._dyn_reward_given = False
            except Exception:
                pass


        # # ── Low-battery action masking (FIX-BATT-MASK) ───────────────────
        # # Only mask true imaging actions (0..N_STATIC+N_DYN-1 excluding DRIFT).
        try:
            _bm_sat = self.env.unwrapped.satellites[0]
            _DRIFT_ACT_LOCAL = len(_bm_sat.scenario.targets) + N_DYN_SLOTS
        except Exception:
            _DRIFT_ACT_LOCAL = N_STATIC_TARGETS + N_DYN_SLOTS
            _bm_sat = None

        if int(action) < _DRIFT_ACT_LOCAL and _bm_sat is not None:
            try:
                # DIRECT access to dynamics battery (most reliable)
                _bm_soc = float(_bm_sat.dynamics.battery_charge_fraction)
                # print(f"[BATT-DEBUG] action={action} SOC={_bm_soc:.3f} threshold={MIN_BATTERY_SAFE_SOC}")
                if _bm_soc < MIN_BATTERY_SAFE_SOC:
                    action = _DRIFT_ACT_LOCAL
                    # print(f"[BATT-MASK] → forced DRIFT")
            except Exception as e:
                print(f"[BATT-MASK] ERROR: {e}")
                pass


        # # ── LOG ACTUAL ACTION TAKEN (after masking) ──────────────────────────
        # try:
        #     _actual_action_log = self.env.unwrapped.satellites[0]
        #     _actual_sim_t = float(_actual_action_log.simulator.sim_time)
        #     _actual_soc = float(_actual_action_log.dynamics.battery_charge_fraction)
            
        #     # Decode action name
        #     _n_static_actual = N_STATIC_TARGETS
        #     if action < _n_static_actual:
        #         _actual_name = f"IMG[{action}]"
        #     elif action < _n_static_actual + N_DYN_SLOTS:
        #         _actual_name = f"DYN[{action - _n_static_actual}]"
        #     else:
        #         _actual_name = "DRIFT"
            
        #     with open('/home/sanaa/Documents/Thesis_Dynamic_Targeting/logs/actions_taken.log', 'a') as f:
        #         f.write(f"[ACTION-TAKEN] t={_actual_sim_t:>8.0f}s | action={action:>2d} ({_actual_name:>8s}) | SOC={_actual_soc:.4f}\n")
        # except Exception as e:
        #     pass

      


        if int(action) < N_STATIC_TARGETS:
           self._n_static_actions_ep = getattr(self, '_n_static_actions_ep', 0) + 1
        # [SMDP] compute actual task duration
        try:
            sat      = self.env.unwrapped.satellites[0]
            # Reset DYN imaging flag for new action
            sat._dyn_reward_given = False
            # FIX-B2: Clear stale current_action_target before non-imaging actions.
            # Root cause: BSK-RL early-terminates a scheduler step when the Basilisk
            # image-confirmed flag fires. If the imaging action's sub-steps didn't
            # see the flag (it fired after those sub-steps), current_action_target
            # remains set when the next DRIFT/DYN action's BSK-RL scheduler step
            # runs — causing compare_log_states() to award full imaging reward to
            # a drift step (visible in decisions.log as non-1200s-gap drift steps
            # with positive rewards). Clearing the pointer here prevents this bleed.
            if int(action) >= N_STATIC_TARGETS:
                sat.current_action_target = None
            # [FIX-A] For DYN actions: drain the image buffer BEFORE sub-steps
            # so that any image taken for a prior static target doesn't bleed
            # a negative slew-energy penalty into total_r during DYN sub-steps.
            if _N_STATIC <= int(action) < _N_STATIC + N_DYN_SLOTS:
                try:
                    sat.was_image_taken_since_last_check()  # drain buffer, discard result
                except Exception:
                    pass
            tau      = _action_duration(sat, int(action))
        except Exception:
            tau = BASE_STEP_S

        _drift_val = N_STATIC_TARGETS + N_DYN_SLOTS
        if int(action) >= _drift_val:
            tau   = float(np.clip(tau, BASE_STEP_S, SCHED_STEP_S))
            n_sub = max(1, min(40, int(math.ceil(tau / BASE_STEP_S))))
        else:
            tau   = float(np.clip(tau, BASE_STEP_S, MAX_ACTION_DUR_S))
            n_sub = max(1, min(MAX_SUB_STEPS, int(math.ceil(tau / BASE_STEP_S))))

        last_obs = None
        term = trunc = False
        info: dict = {}

        # ── Pre-step event spawn (ROOT FIX for n_dyn_imaged=0) ───────────────
        try:

            # Keep sat._event_manager pointing at self._mgr (survives bsk_rl resets)
            for _sx in self.env.unwrapped.satellites:
                if getattr(_sx, '_event_manager', None) is not self._mgr:
                   _sx._event_manager = self._mgr
        except Exception:
            pass

        # Get base env and its simulator to temporarily set max_step_duration to BASE_STEP_S
        base_env = self.env
        while hasattr(base_env, "env"):
            base_env = base_env.env
        
        orig_max_step_dur = getattr(base_env, "max_step_duration", None)
        orig_sim_max_step_dur = getattr(getattr(base_env, "simulator", None), "max_step_duration", None)
        
        if orig_max_step_dur is not None:
            base_env.max_step_duration = BASE_STEP_S
        if orig_sim_max_step_dur is not None:
            base_env.simulator.max_step_duration = BASE_STEP_S

        DRIFT_ACT = N_STATIC_TARGETS + N_DYN_SLOTS  # = 23
        total_r = 0.0
        _is_imaging_action = int(action) < N_STATIC_TARGETS  # static target attempt
        _soc_before_substep = _soc
        try:
            for _i in range(n_sub):
                _sub_a = action
                try:
                    for _sx in self.env.unwrapped.satellites:
                        _sx._event_manager = self._mgr
                except Exception:
                    pass

                # # ── Dynamic power sink: controls battery via Basilisk's own power model ──
                # # nodePowerOut > 0 = power source (charges)  |  < 0 = extra load (drains)
                # # basePowerSink is confirmed writable from probe.
                # try:
                #     _dyn_ps = self.env.unwrapped.satellites[0].dynamics
                #     if _is_imaging_action:
                #         _dyn_ps.basePowerSink.nodePowerOut = -100.0   # +RW drain = -112W net
                #     else:
                #         _dyn_ps.basePowerSink.nodePowerOut = 20.0   # -RW drain = +68W net
                # except Exception:
                #     pass
                # ──────────────────────────────────────────────────

                # # ──────────────────────────────────────────────────────────────────────────────
                # # [LOG2] WHAT POWER MODE IS BASILISK NOW IN?
                # # ──────────────────────────────────────────────────────────────────────────────
                # try:
                #     _ps = self.env.unwrapped.satellites[0].dynamics
                #     _power_out = getattr(_ps, 'basePowerSink', None)
                #     if _power_out is not None:
                #         _pval = float(_power_out.nodePowerOut)
                #         _mode = "DRAIN" if _pval < 0 else "CHARGE"
                #         print(f"[LOG2] basePowerSink.nodePowerOut={_pval:>8.1f}W ({_mode})")
                # except Exception as e:
                #     print(f"[LOG2] ERROR: {e}")
                # 
                # # ─────────────────────────────────────────────────────────────────────────────

                # # ──────────────────────────────────────────────────────────────────────────────
                # # [LOG3] ABOUT TO CALL env.step() - LOG WHICH SUB-ACTION AND ITERATION
                # # ──────────────────────────────────────────────────────────────────────────────
                # try:
                #     _i_sub = _i + 1  # iteration counter (starts at 0)
                #     print(f"[LOG3] sub-step={_i_sub:>2d}/{n_sub} | action={_sub_a} | about to call env.step()")
                # except Exception as e:
                #     print(f"[LOG3] ERROR: {e}")
                # 
                # # ─────────────────────────────────────────────────────────────────────────────

                obs_i, r_i, term, trunc, info = self.env.step(_sub_a)
                total_r += (self._gamma_sub ** _i) * r_i

                # ── Suppress early termination inside the sub-step loop ──
                if term and not trunc:
                    try:
                        _sat_hk   = self.env.unwrapped.satellites[0]
                        _t_now_hk = float(_sat_hk.simulator.sim_time)
                        _t_lim_hk = float(getattr(
                            self.env.unwrapped, 'time_limit',
                            getattr(self.env, 'time_limit', SIM_DURATION_S)
                        ))
                        if not _sat_hk.is_alive(log_failure=True):
                            logger.warning(
                                f'[FATAL] Satellite is not alive (failed check) inside sub-step loop at t={_t_now_hk:.0f}s. '
                                f'Terminating episode immediately.'
                            )
                        elif _t_now_hk < _t_lim_hk - 120.0:  # 120 s grace period
                            logger.debug(
                                f'[HOUSEKEEPING] bsk_rl early-term suppressed inside sub-step: '
                                f't={_t_now_hk:.0f}s < limit={_t_lim_hk:.0f}s — continuing sub-steps'
                            )
                            term = False
                    except Exception:
                        pass

                # ──────────────────────────────────────────────────────────────────────────────
                # [LOG4] WHAT HAPPENED DURING THIS SUB-STEP? SOC CHANGE? REWARD?
                # ──────────────────────────────────────────────────────────────────────────────
                try:
                    _sat_post = self.env.unwrapped.satellites[0]
                    _soc_post = float(_sat_post.dynamics.battery_charge_fraction)
                    _soc_delta = _soc_post - _soc_before_substep  # Use the PREVIOUS sub-step's SOC
                    # print(f"[LOG4] sub={_i_sub} | SOC: {_soc_before_substep:.4f} → {_soc_post:.4f} (Δ={_soc_delta:+.4f}) | r_i={r_i:+.4f}")
                    _soc_before_substep = _soc_post  # Update for next sub-step
                except Exception as e:
                    print(f"[LOG4] ERROR: {e}")
                
                self._log_state_1s(int(action))
                last_obs = obs_i
                if term or trunc:
                    break

                # Interrupted Drift Concept:
                if int(action) >= (N_STATIC_TARGETS + N_DYN_SLOTS):
                    try:
                        _sat_curr = self.env.unwrapped.satellites[0]
                        _now_curr = float(_sat_curr.simulator.sim_time)
                        _slots_curr = self._mgr.get_slots(_sat_curr, _now_curr)
                        _any_targetable = False
                        for _evt_curr in _slots_curr:
                            if _evt_curr is not None:
                                _slew_curr = _slew_safe(_sat_curr, _evt_curr)
                                if _slew_curr <= MAX_OFFNADIR_RAD:
                                    _any_targetable = True
                                    break
                        if _any_targetable:
                            logger.info(
                                f"[InterruptedDrift] Breaking out of drift early at sub-step "
                                f"{_i+1}/{n_sub} (t={_now_curr:.1f}s) because dynamic event became targetable."
                            )
                            tau = (_i + 1) * BASE_STEP_S
                            n_sub = _i + 1
                            break
                    except Exception as _e_drift:
                        logger.debug(f"[InterruptedDrift] Error checking targetability: {_e_drift}")
        finally:
            if orig_max_step_dur is not None:
                base_env.max_step_duration = orig_max_step_dur
            if orig_sim_max_step_dur is not None:
                base_env.simulator.max_step_duration = orig_sim_max_step_dur
          

        # FIX-2: drain Basilisk's image-taken buffer after a STATIC action so
        # the flag cannot bleed into the next action's sub-step and give a
        # duplicate reward (the static→drift +0.439/+0.495 bug in decisions.log).
        if int(action) < N_STATIC_TARGETS:
            try:
                sat.was_image_taken_since_last_check()  # drain; result discarded
            except Exception:
                pass

        if int(action) == N_STATIC_TARGETS + N_DYN_SLOTS:  # drift
            try:
                _soc = float(sat.dynamics.battery_charge_fraction)
                if _soc < MIN_BATTERY_SAFE_SOC:
                    total_r += 0.01   # encourage drift to recharge
            except Exception:
                pass

        # ── FIX-05: Housekeeping mode — suppress bsk_rl early termination ─────
        # bsk_rl sets term=True when it thinks there are no more windows.
        # If the simulation clock has NOT yet reached the time limit, override
        # term=False so the satellite keeps running in housekeeping/drift mode.
        # This ensures episodes always last the full 48 hours.
        if term and not trunc:
            try:
                _sat_hk   = self.env.unwrapped.satellites[0]
                _t_now_hk = float(_sat_hk.simulator.sim_time)
                _t_lim_hk = float(getattr(
                    self.env.unwrapped, 'time_limit',
                    getattr(self.env, 'time_limit', SIM_DURATION_S)
                ))
                if not _sat_hk.is_alive(log_failure=True):
                    logger.warning(
                        f'[FATAL] Satellite is not alive (failed check) at t={_t_now_hk:.0f}s. '
                        f'Terminating episode immediately.'
                    )
                elif _t_now_hk < _t_lim_hk - 120.0:  # 120 s grace period
                    logger.debug(
                        f'[HOUSEKEEPING] bsk_rl early-term suppressed: '
                        f't={_t_now_hk:.0f}s < limit={_t_lim_hk:.0f}s — continuing'
                    )
                    term = False
            except Exception:
                pass
        # ──────────────────────────────────────────────────────────────────────

            logger.debug(
               f"[SMDP] action={action}  tau={tau:.0f}s  n_sub={n_sub}  "
               f"total_r={total_r:.4f}  term={term}  trunc={trunc}"
            ) 
            smdp_discount = self._gamma_sub ** (tau / BASE_STEP_S)


        # ── [ROOT FIX] geometric imaging check — DISABLED (FIX-03 single-reward)
        # The main DYN reward injection block below is the sole place that adds
        # DYN reward.  Running _dyn_imaging_check() here as well created a
        # fragile dual-path that risked double-counting reward for the same event.
        # The main block has attempt shaping + battery penalty + full metrics.
        pass  # legacy pre-block disabled — reward injected by main DYN block below

        # ── DYN event reward injection ────────────────────────────────────
        # bsk_rl's imaging pipeline never fires for DynamicEvent targets
        # (no precomputed access windows).  We inject reward directly here.
        #
        # IMPORTANT: compare_log_states resets current_action_is_dynamic=False
        # and current_action_target=None before this code runs.  Therefore we
        # use P4's LOCKED event (_locked_dyn_event), which persists after
        # the bsk_rl step and is NOT touched by compare_log_states.
        
        _N_STATIC = N_STATIC_TARGETS  # = 20
           
        if _N_STATIC <= int(action) < _N_STATIC + N_DYN_SLOTS:
            try:
                info["dynamic_imaging_occurred"] = False
                _sat    = self.env.unwrapped.satellites[0]
                _slot   = int(action) - _N_STATIC
                # Use P4's locked event (survives compare_log_states reset)
                _target = getattr(_sat, '_locked_dyn_event', None)
                _l_slot = getattr(_sat, '_locked_dyn_slot',  -1)
                
                _offnadir_rad = _slew_safe(_sat, _target) if _target is not None else math.pi
                # Gate on off-nadir ≤ 45°, not pre-slew angle:

                _fired  = getattr(_sat, '_dyn_reward_given', False)
                _already_done = _target.imaged if _target else False

                logger.debug(
                    f"[DYN-CHECK] action={action}  slot={_slot}  "
                    f"target={'None' if _target is None else _target.name}  "
                    f"l_slot={_l_slot}  slew_deg={math.degrees(_offnadir_rad):.1f}  "
                    f"fired={_fired}  already_done={_already_done}  "
                    f"t_now={float(_sat.simulator.sim_time):.0f}  "
                    f"t_exp={getattr(_target,'expiration_time',0):.0f}"
                )
                if (_target is not None
                        and isinstance(_target, DynamicEvent)
                        and _l_slot == _slot
                        and _offnadir_rad <= math.radians(1.0)
                        and not _fired
                        and not _already_done
                        and _target.expiration_time > float(_sat.simulator.sim_time)):

                    _sat._dyn_reward_given = True
                    _cloud  = float(_target.cloud_cover)
                    _prio   = float(_target.priority)

                    # Urgency: HIGHER when fresh, LOWER near expiry.
                    # urgency = 1.0 + 0.5*(remaining/total) → 1.5 fresh, 1.0 expiring.
                    try:
                        _now        = float(_sat.simulator.sim_time)
                        _total_dur  = max(1.0, float(_target.expiration_time)
                                              - float(_target.appearance_time))
                        _remaining  = max(0.0, float(_target.expiration_time) - _now)
                        _frac_remaining = min(1.0, _remaining / _total_dur)
                        _urgency    = 1.0 + 0.5 * _frac_remaining   # 1.5 fresh → 1.0 old
                        logger.debug(
                            f"Urgency: remaining={_remaining:.0f}s  total={_total_dur:.0f}s  "
                            f"frac_remaining={_frac_remaining:.2f}  urgency={_urgency:.2f}"
                        )
                    except Exception:
                        _urgency = 1.0


                    # # Attempt shaping: small reward for valid DYN geometry
                    # # (already gated on _offnadir_rad <= MAX_OFFNADIR_RAD above)
                    # _attempt_shape = 0.05 * float(_prio)   # reduced: 0.10→0.05
                    # total_r += _attempt_shape

                    if _cloud < CLOUD_THRESH:
                        info["dynamic_imaging_occurred"] = True
                        _slew_mult   = getattr(_sat, '_slew_energy_multiplier', 1.0)
                        _slew_energy = calculate_slew_energy_wh(_offnadir_rad, _slew_mult)
                        # [FIX-DYN-1] Shift priority weight to static targets over dynamic events
                        _effective_mult = 1.2
                        _dyn_r = (_effective_mult * _prio * (1.0 - _cloud) * _urgency
                                 - SLEW_ENERGY_ALPHA * _slew_energy)
                        _sat._metrics['n_cloud_free'] += 1
                        _sat._metrics['n_dyn_imaged_clean'] = _sat._metrics.get('n_dyn_imaged_clean', 0) + 1
                    else:
                        # [FIX-DYN-2] Reduce cloudy penalty from -0.3 to -0.05 × priority.
                        # The -0.3 penalty is too harsh: the agent observed cloud_cover_forecast
                        # (noisy CNN) not ground truth. Penalize lightly to avoid suppressing
                        # dynamic exploration entirely.
                        _dyn_r = -0.05 * _prio
                        _sat._metrics['n_cloudy'] += 1

                    # Update metrics
                    _sat._metrics['n_imaged']      += 1
                    _sat._metrics['total_reward']  += _dyn_r
                    _sat._metrics['total_slew_angle_deg'] += math.degrees(_offnadir_rad)



                    # Increment event manager imaged counter
                    _evt_mgr = getattr(_sat, '_event_manager', None)
                    if _evt_mgr is not None:
                        _evt_mgr.mark_imaged(_target,
                                             float(_sat.simulator.sim_time),
                                             _dyn_r)
                        _sat._metrics['n_dyn_imaged'] = _evt_mgr._metrics['n_imaged']
                    else:
                        _sat._metrics['n_dyn_imaged'] += 1

                    # sync counter to info dict at each step
                    info.setdefault('episode_metrics', {})['n_dyn_imaged']       = _sat._metrics.get('n_dyn_imaged', 0)
                    info.setdefault('episode_metrics', {})['n_dyn_imaged_clean'] = _sat._metrics.get('n_dyn_imaged_clean', 0)
                    info.setdefault('episode_metrics', {})['n_static_imaged_clean'] = _sat._metrics.get('n_static_imaged_clean', 0)
                    info.setdefault('episode_metrics', {})['n_dyn_detected']     = _sat._metrics.get('n_dyn_detected', 0)
                    info.setdefault('episode_metrics', {})['n_imaged']           = _sat._metrics.get('n_imaged', 0)

                    soc = getattr(_sat, 'battery_charge_fraction', 1.0)
                    SOC_SAFETY = 0.3
                    if soc < SOC_SAFETY:
                        # Linearly scale down reward as battery depletes below safety threshold.
                        # This connects the observable SOC feature to a reward signal.
                        battery_penalty = max(0.0, 1.0 - soc / SOC_SAFETY)
                        _dyn_r *= (1.0 - 0.3 * battery_penalty)  # max 30% reduction

                    total_r += _dyn_r 

                    info['dyn_event_imaged'] = {
                        "name":     _target.name,
                        "type":     _target.event_type,
                        "lat":      float(math.degrees(_target.lat_rad)),
                        "lon":      float(math.degrees(_target.lon_rad)),
                        "priority": float(_target.priority),
                        "cloud":    float(_cloud),
                        "reward":   float(_dyn_r),
                    }

                    # Inside the DYN reward injection block, after total_r += _dyn_r * smdp_discount:
                    try:
                        _sat._last_dyn_event_log = {
                            "type":     _target.event_type,
                            "lat":      float(math.degrees(_target.lat_rad)),
                            "lon":      float(math.degrees(_target.lon_rad)),
                            "priority": float(_target.priority),
                            "cloud":    float(_cloud),
                            "reward":   float(_dyn_r),
                            "slot":     _slot,
                            "ep":       getattr(_sat, '_episode_count', 0),
                        }
                    except Exception:
                        pass

                    # print(f"DYN reward fired: {_dyn_r:.3f}")

                    
                    logger.debug(
                        f"DYN reward injected: r={_dyn_r:.3f}  "
                        f"cloud={_cloud:.2f}  urgency={_urgency:.2f}  "
                        f"event={type(_target).__name__}"
                    )
                else:
                    # FIX-DYN-FAIL: graduated penalty for failed dynamic actions
                    # Four causes are distinguished for clear RL signal:
                    if _target is None:
                        # Empty slot — agent selected a DYN slot with no event queued.
                        # Mild penalty: the agent could have drifted or picked a static target.
                        _fail_pen = -0.10
                        total_r  += _fail_pen
                        logger.debug(f"[DYN-FAIL] empty slot={_slot}  pen={_fail_pen:.3f}")
                    elif _already_done:
                        # Event already imaged — agent re-selected an exhausted slot.
                        # Mild penalty to discourage repeating done actions.
                        _fail_pen = -0.05
                        total_r  += _fail_pen
                        logger.debug(f"[DYN-FAIL] already-imaged  slot={_slot}  pen={_fail_pen:.3f}")
                    elif _target.expiration_time <= float(_sat.simulator.sim_time):
                        # Event expired — agent pointed at an event that timed out.
                        # Medium penalty proportional to what it could have earned.
                        _prio_f   = float(_target.priority)
                        _fail_pen = -0.3 * _prio_f
                        total_r  += _fail_pen
                        logger.debug(f"[DYN-FAIL] expired  slot={_slot}  prio={_prio_f:.2f}  pen={_fail_pen:.3f}")
                    elif _offnadir_rad > MAX_OFFNADIR_RAD:
                        # Inaccessible event — agent selected a dynamic target out of range.
                        _fail_pen = -0.20
                        total_r  += _fail_pen
                        logger.debug(f"[DYN-FAIL] inaccessible slot={_slot}  slew_deg={math.degrees(_offnadir_rad):.1f}  pen={_fail_pen:.3f}")
                    elif _offnadir_rad > math.radians(1.0) and not _fired:
                        # Still slewing toward event — not a real failure, just a sub-step.
                        # Small slew-energy cost only so the agent prefers efficient slews.
                        _slew_mult   = getattr(_sat, '_slew_energy_multiplier', 1.0)
                        _slew_energy = calculate_slew_energy_wh(_offnadir_rad, _slew_mult)
                        _cost        = SLEW_ENERGY_ALPHA * _slew_energy
                        total_r     -= _cost
                        logger.debug(f"[DYN-SLEW] still slewing  slot={_slot}  slew_deg={math.degrees(_offnadir_rad):.1f}  cost={_cost:.4f}")
                    # else: already_fired or some other edge case → no extra penalty
            except Exception as _exc:
                logger.debug(f"DYN reward injection error: {_exc}")

        # Drive event lifecycle
        try:
            sat  = self.env.unwrapped.satellites[0]
            now  = float(sat.simulator.sim_time)
            dt   = max(0.0, now - self._prev_time)
            new_events = self._gen.step(now, dt)

            self._mgr.add_events(new_events)
            for _evt in self._mgr._events:
                if getattr(_evt, '_was_accessible', False):
                    continue
                try:
                    _slew_chk = _slew_safe(sat, _evt)
                    if _slew_chk <= MAX_OFFNADIR_RAD:
                        _evt._was_accessible = True
                except Exception:
                    pass
            # [FIX-2] Missed-event penalty before purge (Li et al. IEEE TGRS 2023)
            # Missed-event penalty — only for cloud-free events the agent could have imaged.
            # Cloudy events are not imageable so no penalty. Cap total penalty per step
            # to prevent the baseline from dominating the reward signal at high event rates.
          
            _step_miss = 0.0
            _MISS_PER_STEP_CAP = 10.0   # Cap removed (raised to 10.0)
            _n_missed_cf = 0
            for _exp_evt in list(self._mgr._events):
                if not _exp_evt.imaged and _exp_evt.expiration_time <= now:
                    _cloud_e = float(_exp_evt.cloud_cover)
                    _prio_e  = float(_exp_evt.priority)
                    sat._metrics.setdefault('n_missed_events', 0)
                    sat._metrics['n_missed_events'] += 1
                    if _cloud_e >= CLOUD_THRESH:
                        continue
                    # [FIX-MISS-2] Only penalize events that were actually accessible
                    # (off-nadir ≤ 45° at any point in their lifetime). Penalizing
                    # geometrically impossible events gives the agent noise, not signal.
                    _was_accessible = getattr(_exp_evt, '_was_accessible', False)
                    if not _was_accessible:
                        continue
                    _pen = -0.5 * _prio_e * (1.0 - _cloud_e)   # Fully penalize missed events
                    _step_miss += _pen
                    _n_missed_cf += 1
            _miss_applied = max(-_MISS_PER_STEP_CAP, _step_miss)
            if _miss_applied != 0.0:
                logger.debug(
                    f"[MISS] step penalty={_miss_applied:.3f}  "
                    f"(raw={_step_miss:.3f}, n_cf_missed={_n_missed_cf})"
                )
            total_r += _miss_applied
            sat._metrics['total_reward'] += _miss_applied
            self._mgr.purge_expired(now)
            
            # Missed static target opportunity penalty
            # FIX-MISS-STATIC: only penalize when the window ENDS (now > t_end),
            # not on every sub-step inside the window (was double-counting).
            _step_static_miss = 0.0
            _STATIC_MISS_CAP = 10.0
            _n_static_missed_cf = 0
            _penalized = getattr(self, '_penalized_static_windows', set())
            for opp in sat.upcoming_opportunities:
                if opp["type"] == "target":
                    t_start, t_end = opp["window"]
                    # Only fire when window just expired (satellite passed over without imaging)
                    if t_end < now and t_end >= self._prev_time:
                        tgt = opp["object"]
                        try:
                            tgt_idx = sat.scenario.targets.index(tgt)
                        except ValueError:
                            continue
                        _win_key = (tgt_idx, round(t_end, 1))
                        if _win_key in _penalized:
                            continue
                        # Skip if agent WAS imaging this target during the window
                        _imaged_set = getattr(sat, '_imaged_static_set', set())
                        _tgt_id = getattr(tgt, 'name', id(tgt))
                        if _tgt_id in _imaged_set:
                            _penalized.add(_win_key)
                            continue
                        _cloud_s = float(tgt.cloud_cover)
                        if _cloud_s < CLOUD_THRESH:
                            _pen_s = -1.5 * float(tgt.priority) * (1.0 - _cloud_s)
                            _step_static_miss += _pen_s
                            _n_static_missed_cf += 1
                            sat._metrics.setdefault('n_static_missed', 0)
                            sat._metrics['n_static_missed'] += 1
                        _penalized.add(_win_key)
            self._penalized_static_windows = _penalized
            _static_miss_applied = max(-_STATIC_MISS_CAP, _step_static_miss)
            total_r += _static_miss_applied
            sat._metrics.setdefault('total_reward', 0.0)
            sat._metrics['total_reward'] += _static_miss_applied
            # print(f"[EVT-DBG] step t={now:.0f}s: gen.rate_hz={self._gen.rate_hz:.6f} "
            #     f"new_spawned={len(new_events)} active={len(self._mgr._events)} "
            #     f"total_detected={self._mgr._metrics['n_detected']}", flush=True)
            
            _active_now = [e for e in self._mgr._events if not e.imaged and e.expiration_time > now]
            logger.debug(
                f"[EVENTS] t={now:.0f}s  "
                f"new_spawned={len(new_events)}  active={len(_active_now)}  "
                f"total_detected={self._mgr._metrics['n_detected']}  "
                f"total_imaged={self._mgr._metrics['n_imaged']}"
            )
            self._prev_time = now
            sat._metrics['n_dyn_detected'] = self._mgr._metrics['n_detected']
        except Exception as exc:
            logger.debug(f"Event lifecycle error: {exc}")

        info['smdp_tau_s']       = tau
        info['smdp_n_sub']       = n_sub
        info['dynamic_metrics']  = self._mgr.get_metrics()

        # ── FIX-04: Termination reason logger ─────────────────────────────
        if term or trunc:
            # Final episode missed events penalty (ROOT FIX for unimaged active events)
            try:
                _sat_final = self.env.unwrapped.satellites[0]
                for _exp_evt in list(self._mgr._events):
                    if not _exp_evt.imaged:
                        _cloud_e = float(_exp_evt.cloud_cover)
                        _prio_e  = float(_exp_evt.priority)
                        _sat_final._metrics.setdefault('n_missed_events', 0)
                        _sat_final._metrics['n_missed_events'] += 1
                        if _cloud_e >= CLOUD_THRESH:
                            continue
                        _was_accessible = getattr(_exp_evt, '_was_accessible', False)
                        if not _was_accessible:
                            continue
                        _pen = -0.5 * _prio_e * (1.0 - _cloud_e)
                        total_r += _pen
                        _sat_final._metrics.setdefault('total_reward', 0.0)
                        _sat_final._metrics['total_reward'] += _pen
            except Exception as _exc_final:
                logger.debug(f"Final missed event penalty error: {_exc_final}")

            try:
                _sat_log = self.env.unwrapped.satellites[0]
                _t_now   = float(_sat_log.simulator.sim_time)
                _t_limit = float(getattr(
                    self.env.unwrapped, 'time_limit',
                    getattr(self.env, 'time_limit', SIM_DURATION_S)
                ))
                _reason = 'TIME_LIMIT' if trunc or abs(_t_now - _t_limit) < 120 else 'EARLY_bsk_rl'
                _active_dyn = [
                    e for e in self._mgr._events
                    if not e.imaged and e.expiration_time > _t_now
                ]
                _n_static_img = (
                    _sat_log._metrics.get('n_imaged', 0)
                    - _sat_log._metrics.get('n_dyn_imaged', 0)
                )
                _n_dyn_img  = _sat_log._metrics.get('n_dyn_imaged', 0)
                _total_rew  = _sat_log._metrics.get('total_reward', 0.0)
                _n_wins_left = 0
                try:
                    _opps_log = getattr(_sat_log, 'upcoming_opportunities', [])
                    for _opp_l in _opps_log:
                        _w_l = (_opp_l.get('window', [0,1]) if isinstance(_opp_l, dict)
                                else getattr(_opp_l, 'window', [0, 1]))
                        if _w_l[1] > _t_now:
                            _n_wins_left += 1
                except Exception:
                    pass
                # print(
                #     f'\n[EPISODE END] reason={_reason}  '
                #     f'sim_time={_t_now:.0f}s / {_t_limit:.0f}s  '
                #     f'({_t_now / _t_limit * 100:.1f}%)  |  '
                #     f'static_imaged={_n_static_img}  dyn_imaged={_n_dyn_img}  '
                #     f'reward={_total_rew:.2f}  |  '
                #     f'windows_left={_n_wins_left}  active_dyn={len(_active_dyn)}'
                # )
            except Exception as _log_exc:
                print(f'[EPISODE END] term={term} trunc={trunc}  (logger err: {_log_exc})')
        # ──────────────────────────────────────────────────────────────────

        tau_norm = tau / MAX_ACTION_DUR_S

        # ── Multi-objective: coverage completion bonus + static floor ───────
        if (term or trunc):
            try:
                _sat_ep = self.env.unwrapped.satellites[0]
                
                # End-of-episode: count active unimaged events as missed
                _active_unimaged = sum(1 for e in self._mgr._events if not e.imaged)
                _sat_ep._metrics.setdefault('n_missed_events', 0)
                _sat_ep._metrics['n_missed_events'] += _active_unimaged
                
                # Dynamic reward normalization to prevent stochastic event-count reward bias
                _n_dyn_det = _sat_ep._metrics.get('n_dyn_detected', 0)
                _n_dyn_img = _sat_ep._metrics.get('n_dyn_imaged_clean', 0)
                if _n_dyn_det > 0:
                    _dyn_suc = _n_dyn_img / float(_n_dyn_det)
                    # We want the total dynamic reward to be dyn_suc * DYN_TOTAL_WEIGHT
                    # 15.0 matches the maximum static coverage completion bonus (STATIC_COVERAGE_BONUSES[1.00] = 15.0)
                    _DYN_TOTAL_WEIGHT = 15.0
                    _target_dyn_r = _dyn_suc * _DYN_TOTAL_WEIGHT
                    _raw_dyn_r = self._mgr._metrics.get("total_dyn_reward", 0.0)
                    _dyn_correction = _target_dyn_r - _raw_dyn_r
                    
                    total_r += _dyn_correction
                    _sat_ep._metrics['total_reward'] += _dyn_correction
                    
                    logger.debug(
                        f"[DYN NORMALIZATION] dyn_suc={_dyn_suc:.4f} ({_n_dyn_img}/{_n_dyn_det}) → "
                        f"target_r={_target_dyn_r:.2f} (raw={_raw_dyn_r:.2f}) → correction={_dyn_correction:+.4f}"
                    )

                _n_static_clean = _sat_ep._metrics.get('n_static_imaged_clean', 0)
                _coverage_frac  = _n_static_clean / float(N_STATIC_TARGETS)

                # Coverage completion bonuses (one-time per threshold per episode)
                _bonus_awarded = getattr(_sat_ep, '_coverage_bonus_awarded', set())
                _bonus_total = 0.0
                for _thresh, _bonus in sorted(STATIC_COVERAGE_BONUSES.items()):
                    if _coverage_frac >= _thresh and _thresh not in _bonus_awarded:
                        total_r += _bonus
                        _bonus_total += _bonus
                        _bonus_awarded.add(_thresh)
                        logger.debug(
                            f"[COVERAGE BONUS] {_coverage_frac*100:.0f}% coverage → +{_bonus:.1f}  "
                            f"(n_static={_n_static_clean}/{N_STATIC_TARGETS})"
                        )
                _sat_ep._metrics['static_coverage_bonus_total'] = (
                    _sat_ep._metrics.get('static_coverage_bonus_total', 0.0) + _bonus_total
                )
                _sat_ep._metrics['static_coverage_frac'] = _coverage_frac
                _sat_ep._coverage_bonus_awarded = _bonus_awarded

                # Static floor penalty: scale by how many available windows were missed.
                # If agent never attempted static and had ≥2 windows: strong penalty.
                _n_static_attempts = getattr(self, '_n_static_actions_ep', 0)
                _n_avail = _sat_ep._metrics.get('n_static_available_windows', 0)
                if _n_static_clean == 0 and _n_avail >= 2:
                    # Strong penalty: completely ignored static targets despite available windows
                    _floor_pen = -3.0
                    total_r += _floor_pen
                    logger.debug(f"[STATIC FLOOR] zero static images, {_n_avail} windows avail → {_floor_pen:.1f}")
                elif _coverage_frac < 0.10 and _n_avail >= 5:
                    # Mild penalty: very low coverage despite many windows
                    _floor_pen = -1.0
                    total_r += _floor_pen
                    logger.debug(f"[STATIC FLOOR] <10% coverage, {_n_avail} windows avail → {_floor_pen:.1f}")

            except Exception as _cov_exc:
                logger.debug(f"[COVERAGE] error: {_cov_exc}")
                # Fallback to original static floor
                n_static = getattr(self, '_n_static_actions_ep', 0)
                if n_static == 0:
                    total_r -= 1.0

            self._n_static_actions_ep = 0  # reset for next episode
        # ─────────────────────────────────────────────────────────────────

        # ── Per-step reward floor (FIX-REWARD-CAP) ───────────────────────
        # Cap total reward per step. Increased from -2.0 to -5.0 to allow
        # coverage bonus signal (up to +15) and penalties (-3.0) through.
        total_r = float(np.clip(total_r, -5.0, None))
        # ─────────────────────────────────────────────────────────────────


        # ──────────────────────────────────────────────────────────────────────────────
        # [LOG5] FINAL STATE AT END OF STEP - DID SOC CRASH?
        # ──────────────────────────────────────────────────────────────────────────────
        try:
            _sat_final = self.env.unwrapped.satellites[0]
            _soc_final = float(_sat_final.dynamics.battery_charge_fraction)
            _soc_total_change = _soc_final - _soc  # _soc is from LOG1
            _is_crash = _soc_total_change < -0.10
            _crash_flag = "🚨 HUGE DROP!" if _is_crash else ""
            # print(f"[LOG5] FINAL | action={action:>2d} | SOC: {_soc:.4f} → {_soc_final:.4f} (Δ={_soc_total_change:+.4f}) {_crash_flag} | total_reward={total_r:+.4f}")
        except Exception as e:
            print(f"[LOG5] ERROR: {e}")
 
        # ─────────────────────────────────────────────────────────────────────────────

        # Store raw satellite state in info for logging (prevents reset synchronization issues)
        try:
            sat = self.env.unwrapped.satellites[0]
            info['episode_metrics'] = dict(sat._metrics)
            info['sat_state_raw'] = {
                'battery_charge_fraction': float(sat.dynamics.battery_charge_fraction),
                'sim_time': float(sat.simulator.sim_time),
                'r_SC_N': np.asarray(
                    getattr(sat.dynamics, "r_SC_N", None) or
                    getattr(sat.dynamics, "r_BN_N", None) or
                    getattr(sat.dynamics, "r_N",    None),
                    dtype=float
                ).tolist()
            }
        except Exception:
            pass

        # FIX-LOG-RAW: store raw reward so logger/VecNormalize don't hide it
        info['raw_reward'] = float(total_r)
        return self._build_obs(last_obs, tau_norm), total_r, term, trunc, info

    # ---- observation builder ------------------------------------------------

    def _build_obs(self, base_obs: np.ndarray, tau_norm: float) -> np.ndarray:
        try:
            sat   = self.env.unwrapped.satellites[0]
            now   = float(sat.simulator.sim_time)
            slots = self._mgr.get_slots(sat, now)
        except Exception:
            slots = [None] * N_DYN_SLOTS
            sat   = None
            now   = 0.0

        # ── Dynamic event features ────────────────────────────────────────
        feats = []
        for evt in slots:
            if evt is None:
                feats.extend([0.0, -1.0, 1.0, 0.0])
            else:
                try:
                    slew = _slew_safe(sat, evt)
                    tta  = _compute_tta(sat, evt, now)
                    feats.extend([
                        float(np.clip(evt.priority,             0.0, 1.0)),
                        float(np.clip(evt.cloud_cover_forecast, 0.0, 1.0)),
                        float(np.clip(tta / TIME_NORM_S,        0.0, 1.0)),
                        float(np.clip(slew / (math.pi / 2),     0.0, 1.0)),
                    ])
                except Exception:
                    feats.extend([0.0, -1.0, 1.0, 0.0])

        dyn_arr = np.array(feats, dtype=np.float32)

        # ── TARGET-ID FIX: inject action index into each static slot ─────
        # PROBLEM (raised by RL expert): obs[13:43] has 6 target slots × 5
        # features, but NONE encode WHICH action index (0-19) maps to each slot.
        # The bsk_rl top-6 ranking re-sorts every step, so the policy cannot
        # learn a stable slot→action mapping — it acts partially blind.
        #
        # FIX: append target_idx / N_TARGETS as the 6th feature per slot.
        # Now slot i says "these features belong to action X" — the attention
        # head can learn to score slot i and output action X reliably.
        # Obs: 56 → 62 dims  (6 extra target-ID scalars, one per slot).
        #
        # Reference: Herrmann & Schaub (2023) §4.2 pointer-style action head.
        try:
            all_targets = list(sat.scenario.targets)   # fixed order: index = action
            opps = [o for o in getattr(sat, 'upcoming_opportunities', [])
                    if isinstance(o, dict) and o.get('type') == 'target']
            opps = sorted(opps, key=lambda o: o['window'][0])[:6]

            state_part  = base_obs[:13].astype(np.float32)    # satellite state (unchanged)
            target_part = base_obs[13:43].astype(np.float32)  # 6×5 from bsk_rl

            new_target_feats = []
            for slot_i in range(6):
                slot_feats = target_part[slot_i * 5: slot_i * 5 + 5]
                if slot_i < len(opps):
                    tgt = opps[slot_i].get('object', None)
                    try:
                        idx = all_targets.index(tgt)
                    except (ValueError, AttributeError):
                        idx = -1
                    tid = float(idx) / float(N_STATIC_TARGETS)   # ∈ [0, 0.95]
                else:
                    tid = -1.0 / float(N_STATIC_TARGETS)         # sentinel: no target
                new_target_feats.extend(slot_feats.tolist())
                new_target_feats.append(tid)

            # extended_base: 13 (state) + 36 (6 slots × 6 feats) = 49
            extended_base = np.concatenate([
                state_part,
                np.array(new_target_feats, dtype=np.float32),
            ])
        except Exception:
            # Fallback: pad base_obs with zeros to keep total dim consistent
            extended_base = np.concatenate([
                base_obs.astype(np.float32),
                np.zeros(OBS_TARGET_ID_DIM, dtype=np.float32),
            ])

        # ── Static coverage fraction feature ─────────────────────────────
        # Tells the agent how much of the 45-target set has been imaged so far.
        # 0.0 = nothing imaged, 1.0 = all 45 done. Guides toward coverage goals.
        try:
            _sat_obs = self.env.unwrapped.satellites[0]
            _n_done = len(getattr(_sat_obs, '_imaged_static_set', set()) or set())
            _cov_feat = float(np.clip(_n_done / float(N_STATIC_TARGETS), 0.0, 1.0))
            # Also update availability counter for floor penalty
            _opps_obs = [o for o in getattr(_sat_obs, 'upcoming_opportunities', [])
                         if (o.get('type') == 'target' if isinstance(o, dict)
                             else getattr(o, 'type', '') == 'target')]
            _sat_obs._metrics['n_static_available_windows'] = max(
                _sat_obs._metrics.get('n_static_available_windows', 0),
                len(_opps_obs)
            )
        except Exception:
            _cov_feat = 0.0
        coverage_arr = np.array([_cov_feat], dtype=np.float32)

        # Final obs: 49 (extended_base) + 12 (dyn) + 1 (sojourn) + 1 (coverage) = 63
        final_obs = np.concatenate([extended_base, dyn_arr, [np.clip(tau_norm, 0.0, 1.0)], coverage_arr],
                               dtype=np.float32)
        return np.nan_to_num(final_obs, nan=0.0)

    # ---- convenience properties ---------------------------------------------

    @property
    def event_manager(self) -> EventManager:
        return self._mgr

    @property
    def event_generator(self) -> EventGenerator:
        return self._gen

    def close(self):
        if hasattr(self, "_sim_log_file") and self._sim_log_file is not None:
            try:
                self._sim_log_file.close()
            except Exception:
                pass
            self._sim_log_file = None
        return self.env.close()

# =============================================================================
#  Factory
# =============================================================================

def make_dynamic_env(
    targets_path:    str,
    cloud_json_path: str,
    event_rate:      float = 2.0,
    duration_s:      float = SIM_DURATION_S,
    sim_rate:        float = BSK_SIM_RATE_S,
    sat_name:        str   = 'ALSAT-1',
    sat_args:        dict  = None,
    cloud_model              = None,   # pass VisionCloudModel to use CNN forecasts
    safety_monitor           = None,   # pass SafetyMonitor instance
    gamma:           float = DEFAULT_GAMMA,
    seed:            int   = 42,
    render_mode              = None,
) -> DynamicObsWrapper:
    """
    Build the Phase 3 SMDP dynamic targeting environment.

    Returns DynamicObsWrapper  obs=(56,)  actions=Discrete(24)

    Parameters
    ----------
    cloud_model   : if None, uses ModisCloudModel (Gaussian noise).
                    Pass a VisionCloudModel instance to use CNN forecasts.
    safety_monitor: if provided, the satellite's action handler will veto
                    unsafe actions before calling task_target_for_imaging().
    gamma         : SMDP discount factor (per STEP_REF_S = 1200 s)
    """
    targets_cfg  = load_targets_config(targets_path)
    if cloud_model is None:
        cloud_model = ModisCloudModel(cloud_json_path, seed=seed)
    scenario     = AlsatScenario(targets_cfg, cloud_model)
    gen_duration = duration_s

    event_gen = EventGenerator(rate_per_hour=event_rate, seed=seed)
    event_mgr = EventManager()


    satellite = DynamicAlsatSatellite(
        name=sat_name, sat_args=sat_args, scenario=scenario,
        event_manager=event_mgr, safety_monitor=safety_monitor,
        generation_duration=gen_duration, initial_generation_duration=gen_duration , #+ 7200 + 7200
    )

    base_env = GeneralSatelliteTasking(
        satellites=[satellite], scenario=scenario,
        rewarder=DynamicScienceReward(reward_scale=1.0),
        time_limit=duration_s, sim_rate=sim_rate,
        max_step_duration=SCHED_STEP_S, render_mode=render_mode, world_type=AtmosphereEclipseWorldModel,
    )

    flat_env = SingleSatelliteEnv(base_env)
    return DynamicObsWrapper(flat_env, event_gen, event_mgr, gamma=gamma, seed=seed)


# =============================================================================
#  Backwards-compatibility shim
# =============================================================================

def make_smdp_dynamic_env(*args, **kwargs) -> DynamicObsWrapper:
    """Deprecated: make_dynamic_env() already returns a full SMDP env."""
    import warnings
    warnings.warn(
        "make_smdp_dynamic_env() is deprecated. Use make_dynamic_env() directly -- "
        "the SMDP wrapper is now built into DynamicObsWrapper.",
        DeprecationWarning, stacklevel=2)
    kwargs.pop('max_sub_steps', None)
    return make_dynamic_env(*args, **kwargs)


# =============================================================================
#  Quick sanity test
# =============================================================================

if __name__ == '__main__':
    import os, logging
    os.environ.setdefault('BSK_OUTPUT_LEVEL', '2')
    os.environ.setdefault('BSK_LOG_LEVEL',    'WARNING')
    logging.basicConfig(level=logging.INFO)

    import path_setup
    ROOT       = path_setup.root_path()
    TARGETS    = os.path.join(ROOT, 'config/targets/global_45_targets.jsonn')
    CLOUD_JSON = os.path.join(ROOT, 'config/cloud_reality/global_45_clouds.json')

    print('=' * 68)
    print('env_alsat_dynamic.py  --  Phase 3 SMDP sanity check')
    print(f'  Keplerian TTA : {"enabled" if _HAS_KEPLERIAN else "fallback (binary)"}')
    print(f'  DECAY_TAU     : {EVENT_DECAY_TAU_S:.0f} s')
    print(f'  OBS_TOTAL_DIM : {OBS_TOTAL_DIM}')
    print('=' * 68)

    env = make_dynamic_env(TARGETS, CLOUD_JSON, event_rate=2.0, seed=42)
    obs, info = env.reset(seed=42)
    assert obs.shape == (OBS_TOTAL_DIM,), f"Bad obs: {obs.shape}"
    assert env.action_space.n == N_TOTAL_ACTIONS
    print(f'  obs={obs.shape}  actions={env.action_space}  OK')
    print(f'  base[0:6]={obs[:6].round(3)}')
    print(f'  dyn [43:55]={obs[43:55].round(3)}')
    print(f'  sojourn[55]={obs[55]:.3f}  (0 at reset)')

    print('\n  Running 8 steps (mix static + dynamic + drift)...')
    for i in range(8):
        act = [5, 20, 21, 23, 10, 22, 0, 23][i]
        obs, r, term, trunc, info = env.step(act)
        print(f'  step {i+1}  act={act:2d}  r={r:+.4f}  '
              f'tau={info["smdp_tau_s"]:.0f}s  nsub={info["smdp_n_sub"]}  '
              f'dyn_det={info["dynamic_metrics"]["n_detected"]}  '
              f'sojourn={obs[55]:.3f}')
        if term or trunc:
            break

        

    env.close()
    print('\nSanity check passed.')


# ── [ROOT FIX] DYN geometric imaging bypass ───────────────────────────────
import numpy as _np_dyn, math as _math_dyn
from dynamic_event import DYN_MULTIPLIER as _DYN_MULT

_DYN_MAX_OFFNADIR_DEG = 45.0   # must match MAX_OFFNADIR_RAD in dynamic_event.py
_DYN_CLOUD_THRESH     = CLOUD_THRESH    # max cloud cover for successful imaging


def _dyn_imaging_check(sat, info: dict) -> float:
    """
    Called at the end of each SMDP step.
    Returns extra reward if satellite is geometrically pointing at locked DYN event.
    Also increments episode_metrics['n_dyn_imaged'].
    """
    if getattr(sat, '_locked_dyn_event', None) is None:
        return 0.0
    locked_ev = getattr(sat, '_locked_dyn_event', None)
    if locked_ev is None:
        return 0.0
    if getattr(sat, '_dyn_reward_given', False):
        return 0.0   # already credited this event

    try:
        r_sat = _np_dyn.asarray(sat.dynamics.r_BN_P, dtype=float).flatten()
        r_evt = _np_dyn.asarray(locked_ev.r_LP_P,    dtype=float).flatten()
    except AttributeError:
        return 0.0   # dynamics not initialised yet

    norm_sat = float(_np_dyn.linalg.norm(r_sat))
    if norm_sat < 1e3:
        return 0.0

    # Off-nadir angle: angle between nadir direction and vector to event
    nadir_unit = -r_sat / norm_sat
    to_evt = r_evt - r_sat
    d = float(_np_dyn.linalg.norm(to_evt))
    if d < 1.0:
        return 0.0
    to_evt_unit = to_evt / d

    cos_a = float(_np_dyn.clip(_np_dyn.dot(nadir_unit, to_evt_unit), -1.0, 1.0))
    offnadir_deg = _math_dyn.degrees(_math_dyn.acos(cos_a))

    cloud = float(getattr(locked_ev, 'cloud_cover', 1.0))

    if offnadir_deg <= _DYN_MAX_OFFNADIR_DEG and cloud < _DYN_CLOUD_THRESH:
        sat._dyn_reward_given = True
        ep = info.setdefault('episode_metrics', {})
        ep['n_dyn_imaged'] = ep.get('n_dyn_imaged', 0) + 1
        if hasattr(locked_ev, 'mark_accessed'):
            locked_ev.mark_accessed()
        pri = float(getattr(locked_ev, 'priority', 1.0))
        return DYN_MULTIPLIER * pri * (1.0 - cloud)

    return 0.0


# ── Apply skipped patches ───────────────────────────────────────────────────
try:
    import bsk_patches
    bsk_patches._patch_dyn_event_locking()
except Exception as _e:
    logger.warning(f"Failed to apply dynamic event locking patch: {_e}")

