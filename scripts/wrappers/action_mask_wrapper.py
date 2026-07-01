#!/usr/bin/env python3
"""
action_mask_wrapper.py  --  Constraint-Aware Action Masking  (CORRECTED)
=======================================================================
ROOT-CAUSE FIX (static imaging "picks targets but never images them"):

    The previous "FAST v2" version masked a static target as ACCESSIBLE
    whenever the geometric angle from the satellite's CURRENT boresight to
    the target was <= 45 deg (calculate_slew_angle_to_target <= MAX_OFFNADIR_RAD).

    That condition has NOTHING to do with whether bsk_rl can actually image
    the target.  bsk_rl only confirms a static image (was_image_taken... = True,
    which is the SOLE source of static reward) when the target is inside a real
    ACCESS WINDOW that bsk_rl itself computed (the same windows exposed to the
    policy in `upcoming_opportunities` and in the 6 observation slots).

    Because the geometric mask marked ~20 targets "valid" every step, MaskablePPO
    spent all its exploration selecting targets that had no active window ->
    bsk_rl never imaged them -> reward 0 -> the policy could not learn.
    (Confirmed in decisions.log: ~3/574 static attempts succeeded.)

    FIX: a static target is selectable ONLY if it currently appears in
    `upcoming_opportunities` as a 'target' whose window has not yet expired and
    opens within REACH_HORIZON_S (so the satellite can slew to it in time).
    This makes the set of *selectable* actions identical to the set of *imageable*
    targets the policy can also see in its observation.

Unchanged:
  * layout inference from action_space.n (45 static + 3 dyn + 1 drift = 49)
  * DYN-slot masking (mask empty slots)
  * battery-aware masking (only drift below MIN_BATTERY_SAFE_SOC)
"""
import logging
import numpy as np
import gymnasium as gym

MIN_BATTERY_SAFE_SOC = 0.30   # must match env_alsat_dynamic.py / bsk_patches.py
REACH_HORIZON_S      = 1200.0 # allow a target whose window opens within one
                              # scheduler step (= SCHED_STEP_S) so the satellite
                              # has time to slew before the window closes.
logger = logging.getLogger(__name__)


def _opp_field(opp, key, default):
    if isinstance(opp, dict):
        return opp.get(key, default)
    return getattr(opp, key, default)


class ActionMaskWrapper(gym.Wrapper):
    """Provides get_action_mask() aligned with bsk_rl access windows."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._n_static, self._n_dyn = self._infer_layout()
        self._drift_idx = self.action_space.n - 1
        self._last_mask = np.ones(self.action_space.n, dtype=bool)
        logger.debug(
            f"[MASK] Init: n_static={self._n_static} n_dyn={self._n_dyn} "
            f"n_total={self.action_space.n}"
        )

    def _infer_layout(self):
        """(N_STATIC, N_DYN) derived from the fixed action space (49)."""
        try:
            from dynamic_event import N_DYN_SLOTS as _NDYN
        except Exception:
            _NDYN = 3
        n_dyn    = int(_NDYN)
        n_static = int(self.action_space.n) - n_dyn - 1   # 49 - 3 - 1 = 45
        return max(0, n_static), max(0, n_dyn)

    def get_action_mask(self) -> np.ndarray:
        return self._compute_mask()

    def _compute_mask(self) -> np.ndarray:
        n    = self.action_space.n
        mask = np.zeros(n, dtype=bool)   # default: nothing valid except drift

        try:
            obj = self.env
            mgr = None
            while hasattr(obj, "env"):
                if mgr is None:
                    mgr = getattr(obj, "_mgr", None)
                obj = obj.env
            base = getattr(obj, "unwrapped", obj)
            sat  = base.satellites[0]

            now  = float(sat.simulator.sim_time)
            targets = list(sat.scenario.targets)
            opps = [o for o in getattr(sat, "upcoming_opportunities", [])
                    if _opp_field(o, "type", "") == "target"]

            # ── Static targets: selectable ONLY if they have a real bsk_rl
            #    access window that is open now or opens within REACH_HORIZON_S
            #    and has not yet expired.  This is exactly the imaging condition
            #    bsk_rl uses, so "selectable" == "imageable". ───────────────────
            _done = getattr(sat, "_imaged_static_set", None) or set()
            for opp in opps:
                tgt = _opp_field(opp, "object", None)
                win = _opp_field(opp, "window", None)
                if tgt is None or win is None:
                    continue
                try:
                    i = targets.index(tgt)
                except (ValueError, AttributeError):
                    continue
                if i >= self._n_static:
                    continue
                try:
                    w0, w1 = float(win[0]), float(win[1])
                except Exception:
                    continue
                # window must not be expired and must be reachable soon
                if w1 < now or w0 > now + REACH_HORIZON_S:
                    continue
                # don't re-image targets already imaged this episode
                if getattr(tgt, "imaged", False) or getattr(tgt, "name", None) in _done:
                    continue
                # skip clearly cloudy targets (same 0.6 thresh as reward fn)
                try:
                    fcst = float(getattr(tgt, "cloud_cover_forecast",
                                         getattr(tgt, "cloud_cover", 0.0)))
                    if fcst > 0.6:
                        continue
                except Exception:
                    pass
                mask[i] = True

            # ── DYN slots: only unmask slots that actually hold an accessible event ──────
            if mgr is None:
                mgr = getattr(sat, "_event_manager", None)
            if mgr is not None:
                slots = mgr.get_slots(sat, now)
                for j in range(self._n_dyn):
                    event = slots[j] if j < len(slots) else None
                    if event is not None:
                        try:
                            import math
                            from dynamic_event import MAX_OFFNADIR_RAD
                            r_sat = np.asarray(sat.dynamics.r_BN_P, dtype=float).ravel()
                            r_tgt = np.asarray(event.r_LP_P, dtype=float).ravel()
                            los   = r_tgt - r_sat
                            los_n = np.linalg.norm(los)
                            sat_n = np.linalg.norm(r_sat)
                            if los_n < 1.0 or sat_n < 1.0:
                                off_nadir = 0.0
                            else:
                                nadir = -r_sat / sat_n
                                los_unit = los / los_n
                                dot = float(np.clip(np.dot(nadir, los_unit), -1.0, 1.0))
                                off_nadir = float(math.acos(dot))
                            mask[self._n_static + j] = bool(off_nadir <= MAX_OFFNADIR_RAD)
                        except Exception:
                            mask[self._n_static + j] = True
                    else:
                        mask[self._n_static + j] = False
            else:
                for j in range(self._n_dyn):
                    mask[self._n_static + j] = False

        except Exception as exc:
            logger.debug(f"[MASK] Error: {exc}; falling back to drift-only")
            mask[:] = False

        # ── Battery-aware masking: below threshold only drift is allowed ──────
        try:
            sat_bm = getattr(getattr(self.env, "unwrapped", self.env),
                             "satellites", [None])[0]
            if sat_bm is not None:
                soc_bm = float(sat_bm.dynamics.battery_charge_fraction)
                if soc_bm < MIN_BATTERY_SAFE_SOC or np.isnan(soc_bm):
                    mask[:] = False
        except Exception as e:
            logger.debug(f"[MASK] battery check error: {e}")

        mask[self._drift_idx] = True   # drift always valid
        self._last_mask = mask
        return mask

    def step(self, action):
        # Permanent safety shield: redirect invalid actions (both static and dynamic)
        # to DRIFT at runtime. This protects the environment even if model.predict()
        # is called without action_masks during evaluation/inference.
        mask = self._compute_mask()
        if not mask[int(action)]:
            action = self._drift_idx
            logger.debug(f"[MASK] Redirected invalid action to DRIFT")
        return self.env.step(action)


class InferenceTimeMaskWrapper(ActionMaskWrapper):
    """Fallback when sb3-contrib is not installed (inherits permanent shield)."""
    pass


def make_masked_env(base_env: gym.Env) -> gym.Env:
    try:
        from sb3_contrib.common.wrappers import ActionMasker
        wrapped = ActionMaskWrapper(base_env)
        logger.info("[MASK] sb3-contrib ActionMasker attached")
        return ActionMasker(wrapped, lambda e: e.get_action_mask())
    except ImportError:
        logger.warning("[MASK] sb3-contrib not found — using InferenceTimeMaskWrapper")
        return InferenceTimeMaskWrapper(base_env)