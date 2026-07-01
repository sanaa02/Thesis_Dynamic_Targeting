"""
FIX 05 — env_alsat_dynamic.py  housekeeping mode (no early termination)
========================================================================
bsk_rl's GeneralSatelliteTasking sets term=True when it decides there are
no more imaging opportunities.  For a 48-hour episode this can fire after
only a few orbits if windows are thinly distributed.

Fix: in DynamicObsWrapper.step(), after the sub-step loop, if term=True
but the simulation clock has NOT yet reached the time limit (with a 120-s
tolerance), override term=False so the episode keeps running.  The satellite
continues in housekeeping/drift mode and can still service dynamic events.

This matches the spec requirement "Ensure episodes always last 144 decision
steps (48 hours) unless the time limit is reached."
"""

OLD = '''        DRIFT_ACT = N_STATIC_TARGETS + N_DYN_SLOTS  # = 23
        total_r = 0.0
        for _i in range(n_sub):
            _sub_a = action 
            try:

                # Keep sat._event_manager pointing at self._mgr (survives bsk_rl resets)
                for _sx in self.env.unwrapped.satellites:
                    _sx._event_manager = self._mgr
            except Exception:
                pass
            obs_i, r_i, term, trunc, info = self.env.step(_sub_a)
            total_r += (self._gamma_sub ** _i) * r_i
            last_obs = obs_i
            if term or trunc:
                break'''

NEW = '''        DRIFT_ACT = N_STATIC_TARGETS + N_DYN_SLOTS  # = 23
        total_r = 0.0
        for _i in range(n_sub):
            _sub_a = action 
            try:

                # Keep sat._event_manager pointing at self._mgr (survives bsk_rl resets)
                for _sx in self.env.unwrapped.satellites:
                    _sx._event_manager = self._mgr
            except Exception:
                pass
            obs_i, r_i, term, trunc, info = self.env.step(_sub_a)
            total_r += (self._gamma_sub ** _i) * r_i
            last_obs = obs_i
            if term or trunc:
                break

        # ── FIX-05: Housekeeping mode — suppress bsk_rl early termination ─────
        # If bsk_rl says "done" (term=True) but the clock hasn't reached the
        # time limit yet, this is a spurious "no more windows" termination.
        # Override it so the satellite keeps running and can image dynamic events.
        if term and not trunc:
            try:
                _sat_hk   = self.env.unwrapped.satellites[0]
                _t_now_hk = float(_sat_hk.simulator.sim_time)
                _t_lim_hk = float(getattr(
                    self.env.unwrapped, 'time_limit',
                    getattr(self.env, 'time_limit', SIM_DURATION_S)
                ))
                if _t_now_hk < _t_lim_hk - 120.0:  # 120 s tolerance
                    logger.debug(
                        f"[HOUSEKEEPING] bsk_rl early-term suppressed: "
                        f"t={_t_now_hk:.0f}s < limit={_t_lim_hk:.0f}s — continuing"
                    )
                    term = False
            except Exception:
                pass
        # ──────────────────────────────────────────────────────────────────────'''


def apply(text: str) -> str:
    if OLD not in text:
        raise ValueError(
            "FIX-05: sub-step loop block not found in env_alsat_dynamic.py.\n"
            "File may already be patched."
        )
    return text.replace(OLD, NEW, 1)