"""
FIX 03 — env_alsat_dynamic.py  remove double dynamic reward risk
=================================================================
DynamicObsWrapper.step() has TWO code paths that can inject a DYN reward:

  Path A (lines ~2590-2617): the "ROOT FIX geometric bypass" block that calls
    _dyn_imaging_check(), adds total_r += _dyn_r, sets _dyn_reward_given=True,
    and clears _locked_dyn_event.

  Path B (lines ~2630-2768): the main "DYN event reward injection" block that
    checks `not _fired` (= not _dyn_reward_given) before firing.

These are mutually exclusive in normal operation, but the interlock is fragile:
if _dyn_imaging_check fires and THEN an exception clears _dyn_reward_given
before Path B runs, both can fire for the same event.

Fix: disable Path A (the _dyn_imaging_check pre-block).
Path B is more complete (attempt shaping, battery penalty, metrics update) and
is the single authoritative place for DYN reward.  Path A is legacy.
"""

OLD = '''        # ── [ROOT FIX] geometric imaging check (non-critical, wrapped safely)
        # NOTE: do NOT reset _locked_dyn_event here — the main injection
        # block below (lines 506+) uses it and must see it intact.
        try:
            _sat = self.env.unwrapped.satellites[0]
            # [FIX-B] Only run geometric check if there is actually a locked event.
            # Without this guard, _dyn_imaging_check fires on empty slots and
            # gives the agent free +0.29 rewards for doing nothing.
            _locked_evt = getattr(_sat, '_locked_dyn_event', None)
            _has_locked = (
                _locked_evt is not None
                and not getattr(_locked_evt, 'imaged', False)
                and _locked_evt.expiration_time > float(_sat.simulator.sim_time)
            )
            if _has_locked:
                _dyn_r = _dyn_imaging_check(_sat, info)
                if _dyn_r > 0.0:
                    total_r += _dyn_r
                    _sat._dyn_reward_given = True
                    _sat._locked_dyn_event = None
                    _sat._locked_dyn_slot  = None
                    # FIX: _dyn_imaging_check writes to info only.
                    # SingleSatelliteEnv overwrites info with dict(sat._metrics) at
                    # episode end → n_dyn_imaged stays 0 → dyn_suc=0% always.
                    _sat._metrics['n_dyn_imaged'] = (
                        _sat._metrics.get('n_dyn_imaged', 0) + 1)
        except Exception:
            pass'''

NEW = '''        # ── [ROOT FIX] geometric imaging check — DISABLED (FIX-03 single-reward)
        # The main DYN reward injection block below is the sole place that adds
        # DYN reward.  Running _dyn_imaging_check() here as well created a
        # fragile dual-path that risked double-counting reward for the same event.
        # The main block has attempt shaping + battery penalty + complete metrics,
        # so it is strictly better.  Nothing is lost by disabling this pre-block.
        pass  # legacy geometric-check block removed — see main DYN block below'''


def apply(text: str) -> str:
    if OLD not in text:
        raise ValueError(
            "FIX-03: ROOT FIX geometric check block not found in env_alsat_dynamic.py.\n"
            "File may already be patched or structure changed."
        )
    return text.replace(OLD, NEW, 1)