"""
FIX 07 — env_alsat_dynamic.py  per-step diagnostic line
========================================================
When the environment variable ALSAT_DEBUG_STEPS=1 is set, print a one-line
status every 12 steps showing:
  sim_time, battery%, remaining windows, last action, reward

This is opt-in via env var so it does not pollute normal training output.

Usage:
  ALSAT_DEBUG_STEPS=1 python scripts/training/train_full_system.py --quick
"""

import os as _os

OLD = '''        # ── Static-imaging floor penalty ─────────────────────────────────
        # If agent took ZERO static actions this episode, penalize.
        # Prevents catastrophic forgetting of scheduled imaging.
        if (term or trunc):'''

NEW = '''        # ── FIX-07: Per-step diagnostic line (opt-in via ALSAT_DEBUG_STEPS=1) ──
        import os as _os_diag
        if _os_diag.environ.get('ALSAT_DEBUG_STEPS') == '1':
            try:
                _sd_sat  = self.env.unwrapped.satellites[0]
                _sd_t    = float(_sd_sat.simulator.sim_time)
                _sd_batt = float(getattr(
                    getattr(_sd_sat, 'dynamics', _sd_sat),
                    'battery_charge_fraction', 1.0
                )) * 100.0
                _sd_opps = getattr(_sd_sat, 'upcoming_opportunities', [])
                _sd_wins = sum(
                    1 for _o in _sd_opps
                    for _w in [(_o.get('window',[0,1]) if isinstance(_o,dict)
                                else getattr(_o,'window',[0,1]))]
                    if _w[1] > _sd_t
                )
                _sd_act_name = (
                    'DRIFT' if int(action) >= N_STATIC_TARGETS + N_DYN_SLOTS
                    else (f'DYN{int(action)-N_STATIC_TARGETS}' if int(action) >= N_STATIC_TARGETS
                          else f'TGT{int(action)}')
                )
                _step_num = getattr(self, '_step_count_diag', 0) + 1
                self._step_count_diag = _step_num
                if _step_num % 12 == 1:
                    print(
                        f"  [STEP {_step_num:3d}] t={_sd_t:7.0f}s  "
                        f"batt={_sd_batt:5.1f}%  wins={_sd_wins:3d}  "
                        f"act={_sd_act_name:<8s}  r={total_r:+.4f}"
                    )
            except Exception:
                pass
        # ──────────────────────────────────────────────────────────────────

        # ── Static-imaging floor penalty ─────────────────────────────────
        # If agent took ZERO static actions this episode, penalize.
        # Prevents catastrophic forgetting of scheduled imaging.
        if (term or trunc):'''


def apply(text: str) -> str:
    if OLD not in text:
        raise ValueError(
            "FIX-07: static-imaging floor penalty block not found in env_alsat_dynamic.py."
        )
    return text.replace(OLD, NEW, 1)
