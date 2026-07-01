"""
FIX 02 — env_alsat_dynamic.py  correct urgency direction
=========================================================
The DYN reward injection block computed urgency using *elapsed* time,
so urgency grew as the event aged (lowest when fresh, highest near expiry).
The correct direction is HIGHER when fresh, LOWER near expiry:
  urgency = 1.0 + 0.5 * (remaining / total_lifetime)
  → 1.5 when brand-new, 1.0 when about to expire.

This encourages the agent to respond quickly to new events rather than
procrastinating until they are about to expire.
"""

OLD = '''                    # Urgency: newer events pay more
                    try:
                        _now        = float(_sat.simulator.sim_time)
                        _total_dur  = max(1.0, float(_target.expiration_time)
                                              - float(_target.appearance_time))
                        _remaining  = max(0.0, float(_target.expiration_time) - _now)
                        _elapsed    = max(0.0, _now - float(_target.appearance_time))
                        # Guard: if elapsed > total_dur something is wrong, clamp
                        _frac_elapsed = min(1.0, _elapsed / _total_dur)
                        _urgency    = 1.0 + 0.5 * _frac_elapsed
                        logger.debug(
                            f"Urgency: elapsed={_elapsed:.0f}s  total={_total_dur:.0f}s  "
                            f"frac={_frac_elapsed:.2f}  urgency={_urgency:.2f}"
                        )
                    except Exception:
                        _urgency = 1.0'''

NEW = '''                    # Urgency: HIGHER when fresh, LOWER near expiry.
                    # urgency = 1.0 + 0.5 * (remaining / total)  →  1.5 fresh, 1.0 expiring.
                    # This matches the spec and rewards prompt response to new events.
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
                        _urgency = 1.0'''


def apply(text: str) -> str:
    if OLD not in text:
        raise ValueError(
            "FIX-02: expected urgency pattern not found in env_alsat_dynamic.py.\n"
            "File may already be patched."
        )
    return text.replace(OLD, NEW, 1)