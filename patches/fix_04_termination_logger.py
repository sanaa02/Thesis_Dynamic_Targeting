"""
FIX 04 — env_alsat_dynamic.py  termination reason logger
=========================================================
At the end of every episode (term or trunc), print:
  - Why it ended: time limit vs early bsk_rl termination
  - Simulation time at termination
  - Number of unexpired active dynamic events remaining
  - Episode static/dynamic imaging counts and total reward
"""

OLD = '''        info['smdp_tau_s']       = tau
        info['smdp_n_sub']       = n_sub
        info['dynamic_metrics']  = self._mgr.get_metrics()

        tau_norm = tau / MAX_ACTION_DUR_S'''

NEW = '''        info['smdp_tau_s']       = tau
        info['smdp_n_sub']       = n_sub
        info['dynamic_metrics']  = self._mgr.get_metrics()

        # ── FIX-04: Termination reason logger ─────────────────────────────
        if term or trunc:
            try:
                _sat_log = self.env.unwrapped.satellites[0]
                _t_now   = float(_sat_log.simulator.sim_time)
                _t_limit = float(getattr(
                    self.env.unwrapped, 'time_limit',
                    getattr(self.env, 'time_limit', SIM_DURATION_S)
                ))
                _reason = "TIME_LIMIT" if trunc or abs(_t_now - _t_limit) < 120 else "EARLY_bsk_rl"
                _active_dyn = [
                    e for e in self._mgr._events
                    if not e.imaged and e.expiration_time > _t_now
                ]
                _n_static_img  = _sat_log._metrics.get('n_imaged', 0) - _sat_log._metrics.get('n_dyn_imaged', 0)
                _n_dyn_img     = _sat_log._metrics.get('n_dyn_imaged', 0)
                _total_rew     = _sat_log._metrics.get('total_reward', 0.0)
                # Count remaining static access windows
                _n_windows_left = 0
                try:
                    _opps = getattr(_sat_log, 'upcoming_opportunities', [])
                    for _opp in _opps:
                        _w = (_opp.get('window', [0,1]) if isinstance(_opp, dict)
                              else getattr(_opp, 'window', [0, 1]))
                        if _w[1] > _t_now:
                            _n_windows_left += 1
                except Exception:
                    pass
                print(
                    f"\n[EPISODE END] reason={_reason}  "
                    f"sim_time={_t_now:.0f}s / {_t_limit:.0f}s  "
                    f"({_t_now/_t_limit*100:.1f}%)  |  "
                    f"static_imaged={_n_static_img}  dyn_imaged={_n_dyn_img}  "
                    f"reward={_total_rew:.2f}  |  "
                    f"windows_left={_n_windows_left}  active_dyn={len(_active_dyn)}"
                )
            except Exception as _log_exc:
                print(f"[EPISODE END] term={term} trunc={trunc}  (logger err: {_log_exc})")
        # ──────────────────────────────────────────────────────────────────

        tau_norm = tau / MAX_ACTION_DUR_S'''


def apply(text: str) -> str:
    if OLD not in text:
        raise ValueError(
            "FIX-04: smdp_tau_s block not found in env_alsat_dynamic.py.\n"
            "File may already be patched."
        )
    return text.replace(OLD, NEW, 1)
