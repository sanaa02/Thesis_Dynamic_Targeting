"""
FIX 06 — env_alsat_dynamic.py  access window coverage check at reset
=====================================================================
After every reset(), print a one-time summary showing the first and last
access window time for each static target.  This makes it immediately
visible whether windows cover the full 48-hour episode or stop early
(e.g. after orbit 3), which is the leading cause of premature termination.

Output format (example):
  [WINDOWS] 20 targets  episode=172800s
    Algiers   : first=  1240s  last=169120s  count=14
    Oran      : first=  4100s  last=172100s  count=15
    ...
    WARNING: Blida has NO windows in this episode!
"""

OLD = '''        self._n_static_actions_ep = 0 
        return self._build_obs(obs, tau_norm=0.0), info'''

NEW = '''        self._n_static_actions_ep = 0

        # ── FIX-06: Access window coverage check ──────────────────────────
        try:
            _sat_wr = self.env.unwrapped.satellites[0]
            _opps   = getattr(_sat_wr, 'upcoming_opportunities', [])
            _tgts   = list(_sat_wr.scenario.targets)
            _dur    = float(getattr(
                self.env.unwrapped, 'time_limit',
                getattr(self.env, 'time_limit', SIM_DURATION_S)
            ))
            print(f"[WINDOWS] {len(_tgts)} targets  episode={_dur:.0f}s")
            _any_no_window = False
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
                if _wins:
                    _first = min(w[0] for w in _wins)
                    _last  = max(w[1] for w in _wins)
                    print(f"  {getattr(_tgt,'name',str(_tgt)):20s}: "
                          f"first={_first:7.0f}s  last={_last:7.0f}s  count={len(_wins)}")
                else:
                    _any_no_window = True
                    print(f"  {getattr(_tgt,'name',str(_tgt)):20s}: *** NO WINDOWS ***")
            if _any_no_window:
                print("[WINDOWS] WARNING: some targets have no windows — "
                      "check initial_generation_duration in make_dynamic_env()")
        except Exception as _wc_exc:
            logger.debug(f"[WINDOWS] coverage check failed: {_wc_exc}")
        # ──────────────────────────────────────────────────────────────────

        return self._build_obs(obs, tau_norm=0.0), info'''


def apply(text: str) -> str:
    if OLD not in text:
        raise ValueError(
            "FIX-06: reset tail block not found in env_alsat_dynamic.py.\n"
            "File may already be patched."
        )
    return text.replace(OLD, NEW, 1)
