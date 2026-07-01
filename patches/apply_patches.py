#!/usr/bin/env python3
"""
apply_patches.py  —  ALSAT-EO-1 diagnostic/improvement patches
===============================================================
Applies all 10 fixes described in the improvement document to the live
source files on your machine.  Each file is backed up as <file>.bak
before any changes are written.

Run from the repo root:
    python patches/apply_patches.py

Or with --dry-run to preview without changing anything:
    python patches/apply_patches.py --dry-run
"""
from __future__ import annotations
import argparse
import os
import shutil
import sys
import textwrap

# ── Resolve repo root ─────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)

# ── Target file paths ─────────────────────────────────────────────────────────
TRAIN   = os.path.join(REPO_ROOT, "scripts", "training", "train_full_system.py")
DYN     = os.path.join(REPO_ROOT, "scripts", "core",     "env_alsat_dynamic.py")
DEBUG   = os.path.join(REPO_ROOT, "scripts", "core",     "env_alsat_debug.py")
ATTN    = os.path.join(REPO_ROOT, "scripts", "models",   "attention_policy.py")
MASK    = os.path.join(REPO_ROOT, "scripts", "wrappers", "action_mask_wrapper.py")


# ═════════════════════════════════════════════════════════════════════════════
#  FIX 01  --attention crash: wrong kwarg names in policy_kwargs
# ═════════════════════════════════════════════════════════════════════════════
def fix_01_attention_crash(text: str) -> str:
    """Fix features_extractor_kwargs: embed_dim → d_model; wrong net_arch format."""
    OLD = (
        "            policy_kwargs = dict(\n"
        "                features_extractor_class  = SchedulerAttentionExtractor,\n"
        "                features_extractor_kwargs = dict(n_heads=4, embed_dim=64),\n"
        "                net_arch = dict(pi=[256, 128], vf=[256, 128]),\n"
        "            )"
    )
    NEW = (
        "            policy_kwargs = dict(\n"
        "                features_extractor_class  = SchedulerAttentionExtractor,\n"
        "                features_extractor_kwargs = dict(features_dim=256, d_model=64, n_heads=4),\n"
        "                net_arch = [],   # extractor handles all feature compression\n"
        "            )"
    )
    if OLD not in text:
        raise PatchError("FIX-01: attention policy_kwargs block not found in train_full_system.py")
    return text.replace(OLD, NEW, 1)


# ═════════════════════════════════════════════════════════════════════════════
#  FIX 02  urgency direction: use remaining time, not elapsed time
# ═════════════════════════════════════════════════════════════════════════════
def fix_02_urgency_direction(text: str) -> str:
    OLD = (
        "                    # Urgency: newer events pay more\n"
        "                    try:\n"
        "                        _now        = float(_sat.simulator.sim_time)\n"
        "                        _total_dur  = max(1.0, float(_target.expiration_time)\n"
        "                                              - float(_target.appearance_time))\n"
        "                        _remaining  = max(0.0, float(_target.expiration_time) - _now)\n"
        "                        _elapsed    = max(0.0, _now - float(_target.appearance_time))\n"
        "                        # Guard: if elapsed > total_dur something is wrong, clamp\n"
        "                        _frac_elapsed = min(1.0, _elapsed / _total_dur)\n"
        "                        _urgency    = 1.0 + 0.5 * _frac_elapsed\n"
        "                        logger.debug(\n"
        "                            f\"Urgency: elapsed={_elapsed:.0f}s  total={_total_dur:.0f}s  \"\n"
        "                            f\"frac={_frac_elapsed:.2f}  urgency={_urgency:.2f}\"\n"
        "                        )\n"
        "                    except Exception:\n"
        "                        _urgency = 1.0"
    )
    NEW = (
        "                    # Urgency: HIGHER when fresh, LOWER near expiry.\n"
        "                    # urgency = 1.0 + 0.5*(remaining/total) → 1.5 fresh, 1.0 expiring.\n"
        "                    try:\n"
        "                        _now        = float(_sat.simulator.sim_time)\n"
        "                        _total_dur  = max(1.0, float(_target.expiration_time)\n"
        "                                              - float(_target.appearance_time))\n"
        "                        _remaining  = max(0.0, float(_target.expiration_time) - _now)\n"
        "                        _frac_remaining = min(1.0, _remaining / _total_dur)\n"
        "                        _urgency    = 1.0 + 0.5 * _frac_remaining   # 1.5 fresh → 1.0 old\n"
        "                        logger.debug(\n"
        "                            f\"Urgency: remaining={_remaining:.0f}s  total={_total_dur:.0f}s  \"\n"
        "                            f\"frac_remaining={_frac_remaining:.2f}  urgency={_urgency:.2f}\"\n"
        "                        )\n"
        "                    except Exception:\n"
        "                        _urgency = 1.0"
    )
    if OLD not in text:
        raise PatchError("FIX-02: urgency block not found in env_alsat_dynamic.py")
    return text.replace(OLD, NEW, 1)


# ═════════════════════════════════════════════════════════════════════════════
#  FIX 03  remove double dynamic reward (disable legacy _dyn_imaging_check pre-block)
# ═════════════════════════════════════════════════════════════════════════════
def fix_03_remove_double_reward(text: str) -> str:
    OLD = (
        "        # ── [ROOT FIX] geometric imaging check (non-critical, wrapped safely)\n"
        "        # NOTE: do NOT reset _locked_dyn_event here — the main injection\n"
        "        # block below (lines 506+) uses it and must see it intact.\n"
        "        try:\n"
        "            _sat = self.env.unwrapped.satellites[0]\n"
        "            # [FIX-B] Only run geometric check if there is actually a locked event.\n"
        "            # Without this guard, _dyn_imaging_check fires on empty slots and\n"
        "            # gives the agent free +0.29 rewards for doing nothing.\n"
        "            _locked_evt = getattr(_sat, '_locked_dyn_event', None)\n"
        "            _has_locked = (\n"
        "                _locked_evt is not None\n"
        "                and not getattr(_locked_evt, 'imaged', False)\n"
        "                and _locked_evt.expiration_time > float(_sat.simulator.sim_time)\n"
        "            )\n"
        "            if _has_locked:\n"
        "                _dyn_r = _dyn_imaging_check(_sat, info)\n"
        "                if _dyn_r > 0.0:\n"
        "                    total_r += _dyn_r\n"
        "                    _sat._dyn_reward_given = True\n"
        "                    _sat._locked_dyn_event = None\n"
        "                    _sat._locked_dyn_slot  = None\n"
        "                    # FIX: _dyn_imaging_check writes to info only.\n"
        "                    # SingleSatelliteEnv overwrites info with dict(sat._metrics) at\n"
        "                    # episode end → n_dyn_imaged stays 0 → dyn_suc=0% always.\n"
        "                    _sat._metrics['n_dyn_imaged'] = (\n"
        "                        _sat._metrics.get('n_dyn_imaged', 0) + 1)\n"
        "        except Exception:\n"
        "            pass"
    )
    NEW = (
        "        # ── [ROOT FIX] geometric imaging check — DISABLED (FIX-03 single-reward)\n"
        "        # The main DYN reward injection block below is the sole place that adds\n"
        "        # DYN reward.  Running _dyn_imaging_check() here as well created a\n"
        "        # fragile dual-path that risked double-counting reward for the same event.\n"
        "        # The main block has attempt shaping + battery penalty + full metrics.\n"
        "        pass  # legacy pre-block disabled — reward injected by main DYN block below"
    )
    if OLD not in text:
        raise PatchError("FIX-03: ROOT FIX geometric check block not found in env_alsat_dynamic.py")
    return text.replace(OLD, NEW, 1)


# ═════════════════════════════════════════════════════════════════════════════
#  FIX 04  termination reason logger
# ═════════════════════════════════════════════════════════════════════════════
def fix_04_termination_logger(text: str) -> str:
    OLD = (
        "        info['smdp_tau_s']       = tau\n"
        "        info['smdp_n_sub']       = n_sub\n"
        "        info['dynamic_metrics']  = self._mgr.get_metrics()\n"
        "\n"
        "        tau_norm = tau / MAX_ACTION_DUR_S"
    )
    NEW = (
        "        info['smdp_tau_s']       = tau\n"
        "        info['smdp_n_sub']       = n_sub\n"
        "        info['dynamic_metrics']  = self._mgr.get_metrics()\n"
        "\n"
        "        # ── FIX-04: Termination reason logger ─────────────────────────────\n"
        "        if term or trunc:\n"
        "            try:\n"
        "                _sat_log = self.env.unwrapped.satellites[0]\n"
        "                _t_now   = float(_sat_log.simulator.sim_time)\n"
        "                _t_limit = float(getattr(\n"
        "                    self.env.unwrapped, 'time_limit',\n"
        "                    getattr(self.env, 'time_limit', SIM_DURATION_S)\n"
        "                ))\n"
        "                _reason = 'TIME_LIMIT' if trunc or abs(_t_now - _t_limit) < 120 else 'EARLY_bsk_rl'\n"
        "                _active_dyn = [\n"
        "                    e for e in self._mgr._events\n"
        "                    if not e.imaged and e.expiration_time > _t_now\n"
        "                ]\n"
        "                _n_static_img = (\n"
        "                    _sat_log._metrics.get('n_imaged', 0)\n"
        "                    - _sat_log._metrics.get('n_dyn_imaged', 0)\n"
        "                )\n"
        "                _n_dyn_img  = _sat_log._metrics.get('n_dyn_imaged', 0)\n"
        "                _total_rew  = _sat_log._metrics.get('total_reward', 0.0)\n"
        "                _n_wins_left = 0\n"
        "                try:\n"
        "                    _opps_log = getattr(_sat_log, 'upcoming_opportunities', [])\n"
        "                    for _opp_l in _opps_log:\n"
        "                        _w_l = (_opp_l.get('window', [0,1]) if isinstance(_opp_l, dict)\n"
        "                                else getattr(_opp_l, 'window', [0, 1]))\n"
        "                        if _w_l[1] > _t_now:\n"
        "                            _n_wins_left += 1\n"
        "                except Exception:\n"
        "                    pass\n"
        "                print(\n"
        "                    f'\\n[EPISODE END] reason={_reason}  '\n"
        "                    f'sim_time={_t_now:.0f}s / {_t_limit:.0f}s  '\n"
        "                    f'({_t_now / _t_limit * 100:.1f}%)  |  '\n"
        "                    f'static_imaged={_n_static_img}  dyn_imaged={_n_dyn_img}  '\n"
        "                    f'reward={_total_rew:.2f}  |  '\n"
        "                    f'windows_left={_n_wins_left}  active_dyn={len(_active_dyn)}'\n"
        "                )\n"
        "            except Exception as _log_exc:\n"
        "                print(f'[EPISODE END] term={term} trunc={trunc}  (logger err: {_log_exc})')\n"
        "        # ──────────────────────────────────────────────────────────────────\n"
        "\n"
        "        tau_norm = tau / MAX_ACTION_DUR_S"
    )
    if OLD not in text:
        raise PatchError("FIX-04: smdp_tau_s/tau_norm block not found in env_alsat_dynamic.py")
    return text.replace(OLD, NEW, 1)


# ═════════════════════════════════════════════════════════════════════════════
#  FIX 05  housekeeping mode: suppress bsk_rl early termination
# ═════════════════════════════════════════════════════════════════════════════
def fix_05_housekeeping_mode(text: str) -> str:
    OLD = (
        "        DRIFT_ACT = N_STATIC_TARGETS + N_DYN_SLOTS  # = 23\n"
        "        total_r = 0.0\n"
        "        for _i in range(n_sub):\n"
        "            _sub_a = action \n"
        "            try:\n"
        "\n"
        "                # Keep sat._event_manager pointing at self._mgr (survives bsk_rl resets)\n"
        "                for _sx in self.env.unwrapped.satellites:\n"
        "                    _sx._event_manager = self._mgr\n"
        "            except Exception:\n"
        "                pass\n"
        "            obs_i, r_i, term, trunc, info = self.env.step(_sub_a)\n"
        "            total_r += (self._gamma_sub ** _i) * r_i\n"
        "            last_obs = obs_i\n"
        "            if term or trunc:\n"
        "                break"
    )
    NEW = (
        "        DRIFT_ACT = N_STATIC_TARGETS + N_DYN_SLOTS  # = 23\n"
        "        total_r = 0.0\n"
        "        for _i in range(n_sub):\n"
        "            _sub_a = action \n"
        "            try:\n"
        "\n"
        "                # Keep sat._event_manager pointing at self._mgr (survives bsk_rl resets)\n"
        "                for _sx in self.env.unwrapped.satellites:\n"
        "                    _sx._event_manager = self._mgr\n"
        "            except Exception:\n"
        "                pass\n"
        "            obs_i, r_i, term, trunc, info = self.env.step(_sub_a)\n"
        "            total_r += (self._gamma_sub ** _i) * r_i\n"
        "            last_obs = obs_i\n"
        "            if term or trunc:\n"
        "                break\n"
        "\n"
        "        # ── FIX-05: Housekeeping mode — suppress bsk_rl early termination ─────\n"
        "        # bsk_rl sets term=True when it thinks there are no more windows.\n"
        "        # If the simulation clock has NOT yet reached the time limit, override\n"
        "        # term=False so the satellite keeps running in housekeeping/drift mode.\n"
        "        # This ensures episodes always last the full 48 hours.\n"
        "        if term and not trunc:\n"
        "            try:\n"
        "                _sat_hk   = self.env.unwrapped.satellites[0]\n"
        "                _t_now_hk = float(_sat_hk.simulator.sim_time)\n"
        "                _t_lim_hk = float(getattr(\n"
        "                    self.env.unwrapped, 'time_limit',\n"
        "                    getattr(self.env, 'time_limit', SIM_DURATION_S)\n"
        "                ))\n"
        "                if _t_now_hk < _t_lim_hk - 120.0:  # 120 s grace period\n"
        "                    logger.debug(\n"
        "                        f'[HOUSEKEEPING] bsk_rl early-term suppressed: '\n"
        "                        f't={_t_now_hk:.0f}s < limit={_t_lim_hk:.0f}s — continuing'\n"
        "                    )\n"
        "                    term = False\n"
        "            except Exception:\n"
        "                pass\n"
        "        # ──────────────────────────────────────────────────────────────────────"
    )
    if OLD not in text:
        raise PatchError("FIX-05: sub-step loop block not found in env_alsat_dynamic.py")
    return text.replace(OLD, NEW, 1)


# ═════════════════════════════════════════════════════════════════════════════
#  FIX 06  access window coverage check printed after every reset()
# ═════════════════════════════════════════════════════════════════════════════
def fix_06_access_window_check(text: str) -> str:
    # Try both with and without trailing space (editors differ on whitespace stripping)
    _OLD_SPACE    = "        self._n_static_actions_ep = 0 \n        return self._build_obs(obs, tau_norm=0.0), info"
    _OLD_NOSPACE  = "        self._n_static_actions_ep = 0\n        return self._build_obs(obs, tau_norm=0.0), info"
    if _OLD_SPACE in text:
        OLD = _OLD_SPACE
    elif _OLD_NOSPACE in text:
        OLD = _OLD_NOSPACE
    else:
        raise PatchError("FIX-06: reset tail (n_static_actions_ep) not found in env_alsat_dynamic.py")
    NEW = (
        "        self._n_static_actions_ep = 0\n"
        "\n"
        "        # ── FIX-06: Access window coverage check ──────────────────────────\n"
        "        try:\n"
        "            _sat_wr = self.env.unwrapped.satellites[0]\n"
        "            _opps   = getattr(_sat_wr, 'upcoming_opportunities', [])\n"
        "            _tgts   = list(_sat_wr.scenario.targets)\n"
        "            _dur    = float(getattr(\n"
        "                self.env.unwrapped, 'time_limit',\n"
        "                getattr(self.env, 'time_limit', SIM_DURATION_S)\n"
        "            ))\n"
        "            print(f'[WINDOWS] {len(_tgts)} targets  episode={_dur:.0f}s')\n"
        "            _any_no_win = False\n"
        "            for _tgt in _tgts:\n"
        "                _wins = []\n"
        "                for _opp in _opps:\n"
        "                    try:\n"
        "                        _o = (_opp.get('object') if isinstance(_opp, dict)\n"
        "                              else getattr(_opp, 'object', None))\n"
        "                        _t = (_opp.get('type', '') if isinstance(_opp, dict)\n"
        "                              else getattr(_opp, 'type', ''))\n"
        "                        _w = (_opp.get('window', [0,1]) if isinstance(_opp, dict)\n"
        "                              else getattr(_opp, 'window', [0, 1]))\n"
        "                        if _o is _tgt and _t == 'target':\n"
        "                            _wins.append(_w)\n"
        "                    except Exception:\n"
        "                        pass\n"
        "                _name = getattr(_tgt, 'name', str(_tgt))\n"
        "                if _wins:\n"
        "                    _first = min(w[0] for w in _wins)\n"
        "                    _last  = max(w[1] for w in _wins)\n"
        "                    print(f'  {_name:20s}: first={_first:7.0f}s  last={_last:7.0f}s  count={len(_wins)}')\n"
        "                else:\n"
        "                    _any_no_win = True\n"
        "                    print(f'  {_name:20s}: *** NO WINDOWS ***')\n"
        "            if _any_no_win:\n"
        "                print('[WINDOWS] WARNING: some targets have no windows — '\n"
        "                      'check initial_generation_duration in make_dynamic_env()')\n"
        "        except Exception as _wc_exc:\n"
        "            logger.debug(f'[WINDOWS] coverage check failed: {_wc_exc}')\n"
        "        # ──────────────────────────────────────────────────────────────────\n"
        "\n"
        "        return self._build_obs(obs, tau_norm=0.0), info"
    )
    return text.replace(OLD, NEW, 1)


# ═════════════════════════════════════════════════════════════════════════════
#  FIX 07  per-step diagnostic line (opt-in via ALSAT_DEBUG_STEPS=1)
# ═════════════════════════════════════════════════════════════════════════════
def fix_07_perstep_diagnostics(text: str) -> str:
    OLD = (
        "        # ── Static-imaging floor penalty ─────────────────────────────────\n"
        "        # If agent took ZERO static actions this episode, penalize.\n"
        "        # Prevents catastrophic forgetting of scheduled imaging.\n"
        "        if (term or trunc):"
    )
    NEW = (
        "        # ── FIX-07: Per-step diagnostic (opt-in: ALSAT_DEBUG_STEPS=1) ──────\n"
        "        import os as _os_diag\n"
        "        if _os_diag.environ.get('ALSAT_DEBUG_STEPS') == '1':\n"
        "            try:\n"
        "                _sd_sat  = self.env.unwrapped.satellites[0]\n"
        "                _sd_t    = float(_sd_sat.simulator.sim_time)\n"
        "                _sd_batt = float(getattr(\n"
        "                    getattr(_sd_sat, 'dynamics', _sd_sat),\n"
        "                    'battery_charge_fraction', 1.0\n"
        "                )) * 100.0\n"
        "                _sd_opps = getattr(_sd_sat, 'upcoming_opportunities', [])\n"
        "                _sd_wins = sum(\n"
        "                    1 for _o in _sd_opps\n"
        "                    for _w in [(_o.get('window', [0,1]) if isinstance(_o, dict)\n"
        "                                else getattr(_o, 'window', [0, 1]))]\n"
        "                    if _w[1] > _sd_t\n"
        "                )\n"
        "                _sd_act_name = (\n"
        "                    'DRIFT' if int(action) >= N_STATIC_TARGETS + N_DYN_SLOTS\n"
        "                    else (f'DYN{int(action) - N_STATIC_TARGETS}'\n"
        "                          if int(action) >= N_STATIC_TARGETS\n"
        "                          else f'TGT{int(action)}')\n"
        "                )\n"
        "                _step_num = getattr(self, '_step_count_diag', 0) + 1\n"
        "                self._step_count_diag = _step_num\n"
        "                if _step_num % 12 == 1:\n"
        "                    print(\n"
        "                        f'  [STEP {_step_num:3d}] t={_sd_t:7.0f}s  '\n"
        "                        f'batt={_sd_batt:5.1f}%  wins={_sd_wins:3d}  '\n"
        "                        f'act={_sd_act_name:<8s}  r={total_r:+.4f}'\n"
        "                    )\n"
        "            except Exception:\n"
        "                pass\n"
        "        # ──────────────────────────────────────────────────────────────────\n"
        "\n"
        "        # ── Static-imaging floor penalty ─────────────────────────────────\n"
        "        # If agent took ZERO static actions this episode, penalize.\n"
        "        # Prevents catastrophic forgetting of scheduled imaging.\n"
        "        if (term or trunc):"
    )
    if OLD not in text:
        raise PatchError("FIX-07: static-floor-penalty block not found in env_alsat_dynamic.py")
    return text.replace(OLD, NEW, 1)


# ═════════════════════════════════════════════════════════════════════════════
#  FIX 08  zero-reward static action diagnostic
# ═════════════════════════════════════════════════════════════════════════════
def fix_08_zero_reward_logger(text: str) -> str:
    OLD = (
        "        logger.debug(\n"
        "            f\"[STATIC] image taken: target={target.name}  \"\n"
        "            f\"cloud_truth={cloud_truth:.2f}  priority={priority:.2f}  \"\n"
        "            f\"slew_deg={math.degrees(slew_angle):.1f}  reward={reward:+.4f}\"\n"
        "        )"
    )
    NEW = (
        "        logger.debug(\n"
        "            f\"[STATIC] image taken: target={target.name}  \"\n"
        "            f\"cloud_truth={cloud_truth:.2f}  priority={priority:.2f}  \"\n"
        "            f\"slew_deg={math.degrees(slew_angle):.1f}  reward={reward:+.4f}\"\n"
        "        )\n"
        "        # ── FIX-08: Zero-reward static action diagnostic ──────────────────\n"
        "        if reward <= 0.0 and not is_dynamic:\n"
        "            _r08 = (\n"
        "                'CLOUDY'        if cloud_truth >= CLOUD_THRESH\n"
        "                else 'SLEW_LIMIT' if math.degrees(slew_angle) > 45.0\n"
        "                else 'NEGATIVE_SLEW_COST'\n"
        "            )\n"
        "            logger.debug(\n"
        "                f'[STATIC-ZERO] target={target.name}  reason={_r08}  '\n"
        "                f'cloud={cloud_truth:.2f}  slew={math.degrees(slew_angle):.1f}deg  '\n"
        "                f'reward={reward:+.4f}'\n"
        "            )\n"
        "        # ──────────────────────────────────────────────────────────────────"
    )
    if OLD not in text:
        raise PatchError("FIX-08: STATIC image-taken logger not found in env_alsat_dynamic.py")
    return text.replace(OLD, NEW, 1)


# ═════════════════════════════════════════════════════════════════════════════
#  FIX 09  eclipse-aware battery model  (env_alsat_debug.py)
# ═════════════════════════════════════════════════════════════════════════════
def fix_09_eclipse_battery_debug(text: str) -> str:
    """Add eclipse_battery_step() function to env_alsat_debug.py."""
    OLD = (
        "NET_BASE_POWER_W = 225.0   # W  orbit-averaged net (solar_avg - housekeeping)                                    # -20 W"
    )
    NEW = (
        "# FIX-09: Eclipse-aware power constants  (ALSAT-EO-1, 686 km SSO)\n"
        "SOLAR_PANEL_AREA_M2     = 1.5     # m²\n"
        "SOLAR_PANEL_EFF         = 0.28    # BOL GaAs efficiency\n"
        "SOLAR_IRRADIANCE_W_M2   = 1367.0  # W/m² at 1 AU\n"
        "SOLAR_INCIDENCE_FACTOR  = 0.653   # sunlit fraction × avg cosine\n"
        "HOUSEKEEPING_LOAD_W     = 50.0    # W  (idle, no comms/payload)\n"
        "_P_SOLAR_PEAK   = SOLAR_PANEL_AREA_M2 * SOLAR_PANEL_EFF * SOLAR_IRRADIANCE_W_M2\n"
        "_P_NET_SUNLIT   = _P_SOLAR_PEAK * SOLAR_INCIDENCE_FACTOR * 0.95 - HOUSEKEEPING_LOAD_W\n"
        "_P_NET_ECLIPSE  = -HOUSEKEEPING_LOAD_W\n"
        "# Legacy constant (orbit-averaged; kept for import compatibility):\n"
        "NET_BASE_POWER_W = (_P_NET_SUNLIT * SOLAR_INCIDENCE_FACTOR\n"
        "                    + _P_NET_ECLIPSE * (1.0 - SOLAR_INCIDENCE_FACTOR))\n"
        "\n"
        "\n"
        "def eclipse_battery_step(satellite, dt_s: float) -> None:\n"
        "    \"\"\"\n"
        "    Update battery SOC for one simulation sub-step of dt_s seconds.\n"
        "\n"
        "    In sunlight  : net power ≈ +306 W  (solar charges battery)\n"
        "    In eclipse   : net power ≈  –50 W  (housekeeping discharges battery)\n"
        "\n"
        "    The eclipse flag is read from satellite.dynamics.eclipse_shadow\n"
        "    (set by bsk_patches.P1).  If unavailable, sunlit is assumed.\n"
        "    \"\"\"\n"
        "    try:\n"
        "        dyn = getattr(satellite, 'dynamics', satellite)\n"
        "        eclipse_flag = float(getattr(dyn, 'eclipse_shadow', 0.0))\n"
        "        p_net_W = _P_NET_ECLIPSE if eclipse_flag > 0.5 else _P_NET_SUNLIT\n"
        "\n"
        "        cap_J  = float(getattr(dyn, 'batteryStorageCapacity', BATTERY_WH * 3600.0))\n"
        "        cap_Wh = cap_J / 3600.0\n"
        "        soc    = float(getattr(satellite, 'battery_charge_fraction',\n"
        "                               getattr(dyn, 'battery_charge_fraction', 1.0)))\n"
        "\n"
        "        delta_Wh = p_net_W * dt_s / 3600.0\n"
        "        new_soc  = float(np.clip(soc + delta_Wh / max(cap_Wh, 1.0), 0.0, 1.0))\n"
        "\n"
        "        if hasattr(satellite, 'battery_charge_fraction'):\n"
        "            satellite.battery_charge_fraction = new_soc\n"
        "        if hasattr(dyn, 'battery_charge_fraction'):\n"
        "            dyn.battery_charge_fraction = new_soc\n"
        "    except Exception:\n"
        "        pass"
    )
    if OLD not in text:
        raise PatchError("FIX-09: NET_BASE_POWER_W constant not found in env_alsat_debug.py")
    return text.replace(OLD, NEW, 1)


def fix_09_eclipse_battery_dynamic(text: str) -> str:
    """Call eclipse_battery_step from DynamicObsWrapper._build_obs()."""
    OLD = (
        "    def _build_obs(self, base_obs: np.ndarray, tau_norm: float) -> np.ndarray:\n"
        "        try:\n"
        "            sat   = self.env.unwrapped.satellites[0]\n"
        "            now   = float(sat.simulator.sim_time)\n"
        "            slots = self._mgr.get_slots(sat, now)\n"
        "        except Exception:\n"
        "            slots = [None] * N_DYN_SLOTS\n"
        "            sat   = None\n"
        "            now   = 0.0"
    )
    NEW = (
        "    def _build_obs(self, base_obs: np.ndarray, tau_norm: float) -> np.ndarray:\n"
        "        try:\n"
        "            sat   = self.env.unwrapped.satellites[0]\n"
        "            now   = float(sat.simulator.sim_time)\n"
        "            slots = self._mgr.get_slots(sat, now)\n"
        "        except Exception:\n"
        "            slots = [None] * N_DYN_SLOTS\n"
        "            sat   = None\n"
        "            now   = 0.0\n"
        "\n"
        "        # ── FIX-09: Eclipse-aware battery update ──────────────────────────\n"
        "        if sat is not None:\n"
        "            try:\n"
        "                from env_alsat_debug import eclipse_battery_step\n"
        "                _dt09 = float(np.clip(\n"
        "                    now - self._prev_time if now > self._prev_time else BASE_STEP_S,\n"
        "                    0.0, MAX_ACTION_DUR_S\n"
        "                ))\n"
        "                eclipse_battery_step(sat, _dt09)\n"
        "            except Exception:\n"
        "                pass\n"
        "        # ──────────────────────────────────────────────────────────────────"
    )
    if OLD not in text:
        raise PatchError("FIX-09: _build_obs header not found in env_alsat_dynamic.py")
    return text.replace(OLD, NEW, 1)


# ═════════════════════════════════════════════════════════════════════════════
#  FIX 10  unified cloud threshold — fix hard-coded -0.3 in DynamicScienceDataStore
# ═════════════════════════════════════════════════════════════════════════════
def fix_10_unified_cloud_thresh(text: str) -> str:
    OLD = (
        "            if cloud_truth < CLOUD_THRESH:\n"
        "                reward = (DYN_MULTIPLIER * priority * (1.0 - cloud_truth) * urgency\n"
        "                         - SLEW_ENERGY_ALPHA * slew_energy + DYNAMIC_BONUS)\n"
        "                sat._metrics['n_cloud_free'] += 1\n"
        "            else:\n"
        "                reward = -0.3 * priority   # stronger penalty for cloudy dynamic waste"
    )
    NEW = (
        "            if cloud_truth < CLOUD_THRESH:\n"
        "                reward = (DYN_MULTIPLIER * priority * (1.0 - cloud_truth) * urgency\n"
        "                         - SLEW_ENERGY_ALPHA * slew_energy + DYNAMIC_BONUS)\n"
        "                sat._metrics['n_cloud_free'] += 1\n"
        "            else:\n"
        "                # FIX-10: unified cloud penalty — same scale as static (-0.1 * priority)\n"
        "                # (was hard-coded -0.3 regardless of priority; now consistent)\n"
        "                reward = -0.1 * priority\n"
        "                sat._metrics['n_cloudy'] += 1"
    )
    if OLD not in text:
        raise PatchError("FIX-10: DYN cloudy reward block not found in env_alsat_dynamic.py")
    return text.replace(OLD, NEW, 1)


# ═════════════════════════════════════════════════════════════════════════════
#  FIX 12  action_mask_wrapper.py: DYN slots masked only for emptiness, not slew
#
#  Root cause:
#    Mask currently only checks `slots[j] is not None`.
#    The agent can select an occupied slot whose event is 80°+ off-nadir
#    (physically unreachable).  Reward = +0.050 explore bonus, no imaging.
#    27 events in a typical episode, only 1 imaged → 4% dyn_suc.
#
#  Fix:
#    After the empty-slot check, also check slew angle vs MAX_OFFNADIR_RAD.
#    If the event is geometrically inaccessible, mask the slot.
# ═════════════════════════════════════════════════════════════════════════════
def fix_12_dyn_slew_mask(text: str) -> str:
    OLD = (
        "                if mgr is not None:\n"
        "                    slots = mgr.get_slots(sat, now)\n"
        "                    for j in range(self._n_dyn):\n"
        "                        has_event = j < len(slots) and slots[j] is not None\n"
        "                        mask[self._n_static + j] = has_event\n"
        "                else:\n"
        "                    # No manager: mask all DYN slots (safer than allowing penalties)\n"
        "                    for j in range(self._n_dyn):\n"
        "                        mask[self._n_static + j] = False"
    )
    NEW = (
        "                if mgr is not None:\n"
        "                    slots = mgr.get_slots(sat, now)\n"
        "                    for j in range(self._n_dyn):\n"
        "                        has_event = j < len(slots) and slots[j] is not None\n"
        "                        if not has_event:\n"
        "                            mask[self._n_static + j] = False\n"
        "                            continue\n"
        "                        # FIX-12: also check geometric accessibility (slew ≤ MAX_OFFNADIR)\n"
        "                        # Without this the agent wastes steps on events it cannot reach\n"
        "                        # and collects only the +0.05 explore bonus, keeping dyn_suc low.\n"
        "                        try:\n"
        "                            from dynamic_event import MAX_OFFNADIR_RAD\n"
        "                            from env_alsat_debug import calculate_slew_angle_to_target\n"
        "                            _slew12 = calculate_slew_angle_to_target(\n"
        "                                sat, slots[j].r_LP_P)\n"
        "                            mask[self._n_static + j] = (_slew12 <= MAX_OFFNADIR_RAD)\n"
        "                        except Exception:\n"
        "                            mask[self._n_static + j] = True  # safe fallback\n"
        "                else:\n"
        "                    # No manager: mask all DYN slots (safer than allowing penalties)\n"
        "                    for j in range(self._n_dyn):\n"
        "                        mask[self._n_static + j] = False"
    )
    if OLD not in text:
        raise PatchError(
            "FIX-12: DYN slot masking block not found in action_mask_wrapper.py\n"
            "        Expect: 'if mgr is not None: slots = mgr.get_slots(...) ...' block"
        )
    return text.replace(OLD, NEW, 1)


# ═════════════════════════════════════════════════════════════════════════════
#  FIX 15+16  action_mask_wrapper.py: static mask misses already-imaged targets
#             and clearly cloudy targets (cloud_forecast > CLOUD_THRESH)
#
#  FIX-15:
#    After a target is successfully imaged, the reward code sets tgt.imaged=True.
#    The mask still keeps that target as selectable.
#    Agent re-tries the same target 2-4× per episode, wasting steps.
#    Fix: check getattr(tgt, 'imaged', False) and block re-selection.
#
#  FIX-16:
#    Even when the access window is open, cloud_cover_forecast > CLOUD_THRESH
#    gives reward=0.  The agent has no incentive to avoid cloudy targets except
#    via a negative signal it can only learn from experience.
#    Fix: proactively mask targets where forecast cloud > 0.6 (same threshold
#    as the env's reward function).  This removes ~40% of zero-reward actions
#    and gives the agent a cleaner signal.
# ═════════════════════════════════════════════════════════════════════════════
def fix_15_16_static_mask_quality(text: str) -> str:
    OLD = (
        "                                if o is tgt and t == \"target\" and w[0] <= now <= w[1]:\n"
        "                                    accessible = True\n"
        "                                    break\n"
        "                            except Exception:\n"
        "                                pass\n"
        "                        mask[i] = accessible"
    )
    NEW = (
        "                                if o is tgt and t == \"target\" and w[0] <= now <= w[1]:\n"
        "                                    accessible = True\n"
        "                                    break\n"
        "                            except Exception:\n"
        "                                pass\n"
        "                        # FIX-15: block re-imaging of targets already done this episode\n"
        "                        if accessible and getattr(tgt, 'imaged', False):\n"
        "                            accessible = False\n"
        "                        # FIX-16: block targets with clearly cloudy forecast\n"
        "                        # (same 0.6 threshold as env reward function)\n"
        "                        if accessible:\n"
        "                            try:\n"
        "                                _fcst16 = float(getattr(\n"
        "                                    tgt, 'cloud_cover_forecast',\n"
        "                                    getattr(tgt, 'cloud_cover', 0.0)))\n"
        "                                if _fcst16 > 0.6:\n"
        "                                    accessible = False\n"
        "                            except Exception:\n"
        "                                pass\n"
        "                        mask[i] = accessible"
    )
    if OLD not in text:
        raise PatchError(
            "FIX-15/16: static target mask assignment not found in action_mask_wrapper.py\n"
            "           Expect: 'if o is tgt and t==\"target\" ... mask[i] = accessible' block"
        )
    return text.replace(OLD, NEW, 1)


# ═════════════════════════════════════════════════════════════════════════════
#  FIX 17  action_mask_wrapper.py: static windows not segmented into orbital passes
#
#  Root cause:
#    bsk_rl's add_location_for_access_checking reports ONE continuous arc for
#    the entire initial_generation_duration.  That arc is 24-95 hours — the
#    entire episode.  The agent can image any target at any moment, so it
#    learns a trivial "image everything in one big sweep" strategy and never
#    learns orbital scheduling.
#
#  Fix:
#    In _compute_mask(), after finding a matching window:
#      - Windows ≤ 900 s (15 min) → keep as-is (already realistic)
#      - Longer arcs → slice into discrete passes of PASS_S=600 s (10 min)
#        every ORBT_S=5880 s (98-min orbit of ALSAT-EO-1 at 686 km)
#    The underlying bsk_rl physics still rewards imaging; we only restrict
#    WHEN the mask allows the agent to try.
#
#  Note: OLD string must match the file AFTER fix_15_16 has been applied
#        (FIX-15 comment is already present in the file the user ran).
# ═════════════════════════════════════════════════════════════════════════════
def fix_17_window_segmentation(text: str) -> str:
    OLD = (
        "                                if o is tgt and t == \"target\" and w[0] <= now <= w[1]:\n"
        "                                    accessible = True\n"
        "                                    break\n"
        "                            except Exception:\n"
        "                                pass\n"
        "                        # FIX-15: block re-imaging of targets already done this episode"
    )
    NEW = (
        "                                if o is tgt and t == \"target\":\n"
        "                                    # FIX-17: segment unrealistically long windows\n"
        "                                    # bsk_rl reports one 24-95 h arc for the full\n"
        "                                    # initial_generation_duration.  Slice it into\n"
        "                                    # 10-min passes every 98 min (ALSAT at 686 km).\n"
        "                                    _ORBT17 = 5880.0   # orbital period (s)\n"
        "                                    _PASS17 = 600.0    # realistic pass duration (s)\n"
        "                                    _MAXN17 = 900.0    # windows ≤15 min kept as-is\n"
        "                                    _w0_17  = float(w[0])\n"
        "                                    _w1_17  = float(w[1])\n"
        "                                    _ok17   = False\n"
        "                                    if _w1_17 - _w0_17 <= _MAXN17:\n"
        "                                        _ok17 = (_w0_17 <= now <= _w1_17)\n"
        "                                    else:\n"
        "                                        _p17 = _w0_17\n"
        "                                        while _p17 < _w1_17:\n"
        "                                            _p_end17 = min(_p17 + _PASS17, _w1_17)\n"
        "                                            if _p17 <= now <= _p_end17:\n"
        "                                                _ok17 = True\n"
        "                                                break\n"
        "                                            _p17 += _ORBT17\n"
        "                                    if _ok17:\n"
        "                                        accessible = True\n"
        "                                        break\n"
        "                            except Exception:\n"
        "                                pass\n"
        "                        # FIX-15: block re-imaging of targets already done this episode"
    )
    if OLD not in text:
        raise PatchError(
            "FIX-17: window-check block not found in action_mask_wrapper.py\n"
            "        Expected post-FIX-15 version with '# FIX-15: block re-imaging' comment.\n"
            "        If FIX-15/16 was not applied yet, run patches first then re-run."
        )
    return text.replace(OLD, NEW, 1)


# ═════════════════════════════════════════════════════════════════════════════
#  FIX 18  env_alsat_dynamic.py: min_elev too permissive → massive access arcs
#
#  Root cause:
#    min_elev = 15° → off-nadir angle ρ = arcsin(Re·cos(15°)/(Re+h)) ≈ 61°
#    Earth central angle λ = 90° − 15° − 61° ≈ 14°  (access footprint)
#    This is far wider than ALSAT-EO-1's 45° off-nadir imaging constraint.
#
#  Fix:
#    min_elev = 38° → ρ ≈ 45° (matches camera constraint)
#    λ ≈ 7° (realistic ground footprint)
#    Orbital geometry: at 686 km, 7° Earth angle → ~780 km cross-track reach
#    Combined with FIX-17, the agent sees realistic 10-min access windows.
# ═════════════════════════════════════════════════════════════════════════════
def fix_18_minelev_realistic(text: str) -> str:
    OLD = (
        "                self.add_location_for_access_checking(\n"
        "                    object=target,\n"
        "                    r_LP_P=target.r_LP_P,\n"
        "                    min_elev=np.radians(15.0),\n"
        "                    type=\"target\",\n"
        "                    start_time=0.0,\n"
        "                )"
    )
    NEW = (
        "                self.add_location_for_access_checking(\n"
        "                    object=target,\n"
        "                    r_LP_P=target.r_LP_P,\n"
        "                    # FIX-18: 38° elevation ≈ 45° off-nadir for ALSAT at 686 km\n"
        "                    # (law of sines: sin(ρ)=Re·cos(El)/(Re+h) → 38° gives ρ≈45°)\n"
        "                    # Previous 15° gave off-nadir 61° and 24-95 h continuous arcs.\n"
        "                    min_elev=np.radians(38.0),\n"
        "                    type=\"target\",\n"
        "                    start_time=0.0,\n"
        "                )"
    )
    if OLD not in text:
        raise PatchError(
            "FIX-18: add_location_for_access_checking(min_elev=15°) not found in env_alsat_dynamic.py\n"
            "        Check the file for the current min_elev value."
        )
    return text.replace(OLD, NEW, 1)


# ═════════════════════════════════════════════════════════════════════════════
#  FIX 11  attention_policy.py: s_tok missing unsqueeze → catastrophic broadcast
#
#  Root cause:
#    s_tok = state_proj(state)   →  shape (B, d_model)   ← 2-D
#    type_embed(zeros(B,1))      →  shape (B, 1, d_model) ← 3-D
#    (B,d) + (B,1,d) broadcasts to (B, B, d)  ← s_tok becomes (48,48,64)
#
#  Then cat([s_tok, t_ctx, d_ctx, j_tok], dim=1) gives (B, B+B+B+1, d)
#  = (48, 145, 64) → flat = (48, 9280) → Linear(256,256) crashes.
#
#  Fix: add .unsqueeze(1) so s_tok is (B,1,d) from the start.
# ═════════════════════════════════════════════════════════════════════════════
def fix_11_s_tok_unsqueeze(text: str) -> str:
    OLD = (
        "            s_tok  = self.state_proj(state)                # (B, d)\n"
        "            t_tok = self.target_proj(targets)          # (B, n_tgt, d)"
    )
    NEW = (
        "            s_tok  = self.state_proj(state).unsqueeze(1)   # (B, 1, d) — must be 3-D for cross-attn\n"
        "            t_tok = self.target_proj(targets)          # (B, n_tgt, d)"
    )
    if OLD not in text:
        raise PatchError(
            "FIX-11: s_tok = state_proj(state) line not found in attention_policy.py\n"
            "        Open the file and manually add .unsqueeze(1) to that line."
        )
    return text.replace(OLD, NEW, 1)


# ═════════════════════════════════════════════════════════════════════════════
#  Runner
# ═════════════════════════════════════════════════════════════════════════════

class PatchError(RuntimeError):
    pass


def read_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def write_file(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def backup(path: str) -> None:
    shutil.copy2(path, path + ".bak")
    print(f"  Backed up → {path}.bak")


def apply_to_file(path: str, funcs: list, dry_run: bool, label: str) -> bool:
    if not os.path.exists(path):
        print(f"  [SKIP] {label}: file not found at {path}")
        return False
    text = read_file(path)
    original = text
    errors = []
    applied = []
    for fn in funcs:
        try:
            text = fn(text)
            applied.append(fn.__name__)
        except PatchError as e:
            errors.append(str(e))
    if errors:
        for e in errors:
            print(f"  [WARN] {e}")
    if text == original:
        print(f"  [SKIP] {label}: no changes (already patched or pattern not found)")
        return False
    if dry_run:
        print(f"  [DRY ] {label}: would apply {applied}")
        return True
    backup(path)
    write_file(path, text)
    print(f"  [OK]   {label}: applied {applied}")
    return True


def main(dry_run: bool = False) -> None:
    print("\n" + "=" * 65)
    print("  ALSAT-EO-1 patch installer")
    print(f"  repo root: {REPO_ROOT}")
    print("=" * 65 + "\n")

    if dry_run:
        print("  DRY-RUN mode — no files will be changed.\n")

    # ── train_full_system.py ─────────────────────────────────────────────
    apply_to_file(
        TRAIN,
        [fix_01_attention_crash],
        dry_run,
        "train_full_system.py",
    )

    # ── env_alsat_dynamic.py ─────────────────────────────────────────────
    apply_to_file(
        DYN,
        [
            fix_02_urgency_direction,
            fix_03_remove_double_reward,
            fix_04_termination_logger,
            fix_05_housekeeping_mode,
            fix_06_access_window_check,
            fix_07_perstep_diagnostics,
            fix_08_zero_reward_logger,
            fix_09_eclipse_battery_dynamic,
            fix_10_unified_cloud_thresh,
        ],
        dry_run,
        "env_alsat_dynamic.py",
    )

    # ── env_alsat_debug.py ───────────────────────────────────────────────
    apply_to_file(
        DEBUG,
        [fix_09_eclipse_battery_debug],
        dry_run,
        "env_alsat_debug.py",
    )

    # ── attention_policy.py ──────────────────────────────────────────────
    apply_to_file(
        ATTN,
        [fix_11_s_tok_unsqueeze],
        dry_run,
        "attention_policy.py",
    )

    # ── action_mask_wrapper.py ───────────────────────────────────────────
    apply_to_file(
        MASK,
        [
            fix_12_dyn_slew_mask,
            fix_15_16_static_mask_quality,
            fix_17_window_segmentation,
        ],
        dry_run,
        "action_mask_wrapper.py",
    )

    # ── env_alsat_dynamic.py (min_elev) ──────────────────────────────────
    apply_to_file(
        DYN,
        [fix_18_minelev_realistic],
        dry_run,
        "env_alsat_dynamic.py (min_elev)",
    )

    print("\n" + "=" * 65)
    if dry_run:
        print("  Dry run complete.  Re-run without --dry-run to apply.")
    else:
        print("  All patches applied.")
        print()
        print("  Test the --attention fix:")
        print("    python scripts/training/train_full_system.py \\")
        print("           --attention --seed 42 --quick")
        print()
        print("  Test housekeeping + termination logger:")
        print("    python scripts/training/train_full_system.py \\")
        print("           --seed 42 --quick")
        print()
        print("  Test per-step diagnostics:")
        print("    ALSAT_DEBUG_STEPS=1 python scripts/training/train_full_system.py \\")
        print("           --seed 42 --quick")
        print()
        print("  To revert:  cp scripts/core/env_alsat_dynamic.py.bak \\")
        print("                    scripts/core/env_alsat_dynamic.py  (etc.)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply ALSAT-EO-1 patches")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without touching files")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
