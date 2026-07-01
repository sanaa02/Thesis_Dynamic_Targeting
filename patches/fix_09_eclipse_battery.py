"""
FIX 09 — env_alsat_debug.py  eclipse-aware battery model
=========================================================
The current model uses a constant NET_BASE_POWER_W = +225 W, which keeps
the battery perpetually near 100%.  This is unrealistic and means the
battery SOC observation feature carries no useful information for the agent.

Realistic ALSAT-EO-1 power budget (686 km SSO, 97.4° inclination):
  Orbital period      : ~5900 s
  Sunlit fraction     : ~65.3%  (eclipse ~34.7% → ~2047 s per orbit)
  Solar panel output  : 1.5 m² × 0.28 eff × 1367 W/m² = 574 W peak
  Solar @ incidence   : 574 × 0.653 × 0.95 cosine avg = 356 W average sunlit
  Housekeeping load   : ~50 W (comms off, payload idle)
  Net in sunlight     : 356 – 50 = +306 W  (charges battery)
  Net in eclipse      : 0  – 50 = –50 W    (discharges battery)

Orbit-averaged net: 0.653×306 + 0.347×(–50) = +182 W (battery trends up slowly)
Min SOC at eclipse exit ≈ 87–90% for a 40 Wh battery — healthy but visible.

Implementation: patch DynamicAlsatSatellite (and AlsatSatellite) to call a
per-step battery updater that uses the eclipse flag from the observation vector.
The eclipse flag (obs[12] after the base obs block) is 1.0 in eclipse.

NOTE: this patch adds a standalone battery updater that is called from
DynamicObsWrapper._build_obs() — the only place that runs after every step
with access to the current observation.
"""

# ── Patch target: the comment block for the battery model ────────────────────
OLD_DEBUG = '''NET_BASE_POWER_W = 225.0   # W  orbit-averaged net (solar_avg - housekeeping)                                    # -20 W'''

NEW_DEBUG = '''# FIX-09: Eclipse-aware power model constants
# In sunlight the solar panel charges the battery; in eclipse only housekeeping
# draws power.  These values are for ALSAT-EO-1 at 686 km SSO:
SOLAR_PANEL_AREA_M2     = 1.5    # m²
SOLAR_PANEL_EFF         = 0.28   # BOL GaAs efficiency
SOLAR_IRRADIANCE_W_M2   = 1367.0 # W/m²
SOLAR_INCIDENCE_FACTOR  = 0.653  # fraction of orbit in sunlight × avg cosine
HOUSEKEEPING_LOAD_W     = 50.0   # W  (idle — payload off, no comms)
# Derived:
_P_SOLAR_PEAK  = SOLAR_PANEL_AREA_M2 * SOLAR_PANEL_EFF * SOLAR_IRRADIANCE_W_M2
_P_NET_SUNLIT  = _P_SOLAR_PEAK * SOLAR_INCIDENCE_FACTOR * 0.95 - HOUSEKEEPING_LOAD_W  # ~+306 W
_P_NET_ECLIPSE = -HOUSEKEEPING_LOAD_W                                                   # –50 W
# Legacy constant (kept for compatibility; no longer used for dynamics):
NET_BASE_POWER_W = (_P_NET_SUNLIT * SOLAR_INCIDENCE_FACTOR
                    + _P_NET_ECLIPSE * (1.0 - SOLAR_INCIDENCE_FACTOR))  # ~+182 W avg


def eclipse_battery_step(satellite, dt_s: float) -> None:
    """
    Update battery SOC for one simulation sub-step of duration dt_s seconds.
    Uses the eclipse flag from the satellite dynamics to decide charge/discharge.

    Called from DynamicObsWrapper._build_obs() which runs after every step.
    """
    try:
        # Eclipse flag: 1.0 = in eclipse, 0.0 = in sunlight.
        # bsk_rl stores this in dynamics.eclipse_shadow after bsk_patches.P1.
        eclipse_flag = float(getattr(
            getattr(satellite, 'dynamics', satellite),
            'eclipse_shadow', 0.0
        ))
        # Fallback: check obs vector index 12 if eclipse_shadow not available
        if eclipse_flag not in (0.0, 1.0):
            eclipse_flag = 0.0  # assume sunlit if uncertain

        p_net_W = _P_NET_ECLIPSE if eclipse_flag > 0.5 else _P_NET_SUNLIT

        # Current SOC (0–1)
        dyn = getattr(satellite, 'dynamics', satellite)
        cap_Wh = float(getattr(dyn, 'batteryStorageCapacity', BATTERY_WH * 3600.0)) / 3600.0
        soc = float(getattr(satellite, 'battery_charge_fraction',
                    getattr(dyn, 'battery_charge_fraction', 1.0)))

        # ΔE = P × dt (Wh)
        delta_Wh = p_net_W * dt_s / 3600.0
        new_soc  = float(np.clip(soc + delta_Wh / max(cap_Wh, 1.0), 0.0, 1.0))

        # Write back — bsk_rl reads battery_charge_fraction from the satellite shortcut
        if hasattr(satellite, 'battery_charge_fraction'):
            satellite.battery_charge_fraction = new_soc
        if hasattr(dyn, 'battery_charge_fraction'):
            dyn.battery_charge_fraction = new_soc
    except Exception:
        pass  # never crash a step over battery update'''


def apply_debug(text: str) -> str:
    if OLD_DEBUG not in text:
        raise ValueError(
            "FIX-09: NET_BASE_POWER_W constant not found in env_alsat_debug.py.\n"
            "File may already be patched."
        )
    return text.replace(OLD_DEBUG, NEW_DEBUG, 1)


# ── Patch target: _build_obs to call eclipse_battery_step ────────────────────
# We hook into DynamicObsWrapper._build_obs in env_alsat_dynamic.py
OLD_DYNAMIC = '''    def _build_obs(self, base_obs: np.ndarray, tau_norm: float) -> np.ndarray:
        try:
            sat   = self.env.unwrapped.satellites[0]
            now   = float(sat.simulator.sim_time)
            slots = self._mgr.get_slots(sat, now)
        except Exception:
            slots = [None] * N_DYN_SLOTS
            sat   = None
            now   = 0.0'''

NEW_DYNAMIC = '''    def _build_obs(self, base_obs: np.ndarray, tau_norm: float) -> np.ndarray:
        try:
            sat   = self.env.unwrapped.satellites[0]
            now   = float(sat.simulator.sim_time)
            slots = self._mgr.get_slots(sat, now)
        except Exception:
            slots = [None] * N_DYN_SLOTS
            sat   = None
            now   = 0.0

        # ── FIX-09: Eclipse-aware battery update ──────────────────────────
        # Called once per decision step (not per sub-step) to advance battery
        # SOC based on whether the satellite is currently in eclipse or sunlit.
        if sat is not None:
            try:
                from env_alsat_debug import eclipse_battery_step
                _dt = float(np.clip(
                    now - self._prev_time if now > self._prev_time else BASE_STEP_S,
                    0.0, MAX_ACTION_DUR_S
                ))
                eclipse_battery_step(sat, _dt)
            except Exception:
                pass
        # ──────────────────────────────────────────────────────────────────'''


def apply_dynamic(text: str) -> str:
    if OLD_DYNAMIC not in text:
        raise ValueError(
            "FIX-09: _build_obs header not found in env_alsat_dynamic.py."
        )
    return text.replace(OLD_DYNAMIC, NEW_DYNAMIC, 1)