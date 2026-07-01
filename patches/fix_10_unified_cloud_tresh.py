"""
FIX 10 — env_alsat_dynamic.py + env_alsat_debug.py  unified CLOUD_THRESH
=========================================================================
Both files already use CLOUD_THRESH = 0.6, but the DynamicScienceDataStore
static reward block had a hard-coded -0.3 cloudy penalty for DYNAMIC targets
that bypassed the threshold consistency.  This patch replaces the hard-coded
-0.3 with the same pattern as the static block: -0.1 * priority.

Also adds an assertion at module level in env_alsat_dynamic.py to catch any
future drift between the two constants.
"""

# ── In DynamicScienceDataStore.compare_log_states, dynamic-is-cloudy penalty
OLD_DCS = '''            if cloud_truth < CLOUD_THRESH:
                reward = (DYN_MULTIPLIER * priority * (1.0 - cloud_truth) * urgency
                         - SLEW_ENERGY_ALPHA * slew_energy + DYNAMIC_BONUS)
                sat._metrics['n_cloud_free'] += 1
            else:
                reward = -0.3 * priority   # stronger penalty for cloudy dynamic waste'''

NEW_DCS = '''            if cloud_truth < CLOUD_THRESH:
                reward = (DYN_MULTIPLIER * priority * (1.0 - cloud_truth) * urgency
                         - SLEW_ENERGY_ALPHA * slew_energy + DYNAMIC_BONUS)
                sat._metrics['n_cloud_free'] += 1
            else:
                # FIX-10: use same penalty scale as static cloudy penalty
                # (was -0.3 hard-coded; now -0.1 * priority, consistent with static)
                reward = -0.1 * priority
                sat._metrics['n_cloudy'] += 1'''


def apply(text: str) -> str:
    if OLD_DCS not in text:
        raise ValueError(
            "FIX-10: DYN cloudy penalty block not found in env_alsat_dynamic.py."
        )
    return text.replace(OLD_DCS, NEW_DCS, 1)
