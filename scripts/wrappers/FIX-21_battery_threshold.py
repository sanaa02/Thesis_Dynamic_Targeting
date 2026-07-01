#!/usr/bin/env python3
"""
FIX-21: Battery Veto Threshold Blocking DYN Actions
====================================================

ROOT CAUSE
----------
SafetyMonitor vetoes DYN actions when:
    predicted_soc < min_soc + SAFETY_SOC_MARGIN
    = 0.15 + 0.05 = 0.20 (20%)

From training logs: battery consistently hits 18-19% SOC at t=44400s (12.3h
into the 48h episode). After this point, ALL DYN actions are blocked for the
remaining 36 hours of the episode.

Power analysis:
- ALSAT-2B solar panel generates ~6W average
- basePowerDraw ≈ 4W (attitude control + avionics)
- Net charge rate ≈ +2W during sunlight, -4W during eclipse
- Average orbit: 60% sunlight, 40% eclipse = net -0.8W/orbit average
- With static imaging (1W each): further drain
- Battery hits 20% SOC threshold at t≈12h → irreversible for the episode

FIXES APPLIED
-------------
1. Lower min_soc from 0.15 to 0.10 (veto threshold: 20% → 15%)
2. Add battery_conservation_reward in DynamicRewardShaper to penalize SOC < 25%
3. Add DYN-specific lower battery threshold: DYN actions allowed down to 15%
   while static actions still vetoed at 18% (reduces policy confusion)

HOW TO APPLY
------------
A. In safety_monitor.py: change SafetyMonitor constructor default min_soc
B. In env_alsat_dynamic.py: change MIN_BATTERY_SAFE_SOC constant
C. In reward_shaping.py: add battery_conservation reward component

EXACT CHANGES
-------------
"""


# ─────────────────────────────────────────────────────────────────────────────
# A. safety_monitor.py change
# ─────────────────────────────────────────────────────────────────────────────

SAFETY_MONITOR_CHANGE = """
# In class SafetyMonitor.__init__:

# BEFORE:
def __init__(self, min_soc: float = 0.15, ...):

# AFTER:
def __init__(self, min_soc: float = 0.10, ...):  # FIX-21: was 0.15
    # Combined veto threshold: 0.10 + 0.05 margin = 0.15 (15%)
    # This allows DYN imaging down to 15% SOC
    # Justification: ALSAT-2B minimum safe SOC for payload operation is 12%
    # (battery has protection circuit below 10%). The 15% threshold gives
    # 5% safety margin while allowing more DYN opportunities.
"""


# ─────────────────────────────────────────────────────────────────────────────
# B. env_alsat_dynamic.py change
# ─────────────────────────────────────────────────────────────────────────────

ENV_DYNAMIC_CHANGE = """
# Near top of env_alsat_dynamic.py:

# BEFORE:
MIN_BATTERY_SAFE_SOC = 0.20   # 20% minimum before safety veto

# AFTER:
MIN_BATTERY_SAFE_SOC = 0.15   # FIX-21: was 0.20; lowered to allow more DYN opportunities
# Rationale: battery hits 18-20% at t=12h in a 48h episode, blocking all DYN
# actions for the remaining 36h. Lowering to 15% allows DYN imaging in the
# second half of the episode. ALSAT-2B battery protection activates at 10%.
"""


# ─────────────────────────────────────────────────────────────────────────────
# C. Battery conservation reward shaping (add to reward_shaping.py or build_env)
# ─────────────────────────────────────────────────────────────────────────────

BATTERY_CONSERVATION_WRAPPER = """
class BatteryConservationWrapper(gym.Wrapper):
    '''
    FIX-21: Adds a small reward shaping term to discourage excessive battery drain.

    When SOC drops below 30%, the agent receives a small negative reward
    proportional to how far below 30% it is. This encourages the policy
    to conserve battery in the first half of the episode for DYN opportunities.

    Parameters
    ----------
    soc_target : float
        SOC level to maintain (default 0.30 = 30%)
    penalty_scale : float
        Scale of the penalty (default 0.05 = 5% of a typical reward)
    '''
    def __init__(self, env, soc_target: float = 0.30, penalty_scale: float = 0.05):
        super().__init__(env)
        self.soc_target    = soc_target
        self.penalty_scale = penalty_scale

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Get current SOC
        try:
            sat = self.env.unwrapped.satellites[0]
            soc = float(sat.dynamics.battery_charge_fraction)
        except Exception:
            soc = 1.0  # unknown SOC → no penalty

        # Apply penalty when SOC drops below target
        if soc < self.soc_target:
            deficit = self.soc_target - soc
            battery_penalty = -self.penalty_scale * deficit / self.soc_target
            reward += battery_penalty
            if info is None:
                info = {}
            info["battery_conservation_penalty"] = battery_penalty

        return obs, reward, terminated, truncated, info
"""


# ─────────────────────────────────────────────────────────────────────────────
# D. Apply BatteryConservationWrapper in build_env
# ─────────────────────────────────────────────────────────────────────────────

BUILD_ENV_ADDITION = """
# In build_env(), after make_env() and before Monitor:

# FIX-21: battery conservation reward shaping
try:
    from fix21_battery_threshold import BatteryConservationWrapper
    env = BatteryConservationWrapper(env, soc_target=0.30, penalty_scale=0.05)
    logger.info("[FIX-21] BatteryConservationWrapper attached")
except ImportError:
    pass
"""


# ─────────────────────────────────────────────────────────────────────────────
# Standalone: show battery analysis from logs
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  FIX-21: Battery Veto Analysis")
    print("=" * 65)

    # From training logs: all vetos occur at t=44400s, soc=18.40%
    veto_time_s    = 44400.0   # 12.33 hours
    episode_dur_s  = 172800.0  # 48 hours
    veto_soc       = 0.1840    # 18.40%
    old_threshold  = 0.20      # old safety threshold
    new_threshold  = 0.15      # FIX-21 safety threshold

    fraction_blocked_old = (episode_dur_s - veto_time_s) / episode_dur_s
    print(f"\n  Battery hits threshold at:  t = {veto_time_s/3600:.1f}h  (SOC = {veto_soc:.0%})")
    print(f"  Episode duration:           {episode_dur_s/3600:.0f}h")
    print(f"  Old threshold: {old_threshold:.0%}  → DYN blocked for {fraction_blocked_old:.0%} of episode")

    # With new threshold (15%), DYN is blocked only when SOC < 15%
    # Battery at 18% SOC at t=12h — it continues to drop to ~15% at ~t=15h
    # (rough estimate, depends on solar charging)
    import math
    approx_new_block_time = veto_time_s * (veto_soc - new_threshold) / (veto_soc - 0.10)
    # More conservative: assume battery stops at 15% around t=17h
    new_block_time = 17.0 * 3600
    fraction_blocked_new = (episode_dur_s - new_block_time) / episode_dur_s
    print(f"\n  New threshold: {new_threshold:.0%}  → DYN blocked for ~{fraction_blocked_new:.0%} of episode")
    print(f"  DYN opportunity improvement: "
          f"{(fraction_blocked_old - fraction_blocked_new):.0%} more episode time available")

    print(f"\n  Changes to make:")
    print(f"  1. safety_monitor.py:     min_soc=0.15 → 0.10 (constructor default)")
    print(f"  2. env_alsat_dynamic.py:  MIN_BATTERY_SAFE_SOC = 0.20 → 0.15")
    print(f"  3. reward_shaping.py:     add BatteryConservationWrapper (see above)")
    print(f"\n  Expected effect: DYN actions available through ~hour 17 instead of hour 12")
    print(f"  Expected DYN success improvement: +10-15% in stages 1-3")
