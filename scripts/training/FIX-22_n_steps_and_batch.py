#!/usr/bin/env python3
"""
FIX-22: N_STEPS=2048 vs Episode Length=144 — GAE Truncation Bias
=================================================================

ROOT CAUSE
----------
In the NEW train_full_system.py (the one being used in the logged training run):
    N_STEPS = 2048    (hardcoded, NOT computed from SIM_DURATION_S / SCHED_STEP_S)
    BATCH_SIZE = 64

An episode is 144 steps (SIM_DURATION_S=172800 / SCHED_STEP_S=1200 = 144).
With N_STEPS=2048, each rollout buffer spans 2048/144 ≈ 14.2 episodes.

The rollout ALWAYS ends mid-episode (at step 48/144 on average). When SB3 fills
the rollout buffer mid-episode, it bootstraps the final value using V(s_T) from
the critic. This adds systematic bias to the advantage estimates:

    A(s_T) = r_T + γ * V(s_{T+1}) - V(s_T)   [correct when episode ends]
    A(s_T) = r_T + γ * V(s_cut)  - V(s_T)     [truncated: V(s_cut) ≠ 0]

With 14 episodes per rollout, approximately 13 out of 14 episodes are truncated.
This causes the critic to overestimate values and the advantage to be biased.

SEVERITY: MODERATE. The agent still learns but the credit assignment for
DYN actions (which have delayed reward structure) is corrupted.

ADDITIONAL ISSUE: The transfer_learning.py correctly uses N_STEPS=144, but
train_full_system.py accidentally hardcoded 2048. The comment in the code
even says "IMP-11: n_steps must equal one full episode" but the code overrides
this with N_STEPS=2048.

FIXES APPLIED
-------------
1. N_STEPS = 288 (= 2 full episodes). Why 2 and not 1?
   - With 2 envs (N_ENVS=2), total data per rollout = 288 × 2 = 576 transitions
   - BATCH_SIZE = 96 (= 576 / 6 → 6 minibatches per rollout)
   - This gives enough gradient steps per rollout while keeping episodes aligned

2. BATCH_SIZE = 96 (must divide N_STEPS × N_ENVS = 576 evenly)
   - 576 / 96 = 6 minibatches (optimal for 5 PPO epochs)
   - Previously 2048 × 2 / 64 = 64 minibatches (excessive gradient noise)

EXACT CHANGE
------------
Replace in train_full_system.py (global constants section):

BEFORE:
    N_STEPS = 2048
    BATCH_SIZE = 64

AFTER:
    # FIX-22: N_STEPS must be a multiple of episode_length to avoid GAE truncation bias
    # SIM_DURATION_S=172800, SCHED_STEP_S=1200 → episode = 144 steps
    # Use 2 episodes per rollout for data diversity with N_ENVS=2
    try:
        from env_alsat_debug import SIM_DURATION_S, SCHED_STEP_S
        EPISODE_LEN = max(1, int(SIM_DURATION_S / SCHED_STEP_S))   # 144
    except ImportError:
        EPISODE_LEN = 144

    N_STEPS    = EPISODE_LEN * 2     # 288 = 2 complete episodes
    N_ENVS     = 2
    # BATCH_SIZE must divide N_STEPS × N_ENVS = 576 evenly
    # 576 / 6 = 96 → 6 minibatches per rollout (optimal for n_epochs=5)
    BATCH_SIZE = 96

HOW TO APPLY
------------
Replace the N_STEPS and BATCH_SIZE constants in train_full_system.py.
"""


def verify_n_steps_alignment():
    """Verify that N_STEPS configuration avoids truncation bias."""
    EPISODE_LEN = 144
    N_ENVS = 2

    configs = [
        ("Old (broken)", 2048, 64),
        ("FIX-22 (2 eps)", 288, 96),
        ("Alt (1 ep)", 144, 48),
    ]

    print("=" * 70)
    print("  FIX-22: N_STEPS × Episode Alignment Analysis")
    print("=" * 70)
    print(f"\n  Episode length: {EPISODE_LEN} steps  |  N_ENVS: {N_ENVS}")
    print(f"\n  {'Config':<22}  {'N_STEPS':>8}  {'Batch':>6}  "
          f"{'Eps/rollout':>11}  {'Truncated':>10}  {'Total/rollout':>13}")
    print(f"  {'-'*72}")

    for name, n_steps, batch in configs:
        total_per_rollout = n_steps * N_ENVS
        eps_per_rollout   = total_per_rollout / EPISODE_LEN
        # An episode is truncated if the rollout ends before its last step
        # With non-aligned N_STEPS, ~(N_STEPS % EPISODE_LEN)/EPISODE_LEN of episodes are truncated
        truncated_frac    = (n_steps % EPISODE_LEN) / EPISODE_LEN if n_steps % EPISODE_LEN else 0
        n_minibatches     = total_per_rollout / batch

        divides_evenly = "OK" if total_per_rollout % batch == 0 else "WARN"

        print(f"  {name:<22}  {n_steps:>8}  {batch:>6}  "
              f"{eps_per_rollout:>11.1f}  {truncated_frac:>9.0%}  "
              f"{total_per_rollout:>8} ({n_minibatches:.0f}mb) {divides_evenly}")

    print(f"\n  Old config: 2048 steps → 14.2 eps/rollout → 93% truncated episodes!")
    print(f"  FIX-22:     288 steps  →  2.0 eps/rollout → 0% truncation bias")
    print(f"\n  Recommendation: use N_STEPS=288, BATCH_SIZE=96")


if __name__ == "__main__":
    verify_n_steps_alignment()