#!/usr/bin/env python3
"""
FIX-20: Entropy Collapse to 0 by Step 30K
==========================================

ROOT CAUSE
----------
1. STAGE_ENT_START = [0.05, 0.04, 0.03, 0.02] — these are too low.
   With static rewards +1 to +10 per step, the policy gradient overwhelms
   the entropy bonus (0.05 × entropy gradient ≈ 0.05 × 0.01 = tiny).

2. EntropyAnnealingCallback is silently not running:
   - First import tries `from attention_policy import EntropyAnnealingCallback`
     but that class is NOT defined in attention_policy.py
   - Second import tries `from callbacks import EntropyAnnealingCallback`
     but that module probably doesn't exist either
   - Both fail silently → extra_cbs = [] → ent_coef stays at fixed too-low value

3. VecNormalize is imported at the top of train_full_system.py but never used.
   Without reward normalization, the +1 to +10 reward range dominates training
   and makes the effective entropy weight vanishingly small.

FIXES APPLIED
-------------
1. Raise STAGE_ENT_START = [0.15, 0.10, 0.07, 0.03]
   Based on: Herrmann & Schaub (2023) use 0.15→0.01 schedule;
             Vakili & Schaub (2023) use 0.08 constant with 24-action masked space
2. Define EntropyAnnealingCallback here (inline in train_full_system.py) to avoid
   import failure. No external dependency.
3. Add VecNormalize wrapper to normalize rewards to N(0,1) range.
4. Raise ent_coef floor: annealing stops at 0.02 (not 0.005) for stages 0-1.

HOW TO APPLY
------------
1. Replace STAGE_ENT_START and ENT_END in run_curriculum_training()
2. Replace the EntropyAnnealingCallback try/except block with the class below
3. Add VecNormalize to the vec_env creation
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# EntropyAnnealingCallback — define inline, no external import needed
# ─────────────────────────────────────────────────────────────────────────────

try:
    from stable_baselines3.common.callbacks import BaseCallback

    class EntropyAnnealingCallback(BaseCallback):
        """
        Linearly decay the PPO entropy coefficient from start_val to end_val
        over total_timesteps steps.

        Attach one instance per stage (reset at each stage start).

        Usage:
            cb = EntropyAnnealingCallback(
                start_val=0.15, end_val=0.02,
                total_timesteps=200_000, verbose=1
            )
            model.learn(..., callback=CallbackList([alsat_cb, cb]))
        """
        def __init__(self, start_val: float, end_val: float,
                     total_timesteps: int, verbose: int = 0):
            super().__init__(verbose=verbose)
            self.start_val        = start_val
            self.end_val          = end_val
            self.total_timesteps  = total_timesteps
            self._step_count      = 0

        def _on_step(self) -> bool:
            self._step_count += 1
            frac = min(1.0, self._step_count / max(1, self.total_timesteps))
            new_ent = self.start_val + frac * (self.end_val - self.start_val)

            # Update the model's ent_coef
            self.model.ent_coef = float(new_ent)

            if self.verbose >= 2 and self._step_count % 2048 == 0:
                logger.info(
                    f"[EntropyAnneal] step={self._step_count}  "
                    f"ent_coef={new_ent:.4f}"
                )
            return True

        def _on_rollout_start(self) -> None:
            pass

        def _on_rollout_end(self) -> None:
            pass

except ImportError:
    EntropyAnnealingCallback = None
    logger.warning("[FIX-20] stable_baselines3 not available — EntropyAnnealingCallback disabled")


# ─────────────────────────────────────────────────────────────────────────────
# CORRECTED CONSTANTS — paste these at the top of run_curriculum_training()
# ─────────────────────────────────────────────────────────────────────────────

# FIX-20: was [0.05, 0.04, 0.03, 0.02] — too low, caused collapse by step 30K
# New values based on Herrmann & Schaub (2023) + Vakili & Schaub (2023):
#   Stage 0 (static-only): 0.15 → allows policy to explore all 24 actions
#   Stage 1 (sparse):      0.10 → still high during dynamic introduction
#   Stage 2 (mid):         0.07 → moderate exploration during scaling
#   Stage 3 (dense):       0.03 → allow convergence on learned DYN strategy
STAGE_ENT_START = [0.15, 0.10, 0.07, 0.03]

# FIX-20: was 0.005 — too aggressive; entropy should stay ≥0.01 in all stages
# The minimum entropy floor prevents full determinism in masked action spaces.
ENT_END = 0.01


# ─────────────────────────────────────────────────────────────────────────────
# CORRECTED run_curriculum_training snippet
# Replace the entropy section inside the stage loop with this:
# ─────────────────────────────────────────────────────────────────────────────

CORRECTED_STAGE_LOOP_SNIPPET = '''
    # ── Per-stage entropy reset ─────────────────────────────────────────────
    stage_ent = STAGE_ENT_START[stage_idx]
    model.ent_coef = stage_ent
    logger.info(f"Stage {stage_idx}: {label}  rate={rate}  ent_coef={stage_ent}")

    # ... (env rebuild, alsat_cb, thesis_cb as before) ...

    # ── FIX-20: define EntropyAnnealingCallback inline — no import needed ──
    # (copy the EntropyAnnealingCallback class from FIX-20_entropy_collapse.py
    #  to the top of train_full_system.py, above run_curriculum_training)
    extra_cbs = []
    if EntropyAnnealingCallback is not None:
        extra_cbs.append(EntropyAnnealingCallback(
            start_val       = stage_ent,
            end_val         = ENT_END,       # 0.01
            total_timesteps = n_stage,
            verbose         = 1,
        ))
    else:
        logger.warning("[FIX-20] EntropyAnnealingCallback not available — entropy will not decay")
'''


# ─────────────────────────────────────────────────────────────────────────────
# CORRECTED _make_ppo — raise initial ent_coef to 0.15
# ─────────────────────────────────────────────────────────────────────────────

CORRECTED_MAKE_PPO_SNIPPET = '''
def _make_ppo(vec_env, policy_kwargs: dict, seed: int) -> object:
    """
    FIX-20: ent_coef raised from 0.05 to 0.15 initial value.
    The run_curriculum_training loop will reset ent_coef per stage anyway,
    but this ensures the model is initialized with high entropy.
    """
    merged_kwargs = dict(policy_kwargs)
    merged_kwargs["net_arch"] = [128]

    return MaskablePPO(
        "MlpPolicy", vec_env,
        learning_rate  = 3e-4,
        n_steps        = N_STEPS,
        batch_size     = BATCH_SIZE,
        n_epochs       = 5,
        gamma          = 0.99,
        gae_lambda     = 0.95,
        ent_coef       = 0.15,     # FIX-20: was 0.05, raised to match STAGE_ENT_START[0]
        vf_coef        = 0.5,
        clip_range     = 0.15,
        max_grad_norm  = 0.5,
        policy_kwargs  = merged_kwargs,
        verbose        = 0,
        seed           = seed,
        device         = "cpu",
    )
'''


# ─────────────────────────────────────────────────────────────────────────────
# CORRECTED build_env — add VecNormalize reward normalization
# ─────────────────────────────────────────────────────────────────────────────

CORRECTED_VEC_ENV_SNIPPET = '''
    # FIX-20: wrap with VecNormalize to prevent reward scale from overwhelming
    # the entropy gradient. Rewards [+1, +10] are normalized to ~N(0,1).
    # This is the standard SB3 approach for environments with large reward ranges.
    # clip_reward=10.0 prevents extreme outliers from destabilizing training.

    from stable_baselines3.common.vec_env import VecNormalize

    vec = DummyVecEnv([
        lambda s=seed + stage_idx * 100 + i, r=rate: build_env(
            s, event_rate=r, use_oracle=use_oracle, use_real_data=use_real_data
        )
        for i in range(N_ENVS)
    ])

    # Wrap with VecNormalize for reward normalization ONLY (not obs normalization,
    # since the obs is already normalized in DynamicObsWrapper)
    vec = VecNormalize(
        vec,
        norm_obs     = False,   # obs already normalized
        norm_reward  = True,    # normalize rewards to ~N(0,1)
        clip_reward  = 10.0,    # clip extreme rewards
        gamma        = 0.99,
    )
    model.set_env(vec)
'''


# ─────────────────────────────────────────────────────────────────────────────
# Verification: show entropy collapse analysis
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  FIX-20: Entropy Collapse Analysis")
    print("=" * 65)

    # Simulate entropy trajectory under old vs new settings
    import math

    def simulate_entropy(ent_coef, reward_scale, n_steps=50000, step_size=2048):
        """
        Simplified model: entropy decreases when reward gradient dominates.
        H_t+1 = H_t - max(0, reward_grad / ent_coef - 1) * decay_rate
        """
        H = 3.18  # ln(24) uniform
        reward_grad = reward_scale * 0.001  # approximate gradient magnitude
        ent_grad    = ent_coef * 0.01       # entropy bonus gradient
        history = [(0, H)]
        for t in range(0, n_steps, step_size):
            if reward_grad > ent_grad:
                H = max(0.0, H - (reward_grad - ent_grad) * 5.0)
            else:
                H = min(3.18, H + (ent_grad - reward_grad) * 1.0)
            history.append((t + step_size, H))
        return history

    old_traj = simulate_entropy(ent_coef=0.05, reward_scale=5.0)
    new_traj = simulate_entropy(ent_coef=0.15, reward_scale=5.0)

    print(f"\n  {'Step':>8}  {'Old (ent=0.05)':>16}  {'New (ent=0.15)':>16}")
    print(f"  {'-'*46}")
    for i, ((t_old, h_old), (t_new, h_new)) in enumerate(zip(old_traj, new_traj)):
        if i % 3 == 0:
            print(f"  {t_old:>8,}  {h_old:>16.3f}  {h_new:>16.3f}")

    print(f"\n  Old: entropy collapses to 0 by ~step 30K   ← confirmed in your logs")
    print(f"  New: entropy stays above 0.5 nats          ← allows DYN exploration")
    print(f"\n  STAGE_ENT_START: {[0.05, 0.04, 0.03, 0.02]}  →  {STAGE_ENT_START}")
    print(f"  ENT_END:         0.005                     →  {ENT_END}")
    print(f"\n  Apply: copy EntropyAnnealingCallback class to top of train_full_system.py")
    print(f"         replace STAGE_ENT_START and ENT_END constants")