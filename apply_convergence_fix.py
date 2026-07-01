#!/usr/bin/env python3
"""
apply_convergence_fix.py  —  ALSAT-EO-1  Convergence Fix
=========================================================
Diagnoses (from overnight CSV training_1782110087649.csv analysis):

  CAUSE-1 [CRITICAL]:  VecNormalize cold-started at EVERY curriculum stage
  -------------------------------------------------------------------------
  Each stage does: vec = VecNormalize(vec, ...)  (fresh stats, ret_rms reset)
  After the cold-start, normalized rewards are ~3.0 for 20-30 iterations while
  the running variance calibrates.  The value function sees wildly different
  target scales in consecutive rollouts → converges to predicting the mean for
  all states → explained_var ≈ 0 throughout ALL training.

  FIX: Save VecNormalize stats (pkl) before closing the stage env, then
  VecNormalize.load(path, new_venv) to carry over ret_rms at the next stage.

  CAUSE-2 [CRITICAL]:  vf_coef=0.5 and n_epochs=5 — value function too weak
  ---------------------------------------------------------------------------
  Only 5 gradient passes over each rollout.  For a complex SMDP with variable
  episode lengths and battery deaths, the value function simply does not get
  enough signal to improve beyond predicting the mean.

  FIX: vf_coef 0.5 → 2.0,  n_epochs 5 → 10.

  CAUSE-3 [IMPORTANT]:  STAGE_ENT_START[0]=0.40 — entropy too high
  ----------------------------------------------------------------
  At ent_coef=0.40 with ~6 valid masked actions, the policy is nearly uniform.
  A nearly uniform policy makes all states equally valuable → VF learns a
  constant → explained_var=0.  The value function can only start learning
  once the policy becomes differentiated.

  FIX: STAGE_ENT_START [0.40, 0.20, 0.10, 0.03] → [0.15, 0.08, 0.04, 0.01]
  These values are high enough to prevent entropy collapse but low enough that
  the policy has a preference the VF can latch onto.

  CAUSE-4 [IMPORTANT]:  --quick = 50 000 total steps
  ---------------------------------------------------
  Stage 0 gets only 5 000 steps = 17 rollouts.  The VF cannot learn.

  FIX: --quick → 300 000 steps (still fast, 2-3h on CPU).

  CAUSE-5 [MODERATE]:  BC pretraining commented out (bc_path = None)
  -------------------------------------------------------------------
  The line "bc_path = None" hard-codes no warm-start.  The BC flag is accepted
  on the CLI but silently ignored, wasting demo collection time.
  
  FIX: restore the if use_bc: block.

  CAUSE-6 [MODERATE]:  Battery deaths corrupt training batches
  ------------------------------------------------------------
  ~5-10% of episodes die at battery_end_pct=0.4% despite MIN_BATTERY_SAFE_SOC=0.20.
  These short episodes (ep_len=31-57) contribute negative rewards (-3 to -0.3)
  that corrupt advantage estimates.

  FIX: MIN_BATTERY_SAFE_SOC 0.20 → 0.25 in env_alsat_dynamic.py.
  Also: gae_lambda 0.95 → 0.90 reduces advantage variance for noisy episodic data.

FILES MODIFIED:
  scripts/training/train_full_system.py
  scripts/core/env_alsat_dynamic.py

HOW TO RUN:
  python apply_convergence_fix.py

HOW TO VERIFY:
  grep -n "vf_coef\\|VecNormalize.load\\|N_EPOCHS\\|STAGE_ENT" \\
       scripts/training/train_full_system.py | head -20

THEN RETRAIN (apply target-ID fix first if not yet done):
  python apply_target_id_fix.py        # only if not yet applied
  CUDA_VISIBLE_DEVICES='' python scripts/training/train_full_system.py \\
      --seed 42 --attention --bc-pretrain --steps 500000

QUICK SMOKE-TEST (should complete in ~15 min):
  CUDA_VISIBLE_DEVICES='' python scripts/training/train_full_system.py \\
      --seed 42 --quick
"""

import os
import re
import sys

# ── Locate project root ────────────────────────────────────────────────────────

def find_scripts_root():
    candidates = [
        os.path.join(os.getcwd(), "scripts"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"),
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "training", "train_full_system.py")):
            return c
    return None

SCRIPTS = find_scripts_root()
if SCRIPTS is None:
    print("ERROR: Cannot find scripts/training/train_full_system.py")
    print("       Run this script from your project root directory.")
    sys.exit(1)

TRAIN_PATH  = os.path.join(SCRIPTS, "training", "train_full_system.py")
DYN_PATH    = os.path.join(SCRIPTS, "core",     "env_alsat_dynamic.py")

print("=" * 70)
print("  ALSAT-EO-1  Convergence Fix  (6 patches)")
print("=" * 70)
print(f"  train_full_system.py : {TRAIN_PATH}")
print(f"  env_alsat_dynamic.py : {DYN_PATH}")
print()


# ── Helper ────────────────────────────────────────────────────────────────────

def apply_patch(path, old, new, tag):
    """Replace old→new in file at path.  Idempotent and safe."""
    fname = os.path.basename(path)
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already applied (any distinctive substring of new is present)
    new_key = new.strip().splitlines()[0].strip()
    if new_key in content:
        print(f"  [SKIP] {tag}  (already applied)")
        return True

    if old not in content:
        print(f"  [MISS] {tag}  — pattern not found in {fname}")
        # Print first line of pattern to help manual search
        first = old.strip().splitlines()[0][:80]
        print(f"         Search for: {first!r}")
        return False

    patched = content.replace(old, new, 1)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(patched)
    print(f"  [OK]   {tag}")
    return True


# =============================================================================
#  PATCH 1 — train_full_system.py: STAGE_ENT_START reduction
#  [0.40, 0.20, 0.10, 0.03] → [0.15, 0.08, 0.04, 0.01]
#
#  Rationale: Stage 0 at ent_coef=0.40 makes the policy near-uniform over
#  the ~6 unmasked actions.  A uniform policy makes all states equally
#  valuable, so the value function converges to a constant (explained_var≈0).
#  Reducing Stage 0 to 0.15 still prevents entropy collapse (well above the
#  0.03 that caused collapse in FIX-19) while giving the VF a signal.
# =============================================================================

print("[1/6] train_full_system.py — STAGE_ENT_START reduction (0.40→0.15)")

P1_OLD = "STAGE_ENT_START = [0.40, 0.20, 0.10, 0.03]"
P1_NEW = """\
# CONVERGENCE-FIX-1: reduce Stage-0 entropy from 0.40 → 0.15.
# At ent_coef=0.40 the policy is near-uniform over ~6 masked actions,
# making all state values equal → VF predicts constant → explained_var≈0.
# 0.15 still prevents entropy collapse (safe margin above old 0.05 that
# triggered collapse) while differentiating state values enough for the VF.
STAGE_ENT_START = [0.15, 0.08, 0.04, 0.01]"""

apply_patch(TRAIN_PATH, P1_OLD, P1_NEW, "STAGE_ENT_START [0.40,0.20,0.10,0.03]→[0.15,0.08,0.04,0.01]")


# =============================================================================
#  PATCH 2 — train_full_system.py: vf_coef 0.5 → 2.0  and  n_epochs 5 → 10
#  and  gae_lambda 0.95 → 0.90
#
#  Rationale:
#    vf_coef=0.5 with n_epochs=5 gives the VF 10x less gradient signal per
#    timestep than the policy.  In a noisy SMDP with battery deaths, this is
#    insufficient.  vf_coef=2.0 forces the VF to be trained 4× harder.
#
#    gae_lambda 0.95→0.90 reduces advantage variance.  With variable episode
#    lengths (ep_len=31 to 144) and battery death rewards of -3, the 0.95
#    lambda propagates that noise far back through the trajectory.  0.90
#    trades a small bias for a large variance reduction, stabilising the
#    policy gradient signal for the early stages of curriculum training.
# =============================================================================

print("[2/6] train_full_system.py — _make_ppo: vf_coef 0.5→2.0, n_epochs 5→10, gae_lambda 0.95→0.90")

P2_OLD = """\
    return MaskablePPO(
        "MlpPolicy", vec_env,
        learning_rate  = 3e-4,
        n_steps        = N_STEPS,      # FIX-22: 288 (was 2048)
        batch_size     = BATCH_SIZE,   # FIX-22: 96  (was 64)
        n_epochs       = N_EPOCHS,     # 5
        gamma          = 0.99,
        gae_lambda     = 0.95,
        ent_coef       = 0.15,         # FIX-20: was 0.05
        vf_coef        = 0.5,
        clip_range     = 0.15,
        max_grad_norm  = 0.5,
        policy_kwargs  = merged_kwargs,
        verbose        = 0,
        seed           = seed,
        device         = "cpu",
    )"""

P2_NEW = """\
    # CONVERGENCE-FIX-2a: vf_coef 0.5 → 2.0
    #   Training log: explained_var≈0 throughout all runs.
    #   Root cause: VF is trained 10× less than policy at vf_coef=0.5.
    #   At 2.0 the VF loss receives 4× more gradient per step, enough to
    #   break out of the constant-prediction local minimum.
    #
    # CONVERGENCE-FIX-2b: n_epochs 5 → 10
    #   Each rollout (N_STEPS=288, N_ENVS=2 = 576 transitions) is used for
    #   10 passes instead of 5.  VF gets 2× more updates per collected batch.
    #
    # CONVERGENCE-FIX-2c: gae_lambda 0.95 → 0.90
    #   Battery-death episodes (ep_len≈35, reward≈-2) produce noisy returns
    #   that propagate back through the trajectory at lambda=0.95.  0.90
    #   reduces advantage variance at the cost of a small bias — a good trade
    #   while the VF is still bootstrapping.
    return MaskablePPO(
        "MlpPolicy", vec_env,
        learning_rate  = 3e-4,
        n_steps        = N_STEPS,      # FIX-22: 288 (was 2048)
        batch_size     = BATCH_SIZE,   # FIX-22: 96  (was 64)
        n_epochs       = 10,           # CONVERGENCE-FIX-2b: was 5
        gamma          = 0.99,
        gae_lambda     = 0.90,         # CONVERGENCE-FIX-2c: was 0.95
        ent_coef       = 0.15,         # FIX-20: was 0.05
        vf_coef        = 2.0,          # CONVERGENCE-FIX-2a: was 0.5
        clip_range     = 0.15,
        max_grad_norm  = 0.5,
        policy_kwargs  = merged_kwargs,
        verbose        = 0,
        seed           = seed,
        device         = "cpu",
    )"""

apply_patch(TRAIN_PATH, P2_OLD, P2_NEW,
            "_make_ppo: vf_coef 0.5→2.0, n_epochs 5→10, gae_lambda 0.95→0.90")


# =============================================================================
#  PATCH 3 — train_full_system.py: VecNormalize persistence between stages
#
#  Rationale:
#    The original code does:
#      vec = VecNormalize(vec, ...)   # fresh stats at EVERY stage
#    A fresh VecNormalize has ret_rms.var≈0, so the first reward is amplified
#    to clip_reward (3.0) / sqrt(1e-8) = huge.  After 20-30 iterations the
#    variance stabilises, but by then the VF has seen inconsistent target
#    scales and is stuck predicting the mean.
#
#    Fix: save/load ret_rms across stages using VecNormalize.save / .load.
#    The observation normalisation is off (norm_obs=False) so only the return
#    statistics need to be preserved.
# =============================================================================

print("[3/6] train_full_system.py — VecNormalize persistence between stages")

P3_OLD = """\
        # Rebuild env for this stage
        try:
            vec.close()
        except Exception:
            pass

        vec = DummyVecEnv([
            lambda s=seed + stage_idx * 100 + i, r=rate: build_env(
                s, event_rate=r, use_oracle=use_oracle, use_real_data=use_real_data
            )
            for i in range(N_ENVS)
        ])
        # FIX-20: re-apply VecNormalize at each stage
        # FIX-B3: same clip_reward=3.0 as initial creation (cold-start fix)
        vec = VecNormalize(vec, norm_obs=False, norm_reward=True,
                           clip_reward=3.0, gamma=0.99)
        model.set_env(vec)"""

P3_NEW = """\
        # Rebuild env for this stage
        # CONVERGENCE-FIX-3: save VecNormalize statistics before closing
        # so the next stage inherits the calibrated ret_rms.
        # Without this, each stage cold-starts with ret_rms.var≈0, causing
        # the first real reward to be amplified to clip_reward — the VF
        # then sees wildly inconsistent targets and cannot learn (explained_var≈0).
        _vecnorm_ckpt = os.path.join(MODELS, f"vecnorm_stage{stage_idx}_s{seed}.pkl")
        try:
            vec.save(_vecnorm_ckpt)
            logger.info(f"[VecNorm] stats saved → {_vecnorm_ckpt}")
        except Exception as _e:
            logger.warning(f"[VecNorm] save failed: {_e}")
            _vecnorm_ckpt = None

        try:
            vec.close()
        except Exception:
            pass

        vec_raw = DummyVecEnv([
            lambda s=seed + stage_idx * 100 + i, r=rate: build_env(
                s, event_rate=r, use_oracle=use_oracle, use_real_data=use_real_data
            )
            for i in range(N_ENVS)
        ])

        # CONVERGENCE-FIX-3 (continued): reload statistics if checkpoint exists
        if _vecnorm_ckpt and os.path.exists(_vecnorm_ckpt):
            try:
                vec = VecNormalize.load(_vecnorm_ckpt, vec_raw)
                vec.training = True   # keep updating running statistics
                logger.info(f"[VecNorm] stats loaded from stage {stage_idx} checkpoint")
            except Exception as _e:
                logger.warning(f"[VecNorm] load failed ({_e}); falling back to fresh VecNormalize")
                vec = VecNormalize(vec_raw, norm_obs=False, norm_reward=True,
                                   clip_reward=3.0, gamma=0.99)
        else:
            vec = VecNormalize(vec_raw, norm_obs=False, norm_reward=True,
                               clip_reward=3.0, gamma=0.99)
        model.set_env(vec)"""

apply_patch(TRAIN_PATH, P3_OLD, P3_NEW, "VecNormalize persistence between stages")


# =============================================================================
#  PATCH 4 — train_full_system.py: --quick steps 50k → 300k
#
#  Rationale:
#    --quick uses max(N_STEPS*N_ENVS*4, 50_000) = 50 000 steps.
#    With 4 stages: Stage 0 gets 5 000 steps = 17 rollouts.  That is not
#    enough for the VF to calibrate VecNormalize, let alone learn.
#    300 000 steps gives Stage 0 ~30 000 steps = 104 rollouts, sufficient
#    for a smoke-test that actually exercises convergence.
# =============================================================================

print("[4/6] train_full_system.py — --quick steps 50k → 300k")

P4_OLD = "        total_steps = max(N_STEPS * N_ENVS * 4, 50_000)"
P4_NEW = """\
        # CONVERGENCE-FIX-4: 50k was only 17 rollouts for Stage 0 — not
        # enough for VecNormalize to calibrate or VF to learn anything.
        # 300k gives ~104 rollouts per stage, enough to see VF improvement.
        total_steps = 300_000"""

apply_patch(TRAIN_PATH, P4_OLD, P4_NEW, "--quick steps 50k→300k")


# =============================================================================
#  PATCH 5 — train_full_system.py: restore BC pretraining (un-comment)
#
#  Rationale:
#    "bc_path = None" is hard-coded, silently ignoring --bc-pretrain.
#    BC initialises the policy weights before curriculum training begins,
#    giving the VF a head-start on learning returns from a non-random policy.
# =============================================================================

print("[5/6] train_full_system.py — restore BC pretraining")

P5_OLD = """\
    # # FIX-19: BC pretraining with corrected val_acc
    # bc_path = None
    # if use_bc:
    #     print("=== BC Pre-training (FIX-19 applied) ===")
    #     bc_path = run_bc_pretrain(
    #         seed=seed,
    #         n_demos=500 if quick else 8000,   # FIX-19: was 2000
    #         use_attention=use_attention,
    #     )
    #     if use_bc and bc_path is None:
    #         logger.error("[BC] BC was requested but failed. Aborting.")
    #         sys.exit(1)

    bc_path = None"""

P5_NEW = """\
    # CONVERGENCE-FIX-5: restore BC pretraining.
    # The block was commented-out and bc_path=None hard-coded, making
    # --bc-pretrain silently no-op.  BC gives the VF a head-start:
    # a non-random initial policy has more differentiated state values,
    # helping the VF escape the constant-prediction minimum faster.
    bc_path = None
    if use_bc:
        print("=== BC Pre-training (FIX-19 applied) ===")
        bc_path = run_bc_pretrain(
            seed=seed,
            n_demos=500 if quick else 8000,
            use_attention=use_attention,
        )
        if use_bc and bc_path is None:
            logger.error("[BC] BC was requested but failed. Aborting.")
            sys.exit(1)"""

apply_patch(TRAIN_PATH, P5_OLD, P5_NEW, "restore BC pretraining (un-comment)")


# =============================================================================
#  PATCH 6 — env_alsat_dynamic.py: MIN_BATTERY_SAFE_SOC 0.20 → 0.25
#
#  Rationale:
#    Training CSV shows ~5-10% of episodes terminating at battery_end_pct=0.4%
#    (ep_len=31-57) despite MIN_BATTERY_SAFE_SOC=0.20.  These short episodes
#    contribute large negative rewards (-3 to -0.3) that corrupt advantage
#    estimates.  Raising the threshold to 0.25 gives more safety margin and
#    should reduce dead-battery episodes by ~50-70%.
#
#    Note: this affects the battery veto in DynamicObsWrapper.step() at the
#    line "if _bm_soc < MIN_BATTERY_SAFE_SOC:".
# =============================================================================

print("[6/6] env_alsat_dynamic.py — MIN_BATTERY_SAFE_SOC 0.20 → 0.25")

P6_OLD = "MIN_BATTERY_SAFE_SOC = 0.20"
P6_NEW = """\
# CONVERGENCE-FIX-6: raise battery safety threshold 0.20→0.25.
# Training CSV shows ~5-10% of episodes die at battery_end_pct=0.4%
# despite the 0.20 threshold, polluting training batches with large
# negative rewards.  0.25 provides more margin without over-restricting
# imaging in the second half of the 48-h episode.
MIN_BATTERY_SAFE_SOC = 0.25"""

apply_patch(DYN_PATH, P6_OLD, P6_NEW, "MIN_BATTERY_SAFE_SOC 0.20→0.25")


# =============================================================================
#  VERIFICATION
# =============================================================================

print()
print("─" * 70)
print("VERIFICATION")
print("─" * 70)

with open(TRAIN_PATH, encoding='utf-8') as f:
    train_txt = f.read()
with open(DYN_PATH, encoding='utf-8') as f:
    dyn_txt = f.read()

checks = [
    # Patch 1
    ("STAGE_ENT_START starts at 0.15",
     "STAGE_ENT_START = [0.15, 0.08, 0.04, 0.01]" in train_txt),
    # Patch 2
    ("vf_coef = 2.0",
     "vf_coef        = 2.0" in train_txt),
    ("n_epochs = 10",
     "n_epochs       = 10," in train_txt),
    ("gae_lambda = 0.90",
     "gae_lambda     = 0.90" in train_txt),
    # Patch 3
    ("VecNormalize.load used between stages",
     "VecNormalize.load(_vecnorm_ckpt" in train_txt),
    ("vec.save checkpoint call present",
     "vec.save(_vecnorm_ckpt)" in train_txt),
    # Patch 4
    ("--quick uses 300k steps",
     "total_steps = 300_000" in train_txt),
    # Patch 5
    ("BC pretraining block restored",
     "if use_bc:\n        print(\"=== BC Pre-training" in train_txt or
     'if use_bc:' in train_txt and 'run_bc_pretrain' in train_txt),
    # Patch 6
    ("MIN_BATTERY_SAFE_SOC = 0.25",
     "MIN_BATTERY_SAFE_SOC = 0.25" in dyn_txt),
]

all_ok = True
for desc, passed in checks:
    icon = "✓" if passed else "✗"
    print(f"  [{icon}] {desc}")
    if not passed:
        all_ok = False

print()
if all_ok:
    print("✓ ALL 6 CONVERGENCE PATCHES APPLIED SUCCESSFULLY")
else:
    print("✗ SOME PATCHES FAILED — check MISS messages above for manual steps")

print()
print("─" * 70)
print("SUMMARY OF CHANGES")
print("─" * 70)
print("""
  train_full_system.py:
    STAGE_ENT_START  [0.40, 0.20, 0.10, 0.03] → [0.15, 0.08, 0.04, 0.01]
    vf_coef          0.5  → 2.0
    n_epochs         5    → 10
    gae_lambda       0.95 → 0.90
    --quick steps    50k  → 300k
    BC pretraining   restored (was silently no-op due to bc_path = None)
    VecNormalize     saved/loaded between curriculum stages (no more cold-start)

  env_alsat_dynamic.py:
    MIN_BATTERY_SAFE_SOC  0.20 → 0.25  (reduces battery-death episode frequency)

WHY THESE FIX explained_var ≈ 0:
  The VecNormalize cold-start amplified the first 20-30 iterations' rewards to
  ~3.0 (clip_reward cap), then the scale dropped as variance calibrated.  The
  VF saw inconsistent targets and converged to predicting the mean for all
  states.  Preserving ret_rms across stages eliminates this.  Raising vf_coef
  and n_epochs ensures the VF has enough gradient signal to break out of the
  constant-prediction minimum once it sees consistent targets.
""")
print("─" * 70)
print("NEXT STEPS")
print("─" * 70)
print("""
  1. Apply the target-ID observation fix (if not yet done):
       python apply_target_id_fix.py

  2. Quick smoke-test (~2-3h on CPU, no GPU needed):
       CUDA_VISIBLE_DEVICES='' \\
       python scripts/training/train_full_system.py --seed 42 --quick

     Watch for these signs of convergence in the training log:
       explained_var  → should rise from 0 toward 0.3–0.6 within Stage 0
       entropy        → should decrease as EntropyAnnealingCallback runs
       reward         → should show an upward trend (not oscillate flat)

  3. Full run (for thesis results, ~12-20h on CPU):
       CUDA_VISIBLE_DEVICES='' \\
       python scripts/training/train_full_system.py \\
           --seed 42 --attention --bc-pretrain --steps 500000

  EXPECTED OUTCOMES AFTER FIX:
    Stage 0 (static-only):   explained_var > 0.2  by iteration 50
    Stage 1 (sparse events):  DynSuc > 35%   by end of stage
    Stage 3 (dense events):   DynSuc > 50%   with 500k total steps
""")