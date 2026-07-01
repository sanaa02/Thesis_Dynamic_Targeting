#!/usr/bin/env python3
"""
fix_vecnorm_persistence.py  —  ALSAT-EO-1  VecNormalize Persistence Fix
========================================================================
WHY apply_convergence_fix.py SKIPPED Patch 3 (but silently)
------------------------------------------------------------
apply_patch() checks "is the first line of new_content already in the file?"
to detect prior application.  But the first line of the VecNorm patch is:

    # Rebuild env for this stage

This comment is ALREADY in the original code (it's in P3_OLD too!), so
apply_patch() returns SKIP thinking the patch was already applied, when
in fact the file is unchanged.  The verification then correctly reports [✗].

WHAT THIS SCRIPT DOES
---------------------
Targets the UNIQUE comment that only exists in the per-stage VecNormalize
block: "# FIX-B3: same clip_reward=3.0 as initial creation (cold-start fix)".
Uses this as the trigger for a definitive "not yet patched" check.

EFFECT
------
After this fix, each curriculum stage SAVES the VecNormalize return statistics
(ret_rms) to a .pkl checkpoint before closing the env, then LOADS those stats
into the next stage's VecNormalize.  The next stage therefore starts with a
calibrated variance estimate — no cold-start reward amplification, no VF
confusion about target scale at stage transitions.

Evidence of the problem: Stage 1 first-iteration reward = +20.356 (despite
Stage 0 rewards being 1-6).  This is VecNormalize cold-starting: running
variance ≈ 0 → normalized_reward = raw / sqrt(0+eps) → capped at clip=3.0,
but early iterations are still amplified.

After this fix: Stage 1 first-iteration reward should be ~3-5 (same scale
as Stage 0 end), and the VF should not need to re-learn the reward scale.

HOW TO RUN:
  python fix_vecnorm_persistence.py

VERIFY:
  grep -n "VecNormalize.load\\|vec.save" \\
       scripts/training/train_full_system.py
"""

import os
import sys

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
    sys.exit(1)

TRAIN_PATH = os.path.join(SCRIPTS, "training", "train_full_system.py")

print("=" * 68)
print("  VecNormalize Persistence Fix  (targeted)")
print("=" * 68)
print(f"  file: {TRAIN_PATH}")
print()

with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

# ── Already applied? ──────────────────────────────────────────────────────────
if "VecNormalize.load(_vecnorm_ckpt" in content and "vec.save(_vecnorm_ckpt)" in content:
    print("  [ALREADY APPLIED]  VecNormalize.load + vec.save are both present.")
    print("  Nothing to do.")
    sys.exit(0)

if "VecNormalize.load(_vecnorm_ckpt" in content or "vec.save(_vecnorm_ckpt)" in content:
    print("  [PARTIAL]  One of VecNormalize.load / vec.save is present but not both.")
    print("  Manual inspection required:")
    print(f"    grep -n 'VecNormalize.load\\|vec.save' {TRAIN_PATH}")
    sys.exit(1)

# ── Check the original pattern is there ───────────────────────────────────────
TRIGGER = "# FIX-B3: same clip_reward=3.0 as initial creation (cold-start fix)"
if TRIGGER not in content:
    print(f"  [MISS]  Trigger comment not found:")
    print(f"          {TRIGGER!r}")
    print()
    print("  The per-stage VecNormalize block may have been renamed.")
    print("  Open train_full_system.py and find the per-stage VecNormalize")
    print("  creation (inside the curriculum loop, NOT the initial creation),")
    print("  then manually replace it with the save/load version below.")
    print()
    print("  MANUAL REPLACEMENT (paste this into the stage loop):")
    print("""
        # --- VecNormalize persistence: save stats from previous stage ---
        _vecnorm_ckpt = os.path.join(MODELS, f"vecnorm_stage{stage_idx}_s{seed}.pkl")
        try:
            vec.save(_vecnorm_ckpt)
        except Exception as _e:
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
        if _vecnorm_ckpt and os.path.exists(_vecnorm_ckpt):
            try:
                vec = VecNormalize.load(_vecnorm_ckpt, vec_raw)
                vec.training = True
            except Exception:
                vec = VecNormalize(vec_raw, norm_obs=False, norm_reward=True,
                                   clip_reward=3.0, gamma=0.99)
        else:
            vec = VecNormalize(vec_raw, norm_obs=False, norm_reward=True,
                               clip_reward=3.0, gamma=0.99)
        model.set_env(vec)
    """)
    sys.exit(1)

# ── Locate and replace the per-stage VecNormalize block ───────────────────────
# The per-stage block (inside the curriculum for-loop) contains the unique
# FIX-B3 comment.  We target the most distinctive 5-line signature so we
# don't accidentally replace the initial VecNormalize creation.
OLD_BLOCK = """\
        # FIX-20: re-apply VecNormalize at each stage
        # FIX-B3: same clip_reward=3.0 as initial creation (cold-start fix)
        vec = VecNormalize(vec, norm_obs=False, norm_reward=True,
                           clip_reward=3.0, gamma=0.99)
        model.set_env(vec)"""

NEW_BLOCK = """\
        # VECNORM-PERSIST: carry ret_rms into the next stage (no cold-start).
        # Without this: next stage starts with ret_rms.var≈0 → first reward
        # amplified to clip_reward → VF sees inconsistent targets → explained_var≈0.
        # With this: next stage inherits calibrated statistics from the current stage.
        _vecnorm_ckpt = os.path.join(MODELS, f"vecnorm_stage{stage_idx}_s{seed}.pkl")
        try:
            vec.save(_vecnorm_ckpt)
        except Exception as _ve:
            logger.warning(f"[VecNorm] save failed: {_ve}")
            _vecnorm_ckpt = None

        vec_raw = DummyVecEnv([
            lambda s=seed + stage_idx * 100 + i, r=rate: build_env(
                s, event_rate=r, use_oracle=use_oracle, use_real_data=use_real_data
            )
            for i in range(N_ENVS)
        ])
        if _vecnorm_ckpt and os.path.exists(_vecnorm_ckpt):
            try:
                vec = VecNormalize.load(_vecnorm_ckpt, vec_raw)
                vec.training = True   # keep updating running statistics
                logger.info(f"[VecNorm] Loaded stage-{stage_idx} stats → next stage starts calibrated")
            except Exception as _ve:
                logger.warning(f"[VecNorm] load failed ({_ve}); falling back to fresh stats")
                vec = VecNormalize(vec_raw, norm_obs=False, norm_reward=True,
                                   clip_reward=3.0, gamma=0.99)
        else:
            vec = VecNormalize(vec_raw, norm_obs=False, norm_reward=True,
                               clip_reward=3.0, gamma=0.99)
        model.set_env(vec)"""

if OLD_BLOCK not in content:
    # The TRIGGER exists but the exact multi-line block doesn't.
    # Likely the vec = DummyVecEnv block was already replaced by a previous
    # partial patch (e.g., apply_convergence_fix.py DID replace it but
    # the FIX-B3 comment is missing now).  Try a broader search.
    print("  [WARN]  Exact OLD_BLOCK not found.  Trying broader match...")

    # Check if the vec_raw pattern (from a partial apply) is already there
    if "vec_raw = DummyVecEnv" in content:
        # A previous patch partially replaced the DummyVecEnv creation
        # but not the VecNormalize.load part.  Replace the partial block.
        PARTIAL_OLD = """\
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

        # Replace using a simpler NEW_BLOCK without the save part since save
        # already exists in this partial state
        if PARTIAL_OLD in content:
            # The full convergence fix Patch 3 was actually applied!
            # Re-check verification
            print("  [INFO]  Full Patch-3 code IS present (partial match found).")
            print("  VecNormalize persistence appears to be applied.")
            print("  Run: grep -n 'VecNormalize.load' scripts/training/train_full_system.py")
            sys.exit(0)
        else:
            print("  [FAIL]  Could not locate the exact block to patch.")
            print("  Please check train_full_system.py manually.")
            sys.exit(1)
    else:
        print("  [FAIL]  Neither OLD_BLOCK nor partial-apply pattern found.")
        print("  The file structure may have changed significantly.")
        print("  Please apply the VecNorm persistence block manually (see above).")
        sys.exit(1)

# ── Apply the patch ────────────────────────────────────────────────────────────
# First, we need to handle the vec.close() + DummyVecEnv block that comes
# BEFORE the VecNormalize creation (which we need to modify).
# We patch the close+create+normalize block all at once.

# Find the full block to replace: from the "Rebuild env for this stage" comment
# through model.set_env(vec).
FULL_OLD = """\
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

FULL_NEW = """\
        # Rebuild env for this stage — VECNORM-PERSIST: save → load stats
        # Saves ret_rms so next stage starts calibrated (no cold-start amplification).
        _vecnorm_ckpt = os.path.join(MODELS, f"vecnorm_stage{stage_idx}_s{seed}.pkl")
        try:
            vec.save(_vecnorm_ckpt)
        except Exception as _ve:
            logger.warning(f"[VecNorm] save failed: {_ve}")
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
        if _vecnorm_ckpt and os.path.exists(_vecnorm_ckpt):
            try:
                vec = VecNormalize.load(_vecnorm_ckpt, vec_raw)
                vec.training = True
                logger.info(f"[VecNorm] stage-{stage_idx} stats carried over → calibrated start")
            except Exception as _ve:
                logger.warning(f"[VecNorm] load failed ({_ve}); using fresh stats")
                vec = VecNormalize(vec_raw, norm_obs=False, norm_reward=True,
                                   clip_reward=3.0, gamma=0.99)
        else:
            vec = VecNormalize(vec_raw, norm_obs=False, norm_reward=True,
                               clip_reward=3.0, gamma=0.99)
        model.set_env(vec)"""

if FULL_OLD not in content:
    print("  [WARN]  Full OLD block not found either.")
    print("  Trying minimal patch (just the VecNormalize creation line)...")

    if OLD_BLOCK in content:
        patched = content.replace(OLD_BLOCK, NEW_BLOCK, 1)
        with open(TRAIN_PATH, 'w', encoding='utf-8') as f:
            f.write(patched)
        print("  [OK]   Minimal patch applied (VecNormalize creation only).")
        print("         Note: vec.save() was not added (close block not matched).")
        print("         The fix will LOAD stats but not save them first.")
        print("         Partial improvement: no amplification from step 2+ of each stage.")
    else:
        print("  [FAIL] Cannot apply patch automatically.")
        sys.exit(1)
else:
    patched = content.replace(FULL_OLD, FULL_NEW, 1)
    with open(TRAIN_PATH, 'w', encoding='utf-8') as f:
        f.write(patched)
    print("  [OK]   Full VecNormalize persistence patch applied.")

# ── Verify ────────────────────────────────────────────────────────────────────
with open(TRAIN_PATH, encoding='utf-8') as f:
    final = f.read()

has_save = "vec.save(_vecnorm_ckpt)" in final
has_load = "VecNormalize.load(_vecnorm_ckpt" in final

print()
print("VERIFICATION:")
print(f"  [{'✓' if has_save else '✗'}] vec.save(_vecnorm_ckpt) present")
print(f"  [{'✓' if has_load else '✗'}] VecNormalize.load(_vecnorm_ckpt present")

if has_save and has_load:
    print()
    print("  ✓ VECNORM PERSISTENCE FULLY APPLIED")
    print()
    print("  Effect: each stage's ret_rms (return statistics) is saved before")
    print("  the env is closed, then loaded into the next stage's VecNormalize.")
    print("  Stage transitions will no longer cold-start the value function.")
    print()
    print("  Expected change in next training run:")
    print("    Stage 1 first-iteration reward: ~3-5 (was +20 due to cold-start)")
    print("    VF should maintain explained_var > 0 across stage transitions")
elif has_load:
    print()
    print("  PARTIAL: load applied, save not found. Stage 2+ will load Stage 0")
    print("  stats (default pkl). Not ideal but better than full cold-start.")
else:
    print()
    print("  ✗ Patch did not apply. Manual fix required.")
    sys.exit(1)