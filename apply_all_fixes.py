#!/usr/bin/env python3
"""
APPLY_FIXES_V2.py  —  ALSAT-EO-1  Patch script (run after APPLY_ALL_FIXES.py)
===============================================================================
Run from your project root:
    python attached_assets/alsat_fixes/patches/APPLY_FIXES_V2.py

Fixes applied
-------------
FIX-B2  env_alsat_dynamic.py  — Delayed reward STILL bleeding after FIX-2.
         Root cause (confirmed from decisions.log timing analysis):
         BSK-RL early-terminates a scheduler step when the image-confirmed flag
         fires inside Basilisk. The imaging action's sub-steps (n_sub × 1200s)
         finished WITHOUT seeing the confirmation; sat.current_action_target
         was NOT cleared (only cleared in compare_log_states when flag fires).
         The subsequent DRIFT action's BSK-RL sub-step gets the early-termination
         with flag=True, finds current_action_target still pointing at the imaged
         target, and awards full imaging reward to drift.
         Fix: set sat.current_action_target = None at the START of every
         DRIFT/DYN action, before any sub-steps run.

FIX-B3  train_full_system.py  — VecNormalize cold-start reward amplification.
         After 20 zero-reward drift steps, the running return variance is ≈0.
         The first real reward (raw ≈1.55 for Msila) is normalised to 6.559,
         creating a massive single-sample gradient that collapses the policy
         toward that one target after just 1 rollout.
         Fix: lower clip_reward from 10.0 → 3.0 at both VecNormalize creation
         sites (initial + per-stage). The amplified reward is then capped at 3.0
         rather than 6.5, halving the gradient shock while keeping rewards
         distinguishable.
"""

import os
import sys
import shutil
import datetime

# ── Project root detection ─────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
# _PROJ = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_PROJ = os.getcwd()
# Allow override via env var (same as APPLY_ALL_FIXES.py)
PROJECT_ROOT = os.environ.get("ALSAT_PROJECT_ROOT", _PROJ)

def _find(relative_path):
    full = os.path.join(PROJECT_ROOT, relative_path)
    if os.path.exists(full):
        return full
    # Fallback: search common locations
    for base in [
        os.path.expanduser("~/Pictures/Thesis_Dynamic_Targeting_copy"),
        os.path.expanduser("~/Thesis_Dynamic_Targeting_copy"),
        os.path.expanduser("~/scripts"),
        PROJECT_ROOT,
    ]:
        candidate = os.path.join(base, relative_path)
        if os.path.exists(candidate):
            return candidate
    return None


def _backup(path):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path + f".bak_v2_{ts}"
    shutil.copy2(path, bak)
    print(f"  [backup] {os.path.basename(path)} → {os.path.basename(bak)}")
    return bak


def _apply(path, old, new, label):
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    if old not in src:
        print(f"  [SKIP]   {label} — anchor text not found (already patched or file changed)")
        return False
    if new in src:
        print(f"  [SKIP]   {label} — replacement already present (idempotent)")
        return False
    patched = src.replace(old, new, 1)
    if patched == src:
        print(f"  [SKIP]   {label} — no change after replacement")
        return False
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(patched)
    print(f"  [OK]     {label}")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# FIX-B2 — env_alsat_dynamic.py — clear current_action_target before DRIFT/DYN
# ═══════════════════════════════════════════════════════════════════════════

FIX_B2_OLD = """\
        # [SMDP] compute actual task duration
        try:
            sat      = self.env.unwrapped.satellites[0]
            # Reset DYN imaging flag for new action
            sat._dyn_reward_given = False
            # [FIX-A] For DYN actions: drain the image buffer BEFORE sub-steps
            # so that any image taken for a prior static target doesn't bleed
            # a negative slew-energy penalty into total_r during DYN sub-steps.
            if _N_STATIC <= int(action) < _N_STATIC + N_DYN_SLOTS:
                try:
                    sat.was_image_taken_since_last_check()  # drain buffer, discard result
                except Exception:
                    pass
            tau      = _action_duration(sat, int(action))
        except Exception:
            tau = BASE_STEP_S"""

FIX_B2_NEW = """\
        # [SMDP] compute actual task duration
        try:
            sat      = self.env.unwrapped.satellites[0]
            # Reset DYN imaging flag for new action
            sat._dyn_reward_given = False
            # FIX-B2: Clear stale current_action_target before non-imaging actions.
            # Root cause: BSK-RL early-terminates a scheduler step when the Basilisk
            # image-confirmed flag fires. If the imaging action's sub-steps didn't
            # see the flag (it fired after those sub-steps), current_action_target
            # remains set when the next DRIFT/DYN action's BSK-RL scheduler step
            # runs — causing compare_log_states() to award full imaging reward to
            # a drift step (visible in decisions.log as non-1200s-gap drift steps
            # with positive rewards). Clearing the pointer here prevents this bleed.
            if int(action) >= N_STATIC_TARGETS:
                sat.current_action_target = None
            # [FIX-A] For DYN actions: drain the image buffer BEFORE sub-steps
            # so that any image taken for a prior static target doesn't bleed
            # a negative slew-energy penalty into total_r during DYN sub-steps.
            if _N_STATIC <= int(action) < _N_STATIC + N_DYN_SLOTS:
                try:
                    sat.was_image_taken_since_last_check()  # drain buffer, discard result
                except Exception:
                    pass
            tau      = _action_duration(sat, int(action))
        except Exception:
            tau = BASE_STEP_S"""


# ═══════════════════════════════════════════════════════════════════════════
# FIX-B3 — train_full_system.py — reduce VecNormalize clip_reward 10→3
# Applied at TWO sites: initial creation + per-stage recreation
# ═══════════════════════════════════════════════════════════════════════════

# Site 1: initial VecNormalize creation
FIX_B3_SITE1_OLD = """\
    # FIX-20: VecNormalize for reward normalization
    vec = VecNormalize(vec, norm_obs=False, norm_reward=True,
                       clip_reward=10.0, gamma=0.99)"""

FIX_B3_SITE1_NEW = """\
    # FIX-20: VecNormalize for reward normalization
    # FIX-B3: reduced clip_reward 10→3 to limit cold-start gradient amplification.
    # After many zero-reward drift steps, running variance≈0, so the first real
    # reward is normalised to ~4-7× its raw value. Capping at 3.0 rather than 10.0
    # halves the gradient shock without losing reward differentiation.
    vec = VecNormalize(vec, norm_obs=False, norm_reward=True,
                       clip_reward=3.0, gamma=0.99)"""

# Site 2: per-stage VecNormalize recreation
FIX_B3_SITE2_OLD = """\
        # FIX-20: re-apply VecNormalize at each stage
        vec = VecNormalize(vec, norm_obs=False, norm_reward=True,
                           clip_reward=10.0, gamma=0.99)"""

FIX_B3_SITE2_NEW = """\
        # FIX-20: re-apply VecNormalize at each stage
        # FIX-B3: same clip_reward=3.0 as initial creation (cold-start fix)
        vec = VecNormalize(vec, norm_obs=False, norm_reward=True,
                           clip_reward=3.0, gamma=0.99)"""


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print(f"\nALSAT-EO-1 — APPLY_FIXES_V2.py")
    print(f"Project root: {PROJECT_ROOT}\n")

    # --- env_alsat_dynamic.py ---
    env_path = _find("scripts/core/env_alsat_dynamic.py")
    if env_path is None:
        # Try flat layout
        for name in ["env_alsat_dynamic.py"]:
            env_path = _find(name)
            if env_path:
                break
    if env_path is None:
        print("[ERROR] env_alsat_dynamic.py not found. Set ALSAT_PROJECT_ROOT or run from project root.")
        sys.exit(1)

    print(f"=== {os.path.relpath(env_path, PROJECT_ROOT)} ===")
    _backup(env_path)
    _apply(env_path, FIX_B2_OLD, FIX_B2_NEW, "FIX-B2: clear current_action_target before DRIFT/DYN")

    # --- train_full_system.py ---
    train_path = _find("scripts/training/train_full_system.py")
    if train_path is None:
        train_path = _find("train_full_system.py")
    if train_path is None:
        print("[ERROR] train_full_system.py not found. Set ALSAT_PROJECT_ROOT or run from project root.")
        sys.exit(1)

    print(f"\n=== {os.path.relpath(train_path, PROJECT_ROOT)} ===")
    _backup(train_path)
    _apply(train_path, FIX_B3_SITE1_OLD, FIX_B3_SITE1_NEW, "FIX-B3 site-1: initial VecNormalize clip_reward 10→3")
    _apply(train_path, FIX_B3_SITE2_OLD, FIX_B3_SITE2_NEW, "FIX-B3 site-2: per-stage VecNormalize clip_reward 10→3")

    print("\n✓ All V2 fixes applied. Run training and verify decisions.log:")
    print("  • No more non-1200s-gap drift steps with r>0 (FIX-B2)")
    print("  • First imaging reward ≤3.0 after VecNormalize (FIX-B3)")
    print()
    print("NOTE — battery drain during initial drift steps is CORRECT physics:")
    print("  The satellite starts every episode in eclipse (no solar generation).")
    print("  Battery drains at ~1.8%/step (1200s) during eclipse, then recharges")
    print("  rapidly once the satellite enters the sunlit part of its orbit.")
    print("  This is not a bug — it is ALSAT-EO-1 orbital mechanics.")
    print()


if __name__ == "__main__":
    main()