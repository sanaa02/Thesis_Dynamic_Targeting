#!/usr/bin/env python3
"""
fix_stage0_entropy.py  —  ALSAT-EO-1  Per-Stage Entropy Floor Fix
==================================================================
PROBLEM (observed in training log):
  Stage 0 uses EntropyAnnealingCallback(start=0.15, end=ENT_END=0.01).
  By step 9,504 (only 32% of Stage 0), entropy_loss has dropped to -0.006,
  meaning raw policy entropy ≈ 0.6 nats (~1.8 effective actions from 6 valid).
  This causes reward=0.000 in multiple Stage 0 episodes — the agent is
  fully greedy and selects DRIFT for the entire episode.

ROOT CAUSE:
  ENT_END = 0.01 is the floor for ALL stages.  In Stage 0 (static-only,
  only 30k steps), the policy reaches ent_coef=0.01 too quickly and the
  greedy DRIFT bias takes over.  The entropy is so low that the policy
  can't explore static targets at all.

FIX:
  Replace the global ENT_END = 0.01 with per-stage end values:
    STAGE_ENT_END = [0.08, 0.04, 0.02, 0.01]

  Stage 0 now floors at 0.08 (still exploratory), Stage 3 still reaches 0.01.
  The EntropyAnnealingCallback's `end_val` is passed per-stage from
  STAGE_ENT_END[stage_idx] instead of the global ENT_END.

ALSO FIXES:
  The per-stage end values ensure that entering Stage 1 with ent_coef=0.08
  (start) makes sense since Stage 0 ended at 0.08 (same value) — smooth
  curriculum transition rather than jumping from 0.01 back to 0.08.

IMPACT ON TRAINING LOG (expected):
  Stage 0 before: reward=0.000 at steps 4k, 7k, 9k (greedy drift)
  Stage 0 after:  all episodes should have reward > 0 (policy stays exploratory)

FILES MODIFIED:
  scripts/training/train_full_system.py

HOW TO RUN:
  python fix_stage0_entropy.py

VERIFY:
  grep -n "STAGE_ENT_END\\|end_val" scripts/training/train_full_system.py
"""

import os
import sys

def find_scripts_root():
    for c in [
        os.path.join(os.getcwd(), "scripts"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"),
    ]:
        if os.path.isfile(os.path.join(c, "training", "train_full_system.py")):
            return c
    return None

SCRIPTS = find_scripts_root()
if SCRIPTS is None:
    print("ERROR: Cannot find scripts/training/train_full_system.py"); sys.exit(1)

TRAIN_PATH = os.path.join(SCRIPTS, "training", "train_full_system.py")

print("=" * 68)
print("  Stage-0 Entropy Floor Fix  (per-stage ENT_END)")
print("=" * 68)
print(f"  file: {TRAIN_PATH}")
print()

with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

# ── Already applied? ──────────────────────────────────────────────────────────
if "STAGE_ENT_END" in content:
    print("  [ALREADY APPLIED]  STAGE_ENT_END is already defined.")
    # Check the callback call uses it
    if "end_val         = STAGE_ENT_END[stage_idx]" in content or \
       "end_val=STAGE_ENT_END[stage_idx]" in content:
        print("  EntropyAnnealingCallback uses STAGE_ENT_END[stage_idx].")
        print("  Nothing to do.")
    else:
        print("  WARNING: STAGE_ENT_END is defined but the callback may not use it.")
        print("  Check:")
        print("    grep -n 'STAGE_ENT_END\\|end_val' " + TRAIN_PATH)
    sys.exit(0)

# ── Patch 1: Add STAGE_ENT_END after ENT_END ─────────────────────────────────
OLD_ENT_END = "ENT_END         = 0.01   # was 0.005; floor to prevent full determinism"
NEW_ENT_END = """\
ENT_END         = 0.01   # was 0.005; floor to prevent full determinism
# STAGE0-ENTROPY-FIX: per-stage entropy floor.
# Global ENT_END=0.01 causes Stage 0 to go fully greedy by step 9k (32% through
# Stage 0 of 30k steps), producing reward=0.000 episodes (agent picks DRIFT all ep).
# Solution: each stage has its own floor.  Stage 0 stays exploratory (0.08);
# Stage 3 reaches near-deterministic (0.01) after 180k steps of learning.
#   Stage 0 (static-only, 30k steps):    0.15 → 0.08  (always exploring)
#   Stage 1 (sparse events, 45k steps):  0.08 → 0.04
#   Stage 2 (mid events,   45k steps):   0.04 → 0.02
#   Stage 3 (dense events, 180k steps):  0.01 → 0.01  (converge to near-greedy)
STAGE_ENT_END   = [0.08, 0.04, 0.02, 0.01]"""

if OLD_ENT_END not in content:
    # Try alternate wording
    OLD_ENT_END_ALT = "ENT_END         = 0.01"
    if OLD_ENT_END_ALT in content:
        print("  [INFO] Using alternate ENT_END pattern match.")
        # Find and insert after the line
        content = content.replace(
            OLD_ENT_END_ALT,
            OLD_ENT_END_ALT + "\n" + """\
# STAGE0-ENTROPY-FIX: per-stage entropy floor (see fix_stage0_entropy.py)
STAGE_ENT_END   = [0.08, 0.04, 0.02, 0.01]""",
            1
        )
        print("  [OK]   STAGE_ENT_END inserted after ENT_END")
    else:
        print("  [MISS] ENT_END line not found.")
        print("         Manually add after the ENT_END line:")
        print('         STAGE_ENT_END = [0.08, 0.04, 0.02, 0.01]')
        sys.exit(1)
else:
    content = content.replace(OLD_ENT_END, NEW_ENT_END, 1)
    print("  [OK]   STAGE_ENT_END added after ENT_END")

# ── Patch 2: Update the EntropyAnnealingCallback call in the curriculum loop ──
# Original:
#   callbacks.append(EntropyAnnealingCallback(
#       start_val       = stage_ent,
#       end_val         = ENT_END,
#       total_timesteps = n_stage,
#       verbose         = 1,
#   ))
OLD_CB = """\
        callbacks.append(EntropyAnnealingCallback(
            start_val       = stage_ent,
            end_val         = ENT_END,
            total_timesteps = n_stage,
            verbose         = 1,
        ))"""

NEW_CB = """\
        # STAGE0-ENTROPY-FIX: use per-stage end_val (not global ENT_END)
        _stage_ent_end = STAGE_ENT_END[stage_idx] if stage_idx < len(STAGE_ENT_END) else ENT_END
        callbacks.append(EntropyAnnealingCallback(
            start_val       = stage_ent,
            end_val         = _stage_ent_end,   # was ENT_END (global 0.01)
            total_timesteps = n_stage,
            verbose         = 1,
        ))"""

if OLD_CB not in content:
    # Try with verbose=0
    OLD_CB_V0 = """\
        callbacks.append(EntropyAnnealingCallback(
            start_val       = stage_ent,
            end_val         = ENT_END,
            total_timesteps = n_stage,
            verbose         = 0,
        ))"""
    NEW_CB_V0 = NEW_CB.replace("verbose         = 1,", "verbose         = 0,")
    if OLD_CB_V0 in content:
        content = content.replace(OLD_CB_V0, NEW_CB_V0, 1)
        print("  [OK]   EntropyAnnealingCallback updated to use STAGE_ENT_END[stage_idx]")
    else:
        print("  [MISS] EntropyAnnealingCallback call not found in expected format.")
        print("         Manually change:")
        print('           end_val = ENT_END')
        print("         to:")
        print('           end_val = STAGE_ENT_END[stage_idx] if stage_idx < len(STAGE_ENT_END) else ENT_END')
else:
    content = content.replace(OLD_CB, NEW_CB, 1)
    print("  [OK]   EntropyAnnealingCallback updated to use STAGE_ENT_END[stage_idx]")

# ── Write ─────────────────────────────────────────────────────────────────────
with open(TRAIN_PATH, 'w', encoding='utf-8') as f:
    f.write(content)

# ── Verify ────────────────────────────────────────────────────────────────────
with open(TRAIN_PATH, encoding='utf-8') as f:
    final = f.read()

has_def  = "STAGE_ENT_END   = [0.08, 0.04, 0.02, 0.01]" in final or \
           "STAGE_ENT_END = [0.08, 0.04, 0.02, 0.01]" in final
has_use  = "STAGE_ENT_END[stage_idx]" in final

print()
print("VERIFICATION:")
print(f"  [{'✓' if has_def else '✗'}] STAGE_ENT_END defined in file")
print(f"  [{'✓' if has_use else '✗'}] STAGE_ENT_END[stage_idx] used in callback")

if has_def and has_use:
    print()
    print("  ✓ STAGE0 ENTROPY FIX APPLIED")
    print()
    print("  Per-stage entropy floors:")
    print("    Stage 0 (static,  30k steps):   0.15 → 0.08  (stays exploratory)")
    print("    Stage 1 (sparse,  45k steps):   0.08 → 0.04")
    print("    Stage 2 (mid,     45k steps):   0.04 → 0.02")
    print("    Stage 3 (dense,  180k steps):   0.01 → 0.01")
    print()
    print("  Expected improvement: no more reward=0.000 episodes in Stage 0.")
    print("  The policy will stay exploratory (1.08 ENT_END → 2.2 effective actions)")
    print("  throughout Stage 0, learning static target selection properly.")
else:
    print()
    print("  ✗ Some patches failed — manual fix required (see MISS messages above).")