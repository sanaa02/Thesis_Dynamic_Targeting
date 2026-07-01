#!/usr/bin/env python3
"""
apply_target_id_fix.py  —  ALSAT-EO-1  Target-ID + SMDP Verification Fix
=========================================================================
Addresses the expert's two concerns:

  CONCERN 1 — "Does the agent act blind for 14 wilayas?"
  FIX: Add normalized target_id (action_idx / 20.0) as the 6th feature
  in each of the 6 static target slots. Obs: 56 → 62 dims.
  The policy can now learn "if slot_i has target_id = 0.70, output action 14."

  CONCERN 2 — "How is PPO used with SMDP?"
  STATUS: Already correctly implemented in DynamicObsWrapper.step():
    total_r += (gamma^(30/1200))^i * r_i  (proper semi-Markov discount)
  No code change needed for SMDP.

FILES MODIFIED:
  scripts/core/env_alsat_dynamic.py  — _build_obs(), OBS_TOTAL_DIM
  scripts/models/attention_policy.py — _N_TF, _IDX_TARGET_END, _IDX_DYN_END,
                                       _IDX_SOJOURN_END

HOW TO RUN (from project root):
  python apply_target_id_fix.py

HOW TO VERIFY:
  python -c "
  import sys; sys.path.insert(0,'scripts'); sys.path.insert(0,'scripts/core')
  from env_alsat_dynamic import OBS_TOTAL_DIM
  assert OBS_TOTAL_DIM == 62, f'Expected 62, got {OBS_TOTAL_DIM}'
  print('OK: OBS_TOTAL_DIM =', OBS_TOTAL_DIM)
  "

THEN RETRAIN:
  CUDA_VISIBLE_DEVICES='' python scripts/training/train_full_system.py \\
      --seed 42 --attention --bc-pretrain
  (remove --quick for the 300k run)
"""

import os
import sys

# ── Locate project root ────────────────────────────────────────────────────────

def find_scripts_root():
    candidates = [
        os.path.join(os.getcwd(), "scripts"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"),
        os.getcwd(),
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "core", "env_alsat_dynamic.py")):
            return c
    return None

SCRIPTS = find_scripts_root()
if SCRIPTS is None:
    print("ERROR: Cannot find scripts/core/env_alsat_dynamic.py")
    print("       Run this script from your project root directory.")
    sys.exit(1)

DYN_PATH  = os.path.join(SCRIPTS, "core",   "env_alsat_dynamic.py")
ATTN_PATH = os.path.join(SCRIPTS, "models", "attention_policy.py")

print("=" * 65)
print("  ALSAT-EO-1  Target-ID Observation Fix")
print("=" * 65)
print(f"  scripts/  : {SCRIPTS}")
print()


# ── Helper ────────────────────────────────────────────────────────────────────

def apply_patch(path, old, new, tag):
    fname = os.path.basename(path)
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    if new.strip() in content:
        print(f"  [SKIP] {tag}  (already applied in {fname})")
        return True

    if old not in content:
        print(f"  [MISS] {tag}  — pattern not found in {fname}")
        print(f"         Apply manually: find the line containing:")
        first_line = old.strip().splitlines()[0][:80]
        print(f"           {first_line!r}")
        return False

    content = content.replace(old, new, 1)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  [OK]   {tag}")
    return True


# =============================================================================
#  PATCH 1 — env_alsat_dynamic.py: OBS_TOTAL_DIM
# =============================================================================

print("[1/5] env_alsat_dynamic.py — OBS_TOTAL_DIM (56 → 62)")

P1_OLD = "OBS_TOTAL_DIM     = OBS_BASE_DIM + OBS_DYN_DIM + OBS_SOJOURN_DIM  # 56"
P1_NEW = """\
OBS_TARGET_ID_DIM = 6                                          # one ID per static slot
OBS_TOTAL_DIM     = OBS_BASE_DIM + OBS_DYN_DIM + OBS_SOJOURN_DIM + OBS_TARGET_ID_DIM  # 62"""

apply_patch(DYN_PATH, P1_OLD, P1_NEW, "OBS_TOTAL_DIM 56→62")


# =============================================================================
#  PATCH 2 — env_alsat_dynamic.py: _build_obs
# =============================================================================

print("[2/5] env_alsat_dynamic.py — _build_obs (inject target IDs)")

P2_OLD = '''\
    def _build_obs(self, base_obs: np.ndarray, tau_norm: float) -> np.ndarray:
        try:
            sat   = self.env.unwrapped.satellites[0]
            now   = float(sat.simulator.sim_time)
            slots = self._mgr.get_slots(sat, now)
        except Exception:
            slots = [None] * N_DYN_SLOTS
            sat   = None
            now   = 0.0



        # # ── FIX-09: Eclipse-aware battery update ──────────────────────────


        feats = []
        for evt in slots:
            if evt is None:
                feats.extend([0.0, -1.0, 1.0, 0.0])
            else:
                try:
                    slew = _slew_safe(sat, evt)
                    # [TTA] use Keplerian-predicted access time
                    tta  = _compute_tta(sat, evt, now)
                    feats.extend([
                        float(np.clip(evt.priority,             0.0, 1.0)),
                        float(np.clip(evt.cloud_cover_forecast, 0.0, 1.0)),
                        float(np.clip(tta / TIME_NORM_S,        0.0, 1.0)),
                        float(np.clip(slew / (math.pi / 2),     0.0, 1.0)),
                    ])
                except Exception:
                    feats.extend([0.0, -1.0, 1.0, 0.0])

        dyn_arr     = np.array(feats,      dtype=np.float32)
        sojourn_arr = np.array([np.clip(tau_norm, 0.0, 1.0)], dtype=np.float32)
        # Battery SOC — agent can now reason about energy constraint

        

        return np.concatenate([base_obs.astype(np.float32), dyn_arr, [tau_norm]],dtype=np.float32)'''

P2_NEW = '''\
    def _build_obs(self, base_obs: np.ndarray, tau_norm: float) -> np.ndarray:
        try:
            sat   = self.env.unwrapped.satellites[0]
            now   = float(sat.simulator.sim_time)
            slots = self._mgr.get_slots(sat, now)
        except Exception:
            slots = [None] * N_DYN_SLOTS
            sat   = None
            now   = 0.0

        # ── Dynamic event features ────────────────────────────────────────
        feats = []
        for evt in slots:
            if evt is None:
                feats.extend([0.0, -1.0, 1.0, 0.0])
            else:
                try:
                    slew = _slew_safe(sat, evt)
                    tta  = _compute_tta(sat, evt, now)
                    feats.extend([
                        float(np.clip(evt.priority,             0.0, 1.0)),
                        float(np.clip(evt.cloud_cover_forecast, 0.0, 1.0)),
                        float(np.clip(tta / TIME_NORM_S,        0.0, 1.0)),
                        float(np.clip(slew / (math.pi / 2),     0.0, 1.0)),
                    ])
                except Exception:
                    feats.extend([0.0, -1.0, 1.0, 0.0])

        dyn_arr = np.array(feats, dtype=np.float32)

        # ── TARGET-ID FIX: inject action index into each static slot ─────
        # PROBLEM (raised by RL expert): obs[13:43] has 6 target slots × 5
        # features, but NONE encode WHICH action index (0-19) maps to each slot.
        # The bsk_rl top-6 ranking re-sorts every step, so the policy cannot
        # learn a stable slot→action mapping — it acts partially blind.
        #
        # FIX: append target_idx / N_TARGETS as the 6th feature per slot.
        # Now slot i says "these features belong to action X" — the attention
        # head can learn to score slot i and output action X reliably.
        # Obs: 56 → 62 dims  (6 extra target-ID scalars, one per slot).
        #
        # Reference: Herrmann & Schaub (2023) §4.2 pointer-style action head.
        try:
            all_targets = list(sat.scenario.targets)   # fixed order: index = action
            opps = [o for o in getattr(sat, 'upcoming_opportunities', [])
                    if isinstance(o, dict) and o.get('type') == 'target']
            opps = sorted(opps, key=lambda o: o['window'][0])[:6]

            state_part  = base_obs[:13].astype(np.float32)    # satellite state (unchanged)
            target_part = base_obs[13:43].astype(np.float32)  # 6×5 from bsk_rl

            new_target_feats = []
            for slot_i in range(6):
                slot_feats = target_part[slot_i * 5: slot_i * 5 + 5]
                if slot_i < len(opps):
                    tgt = opps[slot_i].get('object', None)
                    try:
                        idx = all_targets.index(tgt)
                    except (ValueError, AttributeError):
                        idx = -1
                    tid = float(idx) / float(N_STATIC_TARGETS)   # ∈ [0, 0.95]
                else:
                    tid = -1.0 / float(N_STATIC_TARGETS)         # sentinel: no target
                new_target_feats.extend(slot_feats.tolist())
                new_target_feats.append(tid)

            # extended_base: 13 (state) + 36 (6 slots × 6 feats) = 49
            extended_base = np.concatenate([
                state_part,
                np.array(new_target_feats, dtype=np.float32),
            ])
        except Exception:
            # Fallback: pad base_obs with zeros to keep total dim consistent
            extended_base = np.concatenate([
                base_obs.astype(np.float32),
                np.zeros(OBS_TARGET_ID_DIM, dtype=np.float32),
            ])

        # Final obs: 49 (extended_base) + 12 (dyn) + 1 (sojourn) = 62
        return np.concatenate([extended_base, dyn_arr, [np.clip(tau_norm, 0.0, 1.0)]],
                               dtype=np.float32)'''

apply_patch(DYN_PATH, P2_OLD, P2_NEW, "_build_obs target-ID injection")


# =============================================================================
#  PATCH 3 — attention_policy.py: _N_TF (5 → 6)
# =============================================================================

print("[3/5] attention_policy.py — _N_TF (5 → 6)")

P3_OLD = "_N_TF      = 5    # props per static slot (priority,cloud,std,opp_open,slew)"
P3_NEW = "_N_TF      = 6    # props per static slot (priority,cloud,std,opp_open,slew,target_id)"

apply_patch(ATTN_PATH, P3_OLD, P3_NEW, "_N_TF 5→6")


# =============================================================================
#  PATCH 4 — attention_policy.py: _IDX_TARGET_END (43 → 49)
# =============================================================================

print("[4/5] attention_policy.py — _IDX_TARGET_END (43 → 49)")

P4_OLD = "_IDX_TARGET_END  = _IDX_STATE_END + _N_TS * _N_TF   # 43"
P4_NEW = "_IDX_TARGET_END  = _IDX_STATE_END + _N_TS * _N_TF   # 49  (13 + 6×6)"

apply_patch(ATTN_PATH, P4_OLD, P4_NEW, "_IDX_TARGET_END 43→49")


# =============================================================================
#  PATCH 5 — attention_policy.py: _IDX_DYN_END + _IDX_SOJOURN_END
# =============================================================================

print("[5/5] attention_policy.py — _IDX_DYN_END (55→61)  _IDX_SOJOURN_END (56→62)")

P5_OLD = """\
_IDX_DYN_END     = _IDX_TARGET_END + _N_DS * _N_DF  # 55
_IDX_SOJOURN_END = _IDX_DYN_END + _N_SOJOURN        # 56"""
P5_NEW = """\
_IDX_DYN_END     = _IDX_TARGET_END + _N_DS * _N_DF  # 61  (49 + 12)
_IDX_SOJOURN_END = _IDX_DYN_END + _N_SOJOURN        # 62"""

apply_patch(ATTN_PATH, P5_OLD, P5_NEW, "_IDX_DYN_END 55→61, _IDX_SOJOURN_END 56→62")


# =============================================================================
#  VERIFY
# =============================================================================

print()
print("─" * 65)
print("VERIFICATION")
print("─" * 65)

with open(DYN_PATH,  encoding='utf-8') as f: dyn_txt  = f.read()
with open(ATTN_PATH, encoding='utf-8') as f: attn_txt = f.read()

all_ok = True
results = [
    ("OBS_TARGET_ID_DIM = 6 defined",       "OBS_TARGET_ID_DIM = 6"    in dyn_txt),
    ("TARGET-ID FIX comment in _build_obs", "TARGET-ID FIX"            in dyn_txt),
    ("extended_base 49 dims comment",       "49 (state) + 36"          in dyn_txt or "49 (extended_base)" in dyn_txt),
    ("_N_TF = 6 in attention_policy",       "_N_TF      = 6"           in attn_txt),
    ("_IDX_TARGET_END = 49 comment",        "# 49"                     in attn_txt),
    ("_IDX_SOJOURN_END = 62 comment",       "# 62"                     in attn_txt),
]

for desc, passed in results:
    icon = "✓" if passed else "✗"
    print(f"  [{icon}] {desc}")
    if not passed:
        all_ok = False

print()
if all_ok:
    print("✓ ALL PATCHES APPLIED SUCCESSFULLY")
    print()
    print("  Obs shape:  56 → 62  (6 extra target-ID scalars, one per slot)")
    print("  Attn input: 6 tokens × 6 features instead of 6 × 5")
    print()
    print("  SMDP STATUS: Already correct (no fix needed).")
    print("    gamma_sub = 0.99^(30/1200) per sub-step — textbook SMDP discount.")
    print()
    print("  NEXT: start the full training run (remove --quick):")
    print()
    print("    CUDA_VISIBLE_DEVICES='' python scripts/training/train_full_system.py \\")
    print("        --seed 42 --attention --bc-pretrain")
    print()
    print("  This will re-run BC pretraining with the new 62-dim obs,")
    print("  then curriculum train for 300k steps (~9 hours on CPU).")
else:
    print("✗ SOME PATCHES FAILED — see MISS messages above.")
    print()
    print("  For any MISS, find the indicated line in the file")
    print("  and apply the change manually (details in the script comments).")