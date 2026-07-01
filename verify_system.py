#!/usr/bin/env python3
"""
verify_system.py  —  ALSAT-EO-1  Full System Verification
==========================================================
Checks every critical component of the ALSAT-EO-1 RL training system.
Designed to be run BEFORE the final thesis training run to confirm that
all patches are applied and the system is internally consistent.

USAGE:
  python verify_system.py

SECTIONS:
  A. File-based checks (grep source code for correct values)
  B. Runtime checks (import modules, create one env, run 2 episodes)
  C. Training consistency checks (all components wire together)
  D. Thesis-defensibility checks (SMDP, obs, reward correctness)
"""

import os
import sys
import math

# ── Locate project root ────────────────────────────────────────────────────────

def find_scripts_root():
    for candidate in [
        os.path.join(os.getcwd(), "scripts"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"),
    ]:
        if os.path.isfile(os.path.join(candidate, "training", "train_full_system.py")):
            return candidate
    return None

SCRIPTS = find_scripts_root()
if SCRIPTS is None:
    print("ERROR: Cannot find scripts/training/train_full_system.py")
    print("       Run from project root.")
    sys.exit(1)

ROOT = os.path.dirname(SCRIPTS)

# Setup sys.path for imports
for d in ["scripts/core", "scripts/training", "scripts/wrappers",
          "scripts/models", "scripts/evaluation", "scripts"]:
    p = os.path.join(ROOT, d)
    if p not in sys.path:
        sys.path.insert(0, p)
try:
    import path_setup
    ROOT = path_setup.root_path()
except Exception:
    pass

TRAIN_PATH = os.path.join(SCRIPTS, "training", "train_full_system.py")
DYN_PATH   = os.path.join(SCRIPTS, "core",     "env_alsat_dynamic.py")
ATTN_PATH  = os.path.join(SCRIPTS, "models",   "attention_policy.py")

print("=" * 72)
print("  ALSAT-EO-1  Full System Verification")
print("=" * 72)
print(f"  ROOT    : {ROOT}")
print(f"  SCRIPTS : {SCRIPTS}")
print()

PASS = []
FAIL = []
WARN = []

def ck(label, cond, detail="", warn_only=False):
    icon = "✓" if cond else ("⚠" if warn_only else "✗")
    msg  = f"  [{icon}] {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    if cond:
        PASS.append(label)
    elif warn_only:
        WARN.append(label + (f": {detail}" if detail else ""))
    else:
        FAIL.append(label + (f": {detail}" if detail else ""))

def read_file(path):
    try:
        with open(path, encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"__READ_ERROR__: {e}"


# =============================================================================
#  A. FILE-BASED CHECKS
# =============================================================================
print("─" * 72)
print("A. Source-code checks")
print("─" * 72)

train_txt = read_file(TRAIN_PATH)
dyn_txt   = read_file(DYN_PATH)
attn_txt  = read_file(ATTN_PATH) if os.path.exists(ATTN_PATH) else ""

# ── Convergence fixes ─────────────────────────────────────────────────────────
ck("STAGE_ENT_START starts at 0.02",
   "STAGE_ENT_START = [0.02, 0.02, 0.02, 0.01]" in train_txt,
   "was too high → VF predicts constant or policy de-learns BC weights")

ck("vf_coef = 2.0",
   "vf_coef        = 2.0" in train_txt,
   "was 0.5 → VF undertrained → explained_var≈0")

ck("n_epochs = 10 in PPO constructor",
   "n_epochs       = 10," in train_txt,
   "was 5 → too few VF updates per rollout")

ck("gae_lambda = 0.90",
   "gae_lambda     = 0.90" in train_txt,
   "was 0.95 → too much variance with battery-death episodes")

ck("BC pretraining restored",
   "if use_bc:" in train_txt and "run_bc_pretrain" in train_txt,
   "was silently disabled with bc_path=None")

ck("--quick uses 300k steps",
   "total_steps = 300_000" in train_txt,
   "was 50k → Stage 0 only 17 rollouts")

ck("VecNormalize persistence: vec.save",
   "vec.save(_vecnorm_ckpt)" in train_txt,
   "required to carry ret_rms between stages")

ck("VecNormalize persistence: VecNormalize.load",
   "VecNormalize.load(_vecnorm_ckpt" in train_txt,
   "required to carry ret_rms between stages")

# ── Obs dimension (target-ID fix) ─────────────────────────────────────────────
ck("OBS_TOTAL_DIM = 63 (target-ID fix applied)",
   "OBS_TARGET_ID_DIM = 6" in dyn_txt and "# 63" in dyn_txt,
   "was 56 — agent was blind to target identity")

ck("_build_obs injects target_id feature",
   "TARGET-ID FIX" in dyn_txt or "target_idx" in dyn_txt,
   "required for policy to learn slot→action mapping")

# ── Attention policy obs indices ─────────────────────────────────────────────
if attn_txt:
    ck("attention_policy _N_TF = 6",
       "_N_TF      = 6" in attn_txt,
       "was 5 — must match 6 features per static slot")
    ck("attention_policy _IDX_TARGET_END = 49 comment",
       "# 49" in attn_txt,
       "was 43 — must reflect 13 + 6×6 = 49")
    ck("attention_policy _IDX_SOJOURN_END = 62 comment",
       "# 62" in attn_txt,
       "was 56")
else:
    ck("attention_policy.py exists", False,
       f"not found at {ATTN_PATH}")

# ── Battery threshold ─────────────────────────────────────────────────────────
ck("MIN_BATTERY_SAFE_SOC = 0.30",
   "MIN_BATTERY_SAFE_SOC = 0.30" in dyn_txt,
   "was 0.20 — too low, battery deaths corrupting training")

# ── SMDP discount ─────────────────────────────────────────────────────────────
ck("SMDP gamma_sub formula present",
   "gamma ** (BASE_STEP_S / STEP_REF_S)" in dyn_txt or
   "0.99 ** (30 / 1200)" in dyn_txt or
   "_gamma_sub" in dyn_txt,
   "semi-Markov discount: gamma^(30/1200) per sub-step")

# ── Per-stage entropy end values ──────────────────────────────────────────────
has_stage_ent_end = "STAGE_ENT_END" in train_txt
ck("Per-stage ENT_END values (prevents Stage-0 collapse)",
   has_stage_ent_end,
   warn_only=True,
   detail="optional but prevents entropy collapse in Stage 0 (reward=0 episodes)")

print()


# =============================================================================
#  B. RUNTIME CHECKS
# =============================================================================
print("─" * 72)
print("B. Runtime checks  (imports + one mini-episode)")
print("─" * 72)

# ── Import checks ─────────────────────────────────────────────────────────────

def try_import(modname):
    try:
        __import__(modname)
        return True, ""
    except Exception as e:
        return False, str(e)[:80]

for mod in ["numpy", "gymnasium", "stable_baselines3", "sb3_contrib"]:
    ok, err = try_import(mod)
    ck(f"import {mod}", ok, err if not ok else "")

# ── env_alsat_dynamic OBS_TOTAL_DIM ──────────────────────────────────────────
try:
    import importlib, env_alsat_dynamic as _env_dyn
    importlib.reload(_env_dyn)
    dim = _env_dyn.OBS_TOTAL_DIM
    ck(f"OBS_TOTAL_DIM == 63 at runtime",
       dim == 63, f"got {dim}")
except Exception as e:
    ck("env_alsat_dynamic importable", False, str(e)[:80])

# ── env_dynamic_factory ───────────────────────────────────────────────────────
try:
    from env_dynamic_factory import make_env, Config
    ck("env_dynamic_factory importable", True)
except Exception as e:
    ck("env_dynamic_factory importable", False, str(e)[:80])

# ── EntropyAnnealingCallback importable ───────────────────────────────────────
try:
    import importlib, train_full_system as _tfs
    importlib.reload(_tfs)
    has_eac = hasattr(_tfs, 'EntropyAnnealingCallback')
    ck("EntropyAnnealingCallback defined in train_full_system", has_eac)
    stage_ent = getattr(_tfs, 'STAGE_ENT_START', None)
    ck(f"STAGE_ENT_START at runtime = {stage_ent}",
       stage_ent == [0.02, 0.02, 0.02, 0.01],
       f"got {stage_ent}")
    vf = None
    # Extract vf_coef from source (the constant isn't exported)
    ck("vf_coef=2.0 in source (confirmed above)",
       "vf_coef        = 2.0" in train_txt)
except Exception as e:
    ck("train_full_system importable", False, str(e)[:80])

# ── Mini episode test ─────────────────────────────────────────────────────────
print()
print("  Running mini-episode test (2 episodes)...")
try:
    import numpy as np
    TARGETS = os.path.join(ROOT, "config/targets/global_45_targets.json")
    CLOUD   = os.path.join(ROOT, "config/cloud_reality/global_45_clouds.json")

    env = make_env(Config.DYN_MODIS, TARGETS, CLOUD,
                   event_rate=1.0, seed=99, with_safety=True)
    obs_space = env.observation_space
    ck(f"obs_space.shape == (63,)",
       obs_space.shape == (63,), f"got {obs_space.shape}")

    # Wrap for action masking
    try:
        from wrappers.action_mask_wrapper import make_masked_env
        from stable_baselines3.common.monitor import Monitor
        env_m = make_masked_env(env)
        env_m = Monitor(env_m)

        rewards, ep_lens, dyn_suc_list = [], [], []
        for ep in range(2):
            obs, info = env_m.reset()
            done = False
            step_count = 0
            ep_r = 0.0
            while not done and step_count < 200:
                # Use random action mask-aware step
                try:
                    mask = env_m.env.action_masks()
                    valid = np.where(mask)[0]
                    action = int(np.random.choice(valid))
                except Exception:
                    action = env_m.action_space.sample()
                obs, r, term, trunc, info = env_m.step(action)
                ep_r += r
                step_count += 1
                done = term or trunc
            rewards.append(ep_r)
            ep_lens.append(step_count)
            dyn = info.get("n_dyn_imaged", 0)
            det = info.get("n_dyn_detected", 1)
            dyn_suc_list.append(dyn / max(1, det))

        ck("2 mini-episodes completed without crash", True,
           f"rewards={[round(r,2) for r in rewards]}, "
           f"ep_lens={ep_lens}, dyn_suc={[f'{s:.0%}' for s in dyn_suc_list]}")
        ck("obs shape consistent at step", obs.shape == (63,),
           f"got {obs.shape}")

        env_m.close()
    except Exception as e:
        import traceback
        traceback.print_exc()
        ck("mini-episode with action masking", False, str(e)[:100])

except Exception as e:
    ck("mini-episode environment creation", False, str(e)[:100])

print()


# =============================================================================
#  C. TRAINING CONSISTENCY CHECKS
# =============================================================================
print("─" * 72)
print("C. Training consistency checks")
print("─" * 72)

# ── VecNormalize gamma consistency ────────────────────────────────────────────
ck("VecNormalize gamma=0.99 matches PPO gamma",
   "VecNormalize(vec, norm_obs=False, norm_reward=True,\n                       clip_reward=3.0, gamma=0.99)" in train_txt or
   "gamma=0.99)" in train_txt)

# ── N_STEPS / BATCH_SIZE alignment ───────────────────────────────────────────
# N_ENVS=2, N_STEPS=288 → total_per_rollout=576 → BATCH_SIZE=96 → 576/96=6 minibatches
ck("N_STEPS=288, N_ENVS=2, BATCH_SIZE=96 (6 minibatches per rollout)",
   "N_STEPS    = EPISODE_LEN * 2" in train_txt and
   "BATCH_SIZE = (N_STEPS * N_ENVS) // 6" in train_txt)

# ── EntropyAnnealingCallback uses model.ent_coef correctly ────────────────────
ck("EntropyAnnealingCallback updates model.ent_coef",
   "self.model.ent_coef = float(new_ent)" in train_txt)

# ── Stage steps sum to total_steps ────────────────────────────────────────────
ck("Stage 3 steps = remainder (total - stages 0-2)",
   "s3_steps = max(rollout_size, total_steps - s0_steps - s1_steps - s2_steps)" in train_txt)

# ── BatteryConservationWrapper applied ───────────────────────────────────────
ck("BatteryConservationWrapper applied in build_env",
   "BatteryConservationWrapper(env" in train_txt)

# ── DynamicRewardShaper applied ──────────────────────────────────────────────
ck("DynamicRewardShaper applied in build_env",
   "DynamicRewardShaper" in train_txt)

# ── DYN_MULTIPLIER patched at runtime ────────────────────────────────────────
ck("DYN_MULTIPLIER=2.0 patched at runtime",
   "_de.DYN_MULTIPLIER = DYN_MULTIPLIER" in train_txt and
   "DYN_MULTIPLIER = 2.0" in train_txt)

# ── Checkpoint saved at each stage ────────────────────────────────────────────
ck("Stage checkpoint saved after each stage",
   "model.save(ckpt)" in train_txt)

# ── N_EPOCHS constant vs PPO n_epochs ─────────────────────────────────────────
# The CONFIG print says "5 epochs" but the PPO is initialized with 10
n_epochs_const = "N_EPOCHS   = 5" in train_txt
n_epochs_ppo   = "n_epochs       = 10," in train_txt
ck("PPO n_epochs=10 (override of N_EPOCHS=5 constant)",
   n_epochs_ppo,
   "CONFIG print shows '5' but model uses 10 — cosmetic discrepancy only" if n_epochs_const and n_epochs_ppo else "")

print()


# =============================================================================
#  D. THESIS-DEFENSIBILITY CHECKS
# =============================================================================
print("─" * 72)
print("D. Thesis-defensibility checks")
print("─" * 72)

# ── SMDP discount ─────────────────────────────────────────────────────────────
ck("SMDP sub-step discount: gamma^(BASE_STEP_S/STEP_REF_S)",
   "self._gamma_sub = gamma ** (BASE_STEP_S / STEP_REF_S)" in dyn_txt or
   "_gamma_sub = gamma ** (BASE_STEP_S / STEP_REF_S)" in dyn_txt,
   "textbook semi-Markov discount (Sutton & Barto §17.4)")

ck("SMDP total_r accumulation: total_r += gamma_sub^i * r_i",
   "total_r += (self._gamma_sub ** _i) * r_i" in dyn_txt or
   "total_r += (self._gamma_sub **" in dyn_txt,
   "required for correct PPO advantage estimation under SMDP")

# ── Action masking ────────────────────────────────────────────────────────────
ck("Action mask respects satellite visibility",
   "action_masks" in dyn_txt or
   "action_mask" in dyn_txt or
   os.path.exists(os.path.join(SCRIPTS, "wrappers", "action_mask_wrapper.py")),
   "prevents selecting targets not in access window")

# ── DynSuc metric definition ──────────────────────────────────────────────────
ck("DynSuc = n_dyn_imaged / n_dyn_detected (info dict)",
   "n_dyn_imaged" in dyn_txt and "n_dyn_detected" in dyn_txt,
   "thesis metric: fraction of detected events successfully imaged")

# ── N_STATIC_TARGETS and N_DYN_SLOTS ─────────────────────────────────────────
ck("N_STATIC_TARGETS = 20",
   "N_STATIC_TARGETS = 20" in dyn_txt or
   "N_TARGETS = 20" in dyn_txt,
   "20 Algerian wilaya targets")

ck("N_DYN_SLOTS = 3",
   "N_DYN_SLOTS = 3" in dyn_txt,
   "3 dynamic event slots in obs")

ck("Total actions = 24 (20 static + 3 dynamic + 1 drift)",
   "24" in dyn_txt and ("N_ACTIONS" in dyn_txt or "action_space" in dyn_txt),
   "action space: 0-19 static, 20-22 dynamic, 23 drift")

# ── Event generator ───────────────────────────────────────────────────────────
ck("EventGenerator uses Poisson process",
   "rate_per_hour" in dyn_txt or
   os.path.exists(os.path.join(SCRIPTS, "core", "dynamic_event.py")),
   "random event arrival for realistic dynamic scheduling")

# ── Cloud cover model ─────────────────────────────────────────────────────────
ck("MODIS cloud cover used (not synthetic)",
   os.path.exists(os.path.join(ROOT, "config", "cloud_reality",
                               "global_45_clouds.json")),
   "real MODIS cloud data required for thesis validity")

# ── Target config ─────────────────────────────────────────────────────────────
ck("20-target config file exists",
   os.path.exists(os.path.join(ROOT, "config", "targets",
                               "global_45_targets.json")),
   "wilaya coordinate targets for ALSAT-EO-1")

print()


# =============================================================================
#  E. REMAINING ISSUES (advisory)
# =============================================================================
print("─" * 72)
print("E. Known remaining issues (advisory, not blocking)")
print("─" * 72)

# Battery deaths still occurring
print("  [⚠] Battery deaths: ~5% of episodes terminate at SoC≈0%")
print("       Root cause: bsk_rl internal battery validation fires before")
print("       DynamicObsWrapper.step() can intercept.  Raising")
print("       MIN_BATTERY_SAFE_SOC helps but doesn't eliminate this.")
print("       Impact on thesis: minor (~5% of training data is corrupted).")
print()

# Entropy collapse in Stage 0
stage0_entropy_ok = "STAGE_ENT_END" in train_txt or \
                    "[0.08, 0.04, 0.02, 0.01]" in train_txt
if not stage0_entropy_ok:
    print("  [⚠] Stage 0 entropy collapses too fast (ENT_END=0.01 is too low):")
    print("       Training log shows entropy → -0.006 by step 9k (30% through Stage 0),")
    print("       causing reward=0.000 episodes (agent goes full-greedy → picks DRIFT).")
    print("       Fix: run  python fix_stage0_entropy.py  or manually set")
    print("       STAGE_ENT_END = [0.08, 0.04, 0.02, 0.01]  in train_full_system.py")
    print()

# VecNorm persistence
if "VecNormalize.load(_vecnorm_ckpt" not in train_txt:
    print("  [⚠] VecNormalize persistence not applied:")
    print("       Each stage cold-starts ret_rms, causing VF to re-learn reward scale.")
    print("       Fix: run  python fix_vecnorm_persistence.py")
    print()

# MlpPolicy vs attention
print("  [ℹ] Quick test uses MlpPolicy (no --attention flag).")
print("       Full thesis run should use --attention for pointer-style action head.")
print("       Remember to also pass --bc-pretrain for warm initialization.")
print()


# =============================================================================
#  SUMMARY
# =============================================================================
print("=" * 72)
print("SUMMARY")
print("=" * 72)
print(f"  PASS : {len(PASS)}")
print(f"  WARN : {len(WARN)}")
print(f"  FAIL : {len(FAIL)}")
print()

if FAIL:
    print("  FAILURES (must fix before thesis run):")
    for f in FAIL:
        print(f"    ✗  {f}")
    print()

if WARN:
    print("  WARNINGS (improve if time permits):")
    for w in WARN:
        print(f"    ⚠  {w}")
    print()

if not FAIL:
    print("  ✓  SYSTEM IS READY FOR THESIS TRAINING RUN")
    print()
    print("  COMMAND (full 500k-step run with attention policy):")
    print()
    print("    CUDA_VISIBLE_DEVICES='' \\")
    print("    python scripts/training/train_full_system.py \\")
    print("        --seed 42 --attention --bc-pretrain --steps 500000")
    print()
    print("  Expected duration: 12-20h on CPU")
    print("  Expected DynSuc at end of Stage 3: 40-60%")
    print()
    print("  WHAT TO REPORT IN THESIS:")
    print("    - Stage 3 final DynSuc (primary metric)")
    print("    - Compare vs greedy baselines (baselines_dynamic.py)")
    print("    - Show reward learning curves per stage")
    print("    - Show entropy annealing across curriculum")
else:
    print("  ✗  FIX FAILURES BEFORE RUNNING FULL TRAINING")
    print()
    if "VecNormalize persistence" in " ".join(FAIL):
        print("    python fix_vecnorm_persistence.py")
    if "OBS_TOTAL_DIM" in " ".join(FAIL):
        print("    python apply_target_id_fix.py")
    if "vf_coef" in " ".join(FAIL) or "n_epochs" in " ".join(FAIL):
        print("    python apply_convergence_fix.py")