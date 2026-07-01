#!/usr/bin/env python3
from __future__ import annotations

"""
train_full_system.py  --  ALSAT-EO-1  Full A1 System Training  (v2)
====================================================================
Main training script with ALL 20 improvements active.

VERSION STAMP: v2 — uses ALSATLogger, batch_size=48, verbose=0
If you see SB3's table output or a batch_size=64 warning,
you are running the OLD version of this file. Replace it with
the one from the Replit workspace:
  scripts/training/train_full_system.py

LOGGING  (via alsat_logger.py)
  Terminal:           one clean line per PPO iteration (reward, loss, entropy, ETA)
  logs/training.csv   per-iteration metrics (machine-readable)
  logs/episodes.jsonl per-episode summary (reward, imaged counts, battery, etc.)
  logs/decisions.log  per-step: which target/event chosen, reward received
  logs/orbit.log      satellite position every 20 steps (lat/lon/wilaya)

USAGE
-----
  python scripts/training/train_full_system.py --seed 42 --quick   # 100k steps
  python scripts/training/train_full_system.py --seed 42            # 1M steps
"""

import argparse
import logging
import os
import sys
import time
# from scripts.evaluation.evaluate_real_data import OUT_DIR
from stable_baselines3.common.vec_env import VecNormalize
import numpy as np
from sb3_contrib import MaskablePPO
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import path_setup  # noqa

ROOT = path_setup.root_path()
for _d in ["scripts/core", "scripts/training", "scripts/wrappers",
           "scripts/models", "scripts/evaluation", "scripts"]:
    _p = os.path.join(ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Silence all Python loggers to keep terminal clean ─────────────────────
# Detailed output goes to logs/ via ALSATLogger.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(name)s  %(message)s")
# Also silence noisy bsk_rl / Basilisk messages
_BSK_MUTE = frozenset([
    "Creating logger for new env", "Old environments in process",
    "basePowerDraw should probably be zero or negative",
    "Could not find eclipse transitions",
    "initial_generation_duration is shorter than the maximum window length",
    "failed battery_valid check",
    "Using user-specified world type. Generally, the env-determined world type is sufficient."
])
_orig_ch = logging.Logger.callHandlers
def _quiet(self, r):
    try:
        if any(s in r.getMessage() for s in _BSK_MUTE): return
    except Exception: pass
    _orig_ch(self, r)
logging.Logger.callHandlers = _quiet

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import Basilisk.simulation as sim
print([name for name in dir(sim) if not name.startswith('_')])

# ── Early import check: fail loudly here, not mid-training ────────────────
try:
    from alsat_logger import make_loggers as _make_loggers_check  # noqa
except ImportError as _e:
    print(
        "\n" + "!"*70 +
        f"\n  ERROR: cannot import alsat_logger: {_e}"
        "\n  Make sure scripts/training/alsat_logger.py exists."
        "\n  Download it from the Replit workspace."
        "\n" + "!"*70 + "\n"
    )
    sys.exit(1)

from thesis_logger import ThesisLogger
from wrappers.action_mask_wrapper import make_masked_env
TARGETS  = os.path.join(ROOT, "config/targets/global_45_targets.json")
CLOUD    = os.path.join(ROOT, "config/cloud_reality/global_45_clouds.json")
MODELS   = os.path.join(ROOT, "models")
RESULTS  = os.path.join(ROOT, "results")
LOGS_DIR = os.path.join(ROOT, "logs")

# IMP-09: optimised reward constants
DYNAMIC_BONUS  = 1.0
DYN_MULTIPLIER = 1.5

# IMP-11: n_steps must equal one full episode
# SIM_DURATION_S=172800, SCHED_STEP_S=1200 → 144 steps per episode
try:
    from env_alsat_debug import SIM_DURATION_S, SCHED_STEP_S
    N_STEPS = 2048
except ImportError:
    N_STEPS = 256

# BATCH_SIZE must divide N_STEPS evenly to avoid truncated mini-batches.
# 144 / 3 = 48  (3 equal mini-batches per rollout)
BATCH_SIZE = 64
N_EPOCHS  = 5
N_ENVS   = 2
NET_ARCH   = [128] 
# ═══════════════════════════════════════════════════════════════════════════
# SNIPPET 1: LOGGING SETUP (paste at TOP of train_full_system.py)
# ═══════════════════════════════════════════════════════════════════════════
 
"""
WHERE: scripts/training/train_full_system.py
WHEN: Very first import section, before any training code
"""
 
import logging
import logging.handlers
from datetime import datetime
 
def setup_battery_logging(log_dir="./logs"):
    """Configure battery debugging logger."""
    import os
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.setLevel(logging.DEBUG)
    
    # ── Console: INFO and above ──────────────────────────────────────────
    console_h = logging.StreamHandler()
    console_h.setLevel(logging.INFO)
    console_h.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(console_h)
    
    # ── File: ALL levels (DEBUG and above) ────────────────────────────────
    battery_log = os.path.join(log_dir, "battery_detailed.log")
    file_h = logging.handlers.RotatingFileHandler(
        battery_log,
        maxBytes=100_000_000,  # 100MB before rollover
        backupCount=5
    )
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(file_h)
    
    logger.info(f"Battery logging initialized → {battery_log}")
    return logger

def build_env(
    seed:         int,
    event_rate:   float = 1.0,
    use_real_data: bool = False,
    use_safety:    bool = True,
    use_oracle:    bool = False,
):
    """Build the full A1 training environment with all wrappers."""
    from env_dynamic_factory import make_env, Config
    from stable_baselines3.common.monitor import Monitor

    print(f"[DEBUG] build_env called with event_rate={event_rate}")

    env = make_env(Config.DYN_MODIS, TARGETS, CLOUD,
                   event_rate=event_rate, seed=seed,
                   with_safety=use_safety)

    # IMP-07: oracle cloud
    if use_oracle:
        try:
            from oracle_cloud_wrapper import OracleCloudWrapper
            env = OracleCloudWrapper(env, oracle_cloud=True)
        except Exception:
            pass

    # IMP-05: reward shaping
    try:
        from reward_shaping import DynamicRewardShaper
        env = DynamicRewardShaper(env, urgency_scale=0.2, explore_bonus_init=0.05)
    except Exception:
        pass

    # IMP-13: real data wrapper
    if use_real_data:
        try:
            from env_alsat_real import RealDataWrapper, RealDataConfig
            cfg = RealDataConfig.auto(root=ROOT)
            env = RealDataWrapper(env, cfg)
        except Exception as exc:
            logger.warning(f"[IMP-13] RealDataWrapper skipped: {exc}")

    env = make_masked_env(env)

    return Monitor(env)



import os, logging
import numpy as np
 
logger = logging.getLogger(__name__)
 
 
def run_bc_pretrain(seed: int, n_demos: int = 8000,
                    use_attention: bool = False) -> "str | None":
    """
    IMP-16 (revised): BC pre-training with balanced static/dynamic demos.
 
    CHANGES vs original:
    1. n_demos default is 8000 (was 2000).  BC generalisation requires
       ~5–10K transitions; 2K caused overfitting (loss rising after batch 500
       in your original log). (Naik et al. 2024; imitation library defaults)
 
    2. 70% of demos collected at event_rate=0.0 (static-only).
       Original used event_rate=1.0 for ALL demos → 37% drift + 57% dynamic
       in the dataset.  This gave BC a prior hostile to Stage 0 training.
 
    3. Per-epoch validation accuracy tracking with patience-based early stop.
       Stops when val_acc stops improving for 8 consecutive epochs.
       This prevents the loss increase observed in your original BC log
       (loss went 1.44 → 1.76 between batch 500 and 2000 = overfitting).
 
    4. Saves the BEST checkpoint (highest val_acc), not the last epoch.
    """
    try:
        from bc_demo_collection import collect_demos
        from stable_baselines3 import PPO
        from sb3_contrib import MaskablePPO
        from imitation.algorithms.bc import BC
        from imitation.data.types import Transitions
        import torch
 
        OUT_DIR = os.path.join(os.path.dirname(MODELS), "data", "demos")
        os.makedirs(OUT_DIR, exist_ok=True)
 
        # ── Step 1: collect balanced demos ────────────────────────────────
        n_static_demos = int(n_demos * 0.70)   # 70% static-only
        n_dyn_demos    = n_demos - n_static_demos
 
        logger.info(f"[BC] Collecting {n_static_demos} static demos + "
                    f"{n_dyn_demos} dynamic demos...")
 
        path_s = collect_demos(
            n_demos    = n_static_demos,
            seed       = seed,
            event_rate = 0.0,                  # ← static-only
            out_path   = os.path.join(OUT_DIR, f"bc_static_s{seed}.npz"),
        )
        path_d = collect_demos(
            n_demos    = n_dyn_demos,
            seed       = seed + 999,
            event_rate = 1.0,
            out_path   = os.path.join(OUT_DIR, f"bc_dyn_s{seed}.npz"),
        )
 
        d_s = np.load(path_s)
        d_d = np.load(path_d)
        obs  = np.concatenate([d_s["obs"],     d_d["obs"]],     axis=0)
        acts = np.concatenate([d_s["actions"], d_d["actions"]], axis=0)
 
        # Shuffle combined dataset
        rng  = np.random.default_rng(seed)
        perm = rng.permutation(len(obs))
        obs, acts = obs[perm], acts[perm]
 
        # Log distribution after patching
        unique, counts = np.unique(acts, return_counts=True)
        logger.info("[BC] Action distribution of combined demos (top 8):")
        for cnt, act in sorted(zip(counts, unique), reverse=True)[:8]:
            atype = "static" if act < 20 else ("dynamic" if act < 23 else "drift")
            logger.info(f"  action={act:2d} ({atype:8s})  {cnt/len(acts):5.1%}")
 
        # ── Step 2: 80/20 train/val split ─────────────────────────────────
        n_val    = max(128, len(obs) // 5)
        obs_val  = obs[-n_val:]
        acts_val = acts[-n_val:]
        obs_trn  = obs[:-n_val]
        acts_trn = acts[:-n_val]
        logger.info(f"[BC] Train: {len(obs_trn)}  Val: {len(obs_val)}")
 
        # ── Step 3: build a MaskablePPO model with matching policy ─────────
        env = build_env(seed=seed, use_safety=False)
        policy_kwargs = {}
        if use_attention:
            try:
                from attention_policy import SchedulerAttentionExtractor
                policy_kwargs = dict(
                    features_extractor_class=SchedulerAttentionExtractor,
                    features_extractor_kwargs=dict(features_dim=256,
                                                    d_model=64, n_heads=4),
                    net_arch=[128],    # must match curriculum training
                )
                logger.info("[BC] Using Attention policy (matches curriculum)")
            except Exception as exc:
                logger.warning(f"[BC] Attention unavailable: {exc}")
 
        model = MaskablePPO("MlpPolicy", env, verbose=0,
                             seed=seed, device="cpu",
                             policy_kwargs=policy_kwargs)
 
        # ── Step 4: BC training with early stopping ─────────────────────────
        trn_t = Transitions(
            obs=obs_trn, acts=acts_trn,
            infos=np.array([{}] * len(obs_trn)),
            next_obs=np.roll(obs_trn, -1, axis=0),
            dones=np.zeros(len(obs_trn), dtype=bool),
        )
 
        bc = BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=trn_t,
            policy=model.policy,
            rng=np.random.default_rng(seed),
        )
 
        bc_path = os.path.join(MODELS, f"ppo_bc_pretrain_s{seed}.zip")
        best_acc = 0.0
        patience = 0
        PATIENCE_MAX = 8
 
        obs_val_t  = torch.as_tensor(obs_val,  dtype=torch.float32)
        acts_val_t = torch.as_tensor(acts_val, dtype=torch.long)
 
        for epoch in range(80):
            bc.train(n_epochs=1)
 
            # Validation accuracy
            with torch.no_grad():
                dist = model.policy.get_distribution(obs_val_t)
                pred = dist.distribution.probs.argmax(dim=-1)
                acc  = float((pred == acts_val_t).float().mean())
 
            logger.info(f"[BC] epoch {epoch+1:3d}  val_acc={acc:.4f}  "
                        f"best={best_acc:.4f}  patience={patience}")
 
            if acc > best_acc + 0.002:
                best_acc = acc
                patience = 0
                model.save(bc_path)   # save best model
            else:
                patience += 1
                if patience >= PATIENCE_MAX:
                    logger.info(f"[BC] Early stop at epoch {epoch+1}  "
                                f"best_val_acc={best_acc:.4f}")
                    break
 
        env.close()
        logger.info(f"[BC] Saved best model → {bc_path}  "
                    f"(val_acc={best_acc:.4f}, target ≥0.40)")
 
        if best_acc < 0.30:
            logger.warning(
                "[BC] val_acc < 0.30 — BC pre-training may not help.  "
                "Check that obs dimensions match between demo collection "
                "and training env (both must use TargetIDObsWrapper or neither)."
            )
        return bc_path
 
    except Exception:
        import traceback
        logger.error(f"[BC] BC pretrain FAILED:\n{traceback.format_exc()}")
        return None
 


def _make_ppo(vec_env, policy_kwargs: dict, seed: int) -> object:
    """
    Create MaskablePPO with tuned hyperparameters.
 
    Key changes vs previous version:
      ent_coef  0.05 (start) → annealed to 0.005 by EntropyAnnealingCallback
                Prevents early entropy collapse seen in logs (entropy → 0 by 30k steps).
                Herrmann & Schaub (2023) use 0.01–0.05 for bsk_rl scheduling tasks.
      n_epochs  5   — more gradient steps per collected rollout without instability.
      clip_range 0.15 — tighter than SB3 default (0.2) to prevent large policy jumps
                         on SMDP rewards that can be much larger than typical tasks.
      vf_coef   0.5  — restored to standard SB3 default (reducing it to 0.25 was too
                         aggressive and destabilised the value function).
      net_arch  [128] — adds one nonlinear layer between the attention extractor's
                         256-dim output and the actor/critic heads.  Without it the
                         mapping from features to 24 logits is purely linear.
    """
    # Merge NET_ARCH into policy_kwargs so attention extractor is preserved
    merged_kwargs = dict(policy_kwargs)          # shallow copy
    merged_kwargs["net_arch"] = NET_ARCH         # [128]
 
    return MaskablePPO(
        "MlpPolicy", vec_env,
        learning_rate  = 3e-4,
        n_steps        = N_STEPS,
        batch_size     = BATCH_SIZE,
        n_epochs       = N_EPOCHS,               # 5
        gamma          = 0.99,
        gae_lambda     = 0.95,
        ent_coef       = 0.05,                   # ← KEY FIX: was 0.01, start high
        vf_coef        = 0.5,                    # ← restored from 0.25
        clip_range     = 0.15,                   # ← tighter clip for large SMDP rewards
        max_grad_norm  = 0.5,
        policy_kwargs  = merged_kwargs,
        verbose        = 0,
        seed           = seed,
        device         = "cpu",
    )


"""
PATCH 2 — scripts/training/train_full_system.py
================================================
Add this function to train_full_system.py, ABOVE the run_full_training() definition.
It replaces the broken IMP-17 evaluation (which crashed with the 'use_sde' error).

This is a self-contained evaluation loop that works with MaskablePPO
and produces the exact metrics your thesis jury needs:
  - mean / std episode reward
  - dynamic success rate
  - static imaging rate
  - drift fraction
  - mean battery end-of-episode
"""

import logging, os
import numpy as np

logger = logging.getLogger(__name__)


def _run_manual_evaluation(
    model,
    seed:         int   = 42,
    n_episodes:   int   = 30,
    event_rate:   float = 2.0,
    use_oracle:   bool  = False,
    use_real_data:bool  = False,
) -> dict:
    """
    Run n_episodes of greedy rollout with the trained MaskablePPO model.
    Returns a dict with 'summary' and 'per_episode' keys.

    WHY this instead of evaluate_full_system:
    - MaskablePPO.load() does not accept 'use_sde' keyword (sb3-contrib limitation)
    - This function constructs the env fresh, loads the model weights, and
      steps with model.predict() using the action mask — no extra kwargs needed.

    Produces the metrics table for thesis Chapter 5:
      mean_reward, std_reward, dyn_suc_mean, static_rate, drift_frac, battery_end
    """
    from sb3_contrib.common.wrappers import ActionMasker

    # ── build a fresh single-env for evaluation ────────────────────────────
    eval_env = build_env(
        seed        = seed + 10000,
        event_rate  = event_rate,
        use_safety  = True,
        use_oracle  = use_oracle,
    )

    per_ep = []

    for ep in range(n_episodes):
        ep_seed = seed + 10000 + ep
        obs, _  = eval_env.reset(seed=ep_seed)
        done    = False

        ep_reward      = 0.0
        n_dyn_success  = 0
        n_dyn_attempts = 0
        n_static_imaged= 0
        n_drift        = 0
        n_total        = 0

        # Walk env stack to check sat battery at end
        def _get_sat():
            obj = eval_env
            while hasattr(obj, "env"):
                obj = obj.env
            try:
                return getattr(obj, "unwrapped", obj).satellites[0]
            except Exception:
                return None

        while not done:
            # Get action mask (wrapper exposes get_action_mask if ActionMaskWrapper)
            try:
                mask = eval_env.get_action_mask()
            except AttributeError:
                mask = None

            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            obs, reward, terminated, truncated, info = eval_env.step(int(action))
            done = terminated or truncated

            ep_reward += float(reward)
            n_total   += 1

            # Classify action
            # N_STATIC=20, N_DYN=3, drift=23
            a = int(action)
            if a < 20:       # static target
                n_static_imaged += (1 if reward > 0.1 else 0)
            elif a < 23:     # dynamic slot
                n_dyn_attempts  += 1
                n_dyn_success   += (1 if reward > 0.1 else 0)
            else:            # drift
                n_drift += 1

        # Battery at end of episode
        sat = _get_sat()
        try:
            batt_end = float(sat.dynamics.battery_charge_fraction)
        except Exception:
            batt_end = float("nan")

        dyn_suc_pct = (100.0 * n_dyn_success / max(1, n_dyn_attempts))
        drift_frac  = n_drift / max(1, n_total)
        static_rate = n_static_imaged / 20.0   # fraction of 20 targets imaged

        per_ep.append({
            "ep":            ep,
            "reward":        ep_reward,
            "dyn_suc_pct":   dyn_suc_pct,
            "n_dyn_success": n_dyn_success,
            "n_dyn_attempt": n_dyn_attempts,
            "static_rate":   static_rate,
            "drift_frac":    drift_frac,
            "battery_end":   batt_end,
        })

        logger.info(
            f"[EVAL] ep={ep+1:3d}  r={ep_reward:+7.2f}  "
            f"dyn={dyn_suc_pct:5.1f}%  static={static_rate:.0%}  "
            f"drift={drift_frac:.0%}  batt={batt_end:.0%}"
        )

    eval_env.close()

    rewards     = [e["reward"]      for e in per_ep]
    dyn_sucs    = [e["dyn_suc_pct"] for e in per_ep]
    static_r    = [e["static_rate"] for e in per_ep]
    drift_f     = [e["drift_frac"]  for e in per_ep]
    batts       = [e["battery_end"] for e in per_ep if not np.isnan(e["battery_end"])]

    summary = {
        "n_episodes":       n_episodes,
        "mean_reward":      float(np.mean(rewards)),
        "std_reward":       float(np.std(rewards)),
        "min_reward":       float(np.min(rewards)),
        "max_reward":       float(np.max(rewards)),
        "dyn_suc_mean":     float(np.mean(dyn_sucs)),
        "dyn_suc_std":      float(np.std(dyn_sucs)),
        "static_rate_mean": float(np.mean(static_r)),
        "drift_frac_mean":  float(np.mean(drift_f)),
        "battery_end_mean": float(np.mean(batts)) if batts else float("nan"),
        "event_rate":       event_rate,
    }

    logger.info("\n" + "="*55)
    logger.info("  ALSAT-EO-1  Final Evaluation Summary")
    logger.info("="*55)
    logger.info(f"  Episodes         : {n_episodes}")
    logger.info(f"  Mean reward      : {summary['mean_reward']:+.2f}  (±{summary['std_reward']:.2f})")
    logger.info(f"  Dynamic success  : {summary['dyn_suc_mean']:.1f}%  (±{summary['dyn_suc_std']:.1f}%)")
    logger.info(f"  Static img rate  : {summary['static_rate_mean']:.0%}")
    logger.info(f"  Drift fraction   : {summary['drift_frac_mean']:.0%}")
    logger.info(f"  Battery end-ep   : {summary['battery_end_mean']:.0%}")
    logger.info("="*55 + "\n")

    # Save CSV for thesis table
    import csv
    results_dir = os.path.join(os.path.dirname(RESULTS), "results", "evaluation")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"eval_final_s{seed}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_ep[0].keys()))
        w.writeheader(); w.writerows(per_ep)
    logger.info(f"  Per-episode CSV → {csv_path}")

    return {"summary": summary, "per_episode": per_ep}



import os, logging
import numpy as np
 
logger = logging.getLogger(__name__)
 
 
def run_curriculum_training(
    seed:          int,
    total_steps:   int,
    use_attention: bool = False,
    use_oracle:    bool = False,
    use_real_data: bool = False,
    fresh_logs:    bool = True,
    bc_path:       str  = None,
) -> object:
    """
    4-stage curriculum training for ALSAT-EO-1.
 
    CHANGES vs original 3-stage version:
    1. Added mid-rate bridge stage (event_rate=1.0) between sparse and dense.
       Going 0.5→2.0 directly is too large a jump — the agent receives
       conflicting gradient signals at the start of the dense stage.
       (Curriculum RL literature consensus: step factor ≤2× per stage.)
 
    2. Per-stage entropy reset: ent_coef is raised at the start of each
       stage so the policy re-explores when environment difficulty changes.
       Without this, the collapsed entropy from Stage N prevents the agent
       from adapting to Stage N+1. Backed by Andrychowicz et al. (2020).
 
    3. EntropyAnnealingCallback is re-created per stage so its internal
       timestep counter matches the stage budget.
 
    Budget: 10% / 15% / 15% / 60%
      - static-only:   10%  (enough to learn basic satellite-target geometry)
      - sparse-events: 15%  (introduce dynamic events slowly)
      - mid-events:    15%  (bridge — keeps learning stable across the jump)
      - dense-events:  60%  (main training budget, dense event regime)
    """
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.callbacks import CallbackList
    from alsat_logger import make_loggers
 
    os.makedirs(MODELS,   exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
 
    # ── Stage budget ──────────────────────────────────────────────────────
    s0 = max(N_STEPS, total_steps * 10 // 100)
    s1 = max(N_STEPS, total_steps * 15 // 100)
    s2 = max(N_STEPS, total_steps * 15 // 100)
    s3 = total_steps - s0 - s1 - s2                 # remainder → dense
 
    stages = [
        {"event_rate": 0.0, "steps": s0, "label": "static-only"},
        {"event_rate": 0.5, "steps": s1, "label": "sparse-events"},
        {"event_rate": 1.0, "steps": s2, "label": "mid-events"},   # ← NEW
        {"event_rate": 2.0, "steps": s3, "label": "dense-events"},
    ]
 
    # Per-stage starting entropy (annealed to end_val within each stage)
    STAGE_ENT_START = [0.05, 0.04, 0.03, 0.02]
    ENT_END         = 0.005   # final entropy for all stages
 
    logger.info(f"Curriculum: {total_steps:,} total steps across {len(stages)} stages")
    for i, st in enumerate(stages):
        logger.info(f"  Stage {i}: {st['label']:20s}  "
                    f"rate={st['event_rate']}  steps={st['steps']:,}")
 
    # ── Policy kwargs ─────────────────────────────────────────────────────
    policy_kwargs: dict = {}
    if use_attention:
        try:
            from attention_policy import SchedulerAttentionExtractor
            policy_kwargs = dict(
                features_extractor_class  = SchedulerAttentionExtractor,
                features_extractor_kwargs = dict(features_dim=256,
                                                  d_model=64, n_heads=4),
                net_arch = [128],     # ← CHANGED: was []
            )
            logger.info("[IMP-01] Attention policy active  net_arch=[128]")
        except Exception as exc:
            logger.warning(f"[IMP-01] Attention unavailable: {exc}")
 
    # ── Build initial env and model ───────────────────────────────────────
    vec = DummyVecEnv([
        lambda s=seed + i: build_env(
            s, event_rate=0.0, use_oracle=use_oracle, use_real_data=use_real_data
        )
        for i in range(N_ENVS)
    ])
    model = _make_ppo(vec, policy_kwargs, seed)
 
    # ── Load BC weights if provided ───────────────────────────────────────
    if bc_path and os.path.exists(bc_path):
        try:
            from sb3_contrib import MaskablePPO as _MPPO
            bc_model = _MPPO.load(bc_path, env=vec)
            model.policy.load_state_dict(bc_model.policy.state_dict(), strict=False)
            logger.info(f"[BC] Weights loaded from {bc_path}")
        except Exception as exc:
            logger.warning(f"[BC] Weight load failed: {exc}")
 
    # ── Curriculum loop ───────────────────────────────────────────────────
    cumulative = 0
    for stage_idx, stage in enumerate(stages):
        n_stage = stage["steps"]
        label   = stage["label"]
        rate    = stage["event_rate"]
 
        # ── Per-stage entropy reset ────────────────────────────────────────
        # CRITICAL: raises entropy so the policy re-explores when the env
        # difficulty changes.  Without this, the collapsed entropy from the
        # previous stage prevents adaptation.
        stage_ent = STAGE_ENT_START[stage_idx]
        model.ent_coef = stage_ent
        logger.info(f"Stage {stage_idx}: {label}  "
                    f"rate={rate}  steps={n_stage:,}  ent_coef={stage_ent}")
 
        # ── Rebuild env for this stage ─────────────────────────────────────
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
        model.set_env(vec)
 
        # ── Callbacks ─────────────────────────────────────────────────────
        alsat_cb = make_loggers(
            total_timesteps  = n_stage,
            stage_label      = label,
            log_dir          = LOGS_DIR,
            orbit_every      = 20,
            fresh_logs       = fresh_logs and (stage_idx == 0),
        )
 
        from thesis_logger import ThesisLogger
        thesis_cb = ThesisLogger(
            log_dir          = os.path.join(LOGS_DIR, "../results/verification"),
            every_n_episodes = 10,
            patches_dir      = os.path.join(ROOT, "data/modis_patches"),
        )
 
        # Entropy annealing: decay from stage_ent → ENT_END within this stage
        extra_cbs = []
        try:
            from attention_policy import EntropyAnnealingCallback
            extra_cbs.append(EntropyAnnealingCallback(
                start_val       = stage_ent,
                end_val         = ENT_END,
                total_timesteps = n_stage,
                verbose         = 0,
            ))
        except ImportError:
            try:
                from callbacks import EntropyAnnealingCallback
                extra_cbs.append(EntropyAnnealingCallback(
                    start_val       = stage_ent,
                    end_val         = ENT_END,
                    total_timesteps = n_stage,
                    verbose         = 0,
                ))
            except ImportError:
                logger.warning("[ENTROPY] EntropyAnnealingCallback not found — "
                               "add it to attention_policy.py (see PATCH 4)")
 
        all_cbs = CallbackList([alsat_cb, thesis_cb] + extra_cbs)
 
        model.learn(
            total_timesteps     = n_stage,
            callback            = all_cbs,
            reset_num_timesteps = (cumulative == 0),
            progress_bar        = False,
        )
        cumulative += n_stage
 
        # Save checkpoint after each stage
        ckpt = os.path.join(MODELS, f"ppo_stage_{label}_s{seed}.zip")
        model.save(ckpt)
        logger.info(f"Checkpoint → {ckpt}")
 
    try:
        vec.close()
    except Exception:
        pass
    return model
 


def run_full_training(
    seed:          int   = 42,
    total_steps:   int   = 1_000_000,
    use_attention: bool  = False,
    use_bc:        bool  = False,
    use_real_data: bool  = False,
    use_oracle:    bool  = False,
    quick:         bool  = False,
    fresh_logs:    bool  = True,
) -> dict:

    os.makedirs(MODELS,  exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    if quick:
        total_steps = 100_000
        print(f"[QUICK mode] total_steps = {total_steps:,}")

    print(f"\n{'='*60}")
    print(f"  ALSAT-EO-1  Full System Training")
    print(f"  seed={seed}  steps={total_steps:,}  n_steps={N_STEPS}  batch={BATCH_SIZE}")
    print(f"  attention={use_attention}  bc={use_bc}  real={use_real_data}  oracle={use_oracle}")
    print(f"{'='*60}\n")

    t0 = time.time()

    # IMP-09: patch reward constants
    try:
        import dynamic_event as _de
        _de.DYNAMIC_BONUS  = DYNAMIC_BONUS
        _de.DYN_MULTIPLIER = DYN_MULTIPLIER
    except Exception:
        pass

    # IMP-16 (optional)
    bc_path = None
    if use_bc:
        print("=== IMP-16: BC Pre-training ===")
        bc_path = run_bc_pretrain(seed=seed, n_demos=500 if quick else 2000,
                                   use_attention=use_attention)
        if use_bc and bc_path is None:
            logger.error("[IMP-16] BC was requested but failed. Aborting.")
            sys.exit(1)   # ← don't continue without BC if user asked for it

    # IMP-11: curriculum training
    model = run_curriculum_training(
        seed=seed, total_steps=total_steps,
        use_attention=use_attention,
        use_oracle=use_oracle,
        use_real_data=use_real_data,
        fresh_logs=fresh_logs,
        bc_path=bc_path,
    )

    model_path = os.path.join(MODELS, f"ppo_full_system_s{seed}.zip")
    model.save(model_path)
    elapsed = time.time() - t0
    print(f"\nModel saved → {model_path}  ({elapsed/60:.1f} min)\n")

    # CHANGE 6: run a manual evaluation rollout that does not rely on
    # evaluate_full_system.py (which crashes on the use_sde keyword).
    # This is a self-contained 30-episode greedy evaluation that produces
    # the thesis metrics table directly.
    try:
        eval_results = _run_manual_evaluation(
            model       = model,
            seed        = seed,
            n_episodes  = 5 if quick else 30,
            event_rate  = 2.0,
            use_oracle  = use_oracle,
            use_real_data = use_real_data,
        )
        logger.info(f"[IMP-17] Evaluation complete: {eval_results.get('summary', {})}")
    except Exception as exc:
        import traceback
        logger.warning(f"[IMP-17] Evaluation failed:\n{traceback.format_exc()}")
        eval_results = {}

    return {
        "model_path":   model_path,
        "bc_path":      bc_path,
        "total_steps":  total_steps,
        "seed":         seed,
        "elapsed_min":  round(elapsed / 60, 2),
        "eval_summary": eval_results.get("summary", {}),
    }


if __name__ == "__main__":

    logger = setup_battery_logging()

    parser = argparse.ArgumentParser(description="ALSAT-EO-1 Full A1 System Training")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--steps",       type=int, default=2000000)
    parser.add_argument("--attention",   action="store_true")
    parser.add_argument("--bc-pretrain", action="store_true")
    parser.add_argument("--real-data",   action="store_true")
    parser.add_argument("--oracle-cloud",action="store_true")
    parser.add_argument("--quick",       action="store_true",
                        help="100k steps for testing")
    parser.add_argument("--fresh-logs",  action="store_true", dest="fresh_logs",
                        help="Delete existing logs/ files before starting (default: True; "
                             "pass --no-fresh-logs to append instead)")
    parser.add_argument("--no-fresh-logs", action="store_false", dest="fresh_logs")
    parser.set_defaults(fresh_logs=True)
    args = parser.parse_args()

    result = run_full_training(
        seed         = args.seed,
        total_steps  = args.steps,
        use_attention= args.attention,
        use_bc       = args.bc_pretrain,
        use_real_data= args.real_data,
        use_oracle   = args.oracle_cloud,
        quick        = args.quick,
        fresh_logs   = args.fresh_logs,
    )
    print(f"Done.  Model: {result['model_path']}  ({result['elapsed_min']:.1f} min)")