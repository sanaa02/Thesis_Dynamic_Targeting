#!/usr/bin/env python3
from __future__ import annotations
"""
train_no_curriculum.py  --  ALSAT-EO-1  Direct Training (Ablation Run)
=============================================================================
Ablation test script that trains A1-PPO directly on the final, dense target environment
without any curriculum stages, allowing you to prove the value of curriculum learning.

Writes logs to logs_no_curriculum/ and models to models/ppo_no_curriculum_s{seed}.zip
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import path_setup  # noqa

ROOT = path_setup.root_path()
for _d in ["scripts/core", "scripts/training", "scripts/wrappers",
           "scripts/models", "scripts/evaluation", "scripts"]:
    _p = os.path.join(ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Silence noisy loggers
logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(name)s  %(message)s")
_BSK_MUTE = frozenset([
    "Creating logger for new env", "Old environments in process",
    "basePowerDraw should probably be zero or negative",
    "Could not find eclipse transitions",
    "initial_generation_duration is shorter than the maximum window length",
    "failed battery_valid check",
    "Using user-specified world type. Generally, the env-determined world type is sufficient."
])
if logging.Logger.callHandlers.__name__ != "_quiet":
    _orig_ch = logging.Logger.callHandlers
    def _quiet(self, r):
        try:
            if any(s in r.getMessage() for s in _BSK_MUTE): return
        except Exception: pass
        _orig_ch(self, r)
    logging.Logger.callHandlers = _quiet

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TARGETS  = os.path.join(ROOT, "config/targets/global_45_targets.json")
CLOUD    = os.path.join(ROOT, "config/cloud_reality/global_45_clouds.json")
MODELS   = os.path.join(ROOT, "models")
RESULTS  = os.path.join(ROOT, "results")
LOGS_DIR = os.path.join(ROOT, "logs_no_curriculum")

DYNAMIC_BONUS  = 1.5
DYN_MULTIPLIER = 2.0

try:
    from env_alsat_debug import SIM_DURATION_S, SCHED_STEP_S
    EPISODE_LEN = max(1, int(SIM_DURATION_S / SCHED_STEP_S))  # 144
except ImportError:
    EPISODE_LEN = 144

N_ENVS     = 4
N_STEPS    = EPISODE_LEN * 2      
BATCH_SIZE = (N_STEPS * N_ENVS) // 6  
N_EPOCHS   = 10
NET_ARCH   = {"pi": [128], "vf": [256, 256]}

# Single stage entropy annealing parameters
STAGE_ENT_START = [0.02]
STAGE_ENT_END   = [0.005]

class EntropyAnnealingCallback(BaseCallback):
    def __init__(self, start_val: float, end_val: float,
                 total_timesteps: int, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.start_val       = float(start_val)
        self.end_val         = float(end_val)
        self.total_timesteps = int(total_timesteps)
        self._stage_steps    = 0

    def _on_step(self) -> bool:
        self._stage_steps += 1
        frac    = min(1.0, self._stage_steps / max(1, self.total_timesteps))
        new_ent = self.start_val + frac * (self.end_val - self.start_val)
        self.model.ent_coef = float(new_ent)
        if self.verbose >= 2 and self._stage_steps % 4096 == 0:
            logger.info(
                f"[EntropyAnneal] step={self._stage_steps}  ent_coef={new_ent:.4f}"
            )
        return True

import gymnasium as gym

class BatteryConservationWrapper(gym.Wrapper):
    def __init__(self, env, soc_target: float = 0.30, penalty_scale: float = 0.03):
        super().__init__(env)
        self.soc_target    = soc_target
        self.penalty_scale = penalty_scale

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        try:
            inner = self.env
            while hasattr(inner, "env"):
                inner = inner.env
            sat = getattr(inner, "unwrapped", inner).satellites[0]
            soc = float(sat.dynamics.battery_charge_fraction)
        except Exception:
            soc = 1.0

        if soc < self.soc_target:
            deficit = self.soc_target - soc
            penalty = -self.penalty_scale * (deficit / self.soc_target)
            reward += penalty
        return obs, reward, terminated, truncated, info

def build_env(
    seed:         int,
    event_rate:   float = 1.0,
    clear_sky:    bool = False,
    use_real_data: bool = False,
    use_safety:    bool = True,
    use_oracle:    bool = False,
):
    from env_dynamic_factory import make_env, Config
    from stable_baselines3.common.monitor import Monitor

    env = make_env(Config.DYN_MODIS, TARGETS, CLOUD,
                   event_rate=event_rate, seed=seed,
                   with_safety=use_safety, with_clear_sky=clear_sky)

    if use_oracle:
        try:
            from oracle_cloud_wrapper import OracleCloudWrapper
            env = OracleCloudWrapper(env, oracle_cloud=True)
        except Exception:
            pass

    try:
        from reward_shaping import DynamicRewardShaper
        env = DynamicRewardShaper(env, urgency_scale=0.3, explore_bonus_init=0.005)
    except Exception:
        pass

    env = BatteryConservationWrapper(env, soc_target=0.30, penalty_scale=0.03)

    if use_real_data:
        try:
            from env_alsat_real import RealDataWrapper, RealDataConfig
            cfg = RealDataConfig.auto(root=ROOT)
            cfg.event_rate = event_rate        
            env = RealDataWrapper(env, cfg)
        except Exception as exc:
            logger.warning(f"[RealData] skipped: {exc}")

    from wrappers.action_mask_wrapper import make_masked_env
    env = make_masked_env(env)
    return Monitor(env)

def _make_ppo(vec_env, policy_kwargs: dict, seed: int) -> MaskablePPO:
    merged_kwargs = dict(policy_kwargs)
    merged_kwargs["net_arch"] = NET_ARCH
    return MaskablePPO(
        "MlpPolicy", vec_env,
        learning_rate  = 2.5e-4,
        n_steps        = N_STEPS,
        batch_size     = BATCH_SIZE,
        n_epochs       = 10,
        gamma          = 0.99,
        gae_lambda     = 0.90,
        ent_coef       = STAGE_ENT_START[0],   
        vf_coef        = 2.0,                  
        clip_range     = 0.20,
        max_grad_norm  = 0.5,
        policy_kwargs  = merged_kwargs,
        verbose        = 0,
        seed           = seed,
        device         = "cpu",
    )

def run_bc_pretrain(seed: int, n_demos: int = 8000,
                    use_attention: bool = False) -> "str | None":
    try:
        import torch
        from bc_demo_collection import collect_demos
        from imitation.algorithms.bc import BC
        from imitation.data.types import Transitions

        OUT_DIR = os.path.join(ROOT, "data", "demos")
        os.makedirs(OUT_DIR, exist_ok=True)

        n_static_demos = int(n_demos * 0.60)
        n_dyn_demos    = n_demos - n_static_demos
        logger.info(f"[BC] Collecting {n_static_demos} static + {n_dyn_demos} dynamic demos")

        path_s = collect_demos(
            n_demos=n_static_demos, seed=seed, event_rate=0.0,
            out_path=os.path.join(OUT_DIR, f"bc_static_s{seed}.npz"),
            use_target_id_wrapper=False,
        )
        path_d = collect_demos(
            n_demos=n_dyn_demos, seed=seed + 1, event_rate=1.0,
            out_path=os.path.join(OUT_DIR, f"bc_dyn_s{seed}.npz"),
            use_target_id_wrapper=False,
        )

        logger.info("[BC] Loading demonstrations...")
        d_s = np.load(path_s)
        d_d = np.load(path_d)

        all_obs  = np.concatenate([d_s["obs"],     d_d["obs"]],     axis=0)
        all_acts = np.concatenate([d_s["actions"], d_d["actions"]], axis=0)

        # Slice to exact demo count
        all_obs  = all_obs[:n_demos]
        all_acts = all_acts[:n_demos]

        logger.info(f"[BC] Demo dimensions: obs={all_obs.shape} acts={all_acts.shape}")
        
        transitions = Transitions(
            obs=all_obs,
            acts=all_acts,
            infos=np.array([{}] * len(all_obs)),
            next_obs=all_obs, 
            dones=np.array([False] * len(all_obs))
        )

        dummy_env = build_env(seed)
        dummy_vec = DummyVecEnv([lambda: dummy_env])
        policy_kwargs: dict = {}
        if use_attention:
            from attention_policy import SchedulerAttentionExtractor
            policy_kwargs = dict(
                features_extractor_class  = SchedulerAttentionExtractor,
                features_extractor_kwargs = dict(features_dim=256, d_model=64, n_heads=4),
            )
        
        ppo = _make_ppo(dummy_vec, policy_kwargs, seed)
        bc_algo = BC(
            observation_space=dummy_env.observation_space,
            action_space=dummy_env.action_space,
            demonstrations=transitions,
            rng=np.random.default_rng(seed),
            policy=ppo.policy,
            batch_size=256,
        )

        val_size = int(n_demos * 0.15)
        indices = np.arange(len(all_obs))
        np.random.default_rng(seed).shuffle(indices)
        val_idx = indices[:val_size]
        
        obs_val_t = torch.from_numpy(all_obs[val_idx]).float()
        acts_val_t = torch.from_numpy(all_acts[val_idx]).long()

        logger.info(f"[BC] Starting pre-training (patience=15, validation_size={val_size})...")
        best_acc = 0.0
        patience = 15
        patience_counter = 0
        best_state = None

        bc_path = os.path.join(MODELS, f"bc_pretrained_no_curr_s{seed}.pt")

        for epoch in range(100):
            bc_algo.train(n_epochs=1, progress_bar=False)
            
            with torch.no_grad():
                features = bc_algo.policy.features_extractor(obs_val_t)
                latent_pi, _ = bc_algo.policy.mlp_extractor(features)
                logits = bc_algo.policy.action_net(latent_pi)
                preds = logits.argmax(dim=-1)
                val_acc = (preds == acts_val_t).float().mean().item()
                
                # Check top-3 accuracy
                top3_preds = logits.topk(3, dim=-1).indices
                val_top3 = (top3_preds == acts_val_t.unsqueeze(-1)).any(dim=-1).float().mean().item()

            if val_acc > best_acc + 1e-4:
                best_acc = val_acc
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in bc_algo.policy.state_dict().items()}
                torch.save(best_state, bc_path)
                logger.info(f"[BC] epoch {epoch:3d}  val_acc={val_acc:.4f}  top3={val_top3:.4f}  [NEW BEST]")
            else:
                patience_counter += 1
                if epoch % 5 == 0 or patience_counter > patience - 3:
                    logger.info(f"[BC] epoch {epoch:3d}  val_acc={val_acc:.4f}  top3={val_top3:.4f}  best={best_acc:.4f}  patience={patience_counter}")
                if patience_counter > patience:
                    logger.info(f"[BC] Early stopping triggered at epoch {epoch}")
                    break

        if best_state is not None:
            return bc_path
        return None
    except Exception as e:
        logger.error(f"[BC] Pre-training failed: {e}", exc_info=True)
        return None

def run_direct_training(
    seed:          int,
    total_steps:   int,
    use_attention: bool = False,
    use_oracle:    bool = False,
    use_real_data: bool = False,
    fresh_logs:    bool = True,
    bc_path:       "str | None" = None,
) -> MaskablePPO:
    from alsat_logger import make_loggers
    try:
        from thesis_logger import ThesisLogger
        HAS_THESIS_LOGGER = True
    except ImportError:
        HAS_THESIS_LOGGER = False

    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    rollout_size = N_STEPS * N_ENVS
    total_steps  = max(rollout_size * 4, (total_steps // rollout_size) * rollout_size)

    # Directly train on Stage 3 parameters: dense event rate
    stages = [
        {"event_rate": 1.0, "clear_sky": False, "steps": total_steps, "label": "no-curriculum-dense"}
    ]

    logger.info(f"Direct Training: {total_steps:,} total steps (Ablation Run)")

    policy_kwargs: dict = {}
    if use_attention:
        from attention_policy import SchedulerAttentionExtractor
        policy_kwargs = dict(
            features_extractor_class  = SchedulerAttentionExtractor,
            features_extractor_kwargs = dict(features_dim=256, d_model=64, n_heads=4),
        )

    # Build initial vec env
    vec_raw = DummyVecEnv([
        lambda s=seed + i: build_env(
            s, event_rate=stages[0]["event_rate"], clear_sky=stages[0]["clear_sky"],
            use_oracle=use_oracle, use_real_data=use_real_data
        )
        for i in range(N_ENVS)
    ])
    vec = VecNormalize(vec_raw, norm_obs=False, norm_reward=True, clip_reward=3.0, gamma=0.99)

    model = _make_ppo(vec, policy_kwargs, seed)

    if bc_path and os.path.exists(bc_path):
        try:
            import torch
            loaded = MaskablePPO.load(bc_path)
            model.policy.load_state_dict(loaded.policy.state_dict(), strict=False)
            logger.info(f"[BC] Loaded BC weights from {bc_path}")
        except Exception as exc:
            logger.warning(f"[BC] Load failed: {exc}")

    # Single Curriculum loop step
    stage = stages[0]
    n_stage = stage["steps"]
    label   = stage["label"]
    
    model.ent_coef = STAGE_ENT_START[0]
    model.set_env(vec)

    alsat_cb = make_loggers(
        total_timesteps = n_stage,
        stage_label     = label,
        log_dir         = LOGS_DIR,
        orbit_every     = 20,
        fresh_logs      = fresh_logs,
        start_ep        = 0,
        start_step      = 0,
    )
    callbacks = [alsat_cb]
    
    if HAS_THESIS_LOGGER:
        callbacks.append(ThesisLogger(
            log_dir          = os.path.join(LOGS_DIR, "../results/verification"),
            every_n_episodes = 10,
            patches_dir      = os.path.join(ROOT, "data/modis_patches"),
        ))
    
    callbacks.append(EntropyAnnealingCallback(
        start_val       = STAGE_ENT_START[0],
        end_val         = STAGE_ENT_END[0],
        total_timesteps = n_stage,
        verbose         = 1,
    ))

    logger.info(f"Starting direct model training for {n_stage:,} steps...")
    model.learn(
        total_timesteps = n_stage,
        callback        = CallbackList(callbacks),
        reset_num_timesteps = True,
    )
    
    try:
        vec.save(os.path.join(MODELS, f"vecnorm_no_curr_s{seed}.pkl"))
    except Exception as e:
        logger.warning(f"Could not save VecNormalize stats: {e}")

    vec.close()
    return model

def run_full_training(
    seed:          int,
    total_steps:   int,
    use_attention: bool = False,
    use_bc:        bool = False,
    use_real_data: bool = False,
    use_oracle:    bool = False,
    quick:         bool = False,
    fresh_logs:    bool = True,
) -> dict:
    t0 = time.time()

    # Patch reward constants
    try:
        import dynamic_event as _de
        _de.DYNAMIC_BONUS  = DYNAMIC_BONUS
        _de.DYN_MULTIPLIER = DYN_MULTIPLIER
    except Exception:
        pass

    bc_path = None
    if use_bc:
        print("=== BC Pre-training for Direct Run ===")
        bc_path = run_bc_pretrain(
            seed=seed,
            n_demos=500 if quick else 8000,
            use_attention=use_attention,
        )
        if bc_path is None:
            logger.error("[BC] BC pre-training failed. Aborting.")
            sys.exit(1)

    model = run_direct_training(
        seed=seed, total_steps=total_steps,
        use_attention=use_attention,
        use_oracle=use_oracle,
        use_real_data=use_real_data,
        fresh_logs=fresh_logs,
        bc_path=bc_path,
    )

    model_path = os.path.join(MODELS, f"ppo_no_curriculum_s{seed}.zip")
    model.save(model_path)
    elapsed = time.time() - t0
    print(f"\nAblation Model saved → {model_path}  ({elapsed/60:.1f} min)\n")

    return {
        "model_path":  model_path,
        "bc_path":     bc_path,
        "total_steps": total_steps,
        "seed":        seed,
        "elapsed_min": round(elapsed / 60, 2),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ALSAT-EO-1 Direct Training Ablation (No Curriculum)")
    parser.add_argument("--seed",        type=int,  default=42)
    parser.add_argument("--steps",       type=int,  default=1_000_000)
    parser.add_argument("--attention",   action="store_true")
    parser.add_argument("--bc-pretrain", action="store_true")
    parser.add_argument("--real-data",   action="store_true")
    parser.add_argument("--oracle-cloud",action="store_true")
    parser.add_argument("--quick",       action="store_true")
    parser.add_argument("--fresh-logs",  action="store_true", dest="fresh_logs")
    parser.add_argument("--no-fresh-logs", action="store_false", dest="fresh_logs")
    parser.set_defaults(fresh_logs=True)
    args = parser.parse_args()

    steps = 100_000 if args.quick else args.steps

    result = run_full_training(
        seed         = args.seed,
        total_steps  = steps,
        use_attention= args.attention,
        use_bc       = args.bc_pretrain,
        use_real_data= args.real_data,
        use_oracle   = args.oracle_cloud,
        quick        = args.quick,
        fresh_logs   = args.fresh_logs,
    )
    print(f"Direct training complete. Model: {result['model_path']}")
