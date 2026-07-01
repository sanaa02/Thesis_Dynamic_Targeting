#!/usr/bin/env python3
"""
eval_robustness.py  --  ALSAT-EO-1  Robustness to CNN Cloud Forecasting Noise
=============================================================================
Stress tests a trained A1-PPO policy against simulated sensor degradation
by patching the cloud model to return degraded cloud forecasts.

Evaluates at accuracies: [0.95, 0.85, 0.75, 0.65, 0.50]
Runs 10 episodes per accuracy level, logging reward, success rates, and regret.

Output:
-------
  results/evaluation/robustness_results.json
  results/evaluation/robustness_table.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

# ---- path-setup ----
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts/core"))
sys.path.insert(0, os.path.join(ROOT, "scripts/wrappers"))
sys.path.insert(0, os.path.join(ROOT, "scripts/evaluation"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

TARGETS  = os.path.join(ROOT, "config/targets/global_45_targets.json")
CLOUD    = os.path.join(ROOT, "config/cloud_reality/global_45_clouds.json")
OUT_DIR  = os.path.join(ROOT, "results/evaluation")

def run_robustness_test(model_path: str, n_episodes: int = 10, seed: int = 42):
    from stable_baselines3 import PPO
    from env_dynamic_factory import make_env, Config
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Resolve targets and cloud files based on model action space size
    import zipfile
    n_actions = 24
    try:
        with zipfile.ZipFile(model_path, "r") as archive:
            data = json.loads(archive.read("data").decode("utf-8"))
            action_space_info = data.get("action_space", {})
            if isinstance(action_space_info, dict):
                n_actions = int(action_space_info.get("n", 24))
            elif isinstance(action_space_info, str) and "Discrete(" in action_space_info:
                import re
                match = re.search(r"Discrete\((\d+)\)", action_space_info)
                if match:
                    n_actions = int(match.group(1))
    except Exception as exc:
        logger.warning(f"Could not read action space size from model zip: {exc}. Defaulting to 24 actions.")

    if n_actions == 24:
        targets_path = os.path.join(ROOT, "config/targets/algeria_20_targets.json")
        cloud_json_path = os.path.join(ROOT, "config/cloud_reality/algeria_real_clouds.json")
        logger.info("Model matches 20 static targets. Using algeria_20_targets.json.")
    else:
        targets_path = os.path.join(ROOT, "config/targets/global_45_targets.json")
        cloud_json_path = os.path.join(ROOT, "config/cloud_reality/global_45_clouds.json")
        logger.info(f"Model matches {n_actions - 4} static targets. Using global_45_targets.json.")

    # 1. Create environment
    logger.info("Initializing environment wrapper...")
    env = make_env(Config.DYN_REAL_VISION, targets_path, cloud_json_path, event_rate=1.0, seed=seed, with_safety=False)
    
    # Load model with custom_objects and temporary redirects to bypass pickle version mismatch
    import numpy.core
    import numpy.core.numeric
    sys.modules["numpy._core"] = numpy.core
    sys.modules["numpy._core.numeric"] = numpy.core.numeric
    
    # sb3-contrib MaskableActorCriticPolicy use_sde compatibility patch
    try:
        from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
        import inspect
        _orig_init = MaskableActorCriticPolicy.__init__
        def _patched_init(self, *args, **kwargs):
            sig = inspect.signature(_orig_init)
            if "use_sde" not in sig.parameters:
                kwargs.pop("use_sde", None)
            _orig_init(self, *args, **kwargs)
        MaskableActorCriticPolicy.__init__ = _patched_init
    except Exception:
        pass
    
    custom_objects = {"action_space": env.action_space, "observation_space": env.observation_space}
    try:
        model = PPO.load(model_path, env=env, custom_objects=custom_objects)
    finally:
        # Clean up redirects to prevent SystemError in subsequent execution
        if "numpy._core" in sys.modules:
            del sys.modules["numpy._core"]
        if "numpy._core.numeric" in sys.modules:
            del sys.modules["numpy._core.numeric"]
    
    # Unpack wrapper to get the core scenario and active cloud model
    curr = env
    while hasattr(curr, "env"):
        curr = curr.env
    
    # Retrieve the active cloud model from the first satellite
    cloud_model = curr.satellites[0].scenario._cloud_model
    original_forecast = cloud_model.forecast
    
    accuracies = [0.95, 0.85, 0.75, 0.65, 0.50]
    results = {}
    
    logger.info("=== Starting Robustness Stress Tests ===")
    for acc in accuracies:
        logger.info(f"\nEvaluating at Accuracy = {acc:.2f} (degraded cloud model)...")
        
        # Monkey patch forecast to return degraded forecast
        def make_patched_forecast(accuracy_level):
            def patched(target_id: int, sim_time_s: float):
                cnn_forecast, truth = original_forecast(target_id, sim_time_s)
                # Inject noise: accuracy * cnn_forecast + (1 - accuracy) * random_noise
                noise = np.random.uniform(0.0, 1.0)
                degraded = accuracy_level * cnn_forecast + (1.0 - accuracy_level) * noise
                print(f"[DEBUG-ROBUST] Target {target_id} forecast patched: {cnn_forecast:.4f} -> {degraded:.4f} (truth={truth:.4f})", flush=True)
                return float(np.clip(degraded, 0.0, 1.0)), truth
            return patched
            
        cloud_model.forecast = make_patched_forecast(acc)
        
        rewards = []
        dyn_suc_rates = []
        cf_rates = []
        false_positives = []
        false_negatives = []
        
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            ep_r = 0.0
            
            # Additional metrics per step
            n_fp = 0  # false positives: clear target forecast as cloudy (agent skips it, or opposite)
            n_fn = 0  # false negatives: cloudy target forecast as clear
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(int(action))
                ep_r += r
                done = term or trunc
                
            rewards.append(ep_r)
            m = info.get("episode_metrics", {})
            
            # Read clean dynamics imaging metrics
            n_dyn = max(1, m.get("n_dyn_detected", 1))
            n_dyn_ok = m.get("n_dyn_imaged", 0)
            n_cf = m.get("n_cloud_free", 0)
            n_img = max(1, m.get("n_imaged", 1))
            
            dyn_suc_rates.append(n_dyn_ok / n_dyn)
            cf_rates.append(n_cf / n_img)
            
            # Wasted acquisitions or skips (simplified proxy based on cloud stats)
            n_cloudy = m.get("n_cloudy", 0)
            false_negatives.append(n_cloudy / n_img if n_img > 0 else 0.0)
            
            logger.info(f"    Episode {ep + 1}/{n_episodes} complete: Reward={ep_r:.3f} | Dyn Success={n_dyn_ok / n_dyn * 100:.1f}%")
            
        results[acc] = {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_dyn_suc": float(np.mean(dyn_suc_rates)),
            "std_dyn_suc": float(np.std(dyn_suc_rates)),
            "mean_cf_rate": float(np.mean(cf_rates)),
            "mean_wasted_cloudy": float(np.mean(false_negatives)),
        }
        
        logger.info(f"  Accuracy {acc:.2f} results: Reward={results[acc]['mean_reward']:.3f} | Dyn Success={results[acc]['mean_dyn_suc']*100:.1f}%")
        
    env.close()
    
    # Save JSON and CSV
    json_path = os.path.join(OUT_DIR, "robustness_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
        
    csv_path = os.path.join(OUT_DIR, "robustness_table.csv")
    with open(csv_path, "w") as f:
        f.write("Accuracy,Mean_Reward,Std_Reward,Mean_Dyn_Success,Mean_Cloud_Free_Rate,Mean_Wasted_Cloudy\n")
        for acc in accuracies:
            r = results[acc]
            f.write(f"{acc},{r['mean_reward']:.4f},{r['std_reward']:.4f},{r['mean_dyn_suc']:.4f},{r['mean_cf_rate']:.4f},{r['mean_wasted_cloudy']:.4f}\n")
            
    # Print Markdown table
    print("\n\n" + "="*50)
    print("      CNN ROBUSTNESS STRESS TEST RESULTS")
    print("="*50)
    print("| CNN Accuracy | Mean Reward | Dyn Success | Cloud Free Rate | Wasted (Cloudy) |")
    print("| :---: | :---: | :---: | :---: | :---: |")
    for acc in accuracies:
        r = results[acc]
        print(f"| {acc*100:.0f}% | {r['mean_reward']:+.3f} | {r['mean_dyn_suc']*100:.1f}% | {r['mean_cf_rate']*100:.1f}% | {r['mean_wasted_cloudy']*100:.1f}% |")
    print("="*50)
    print(f"Saved results to:\n  - {json_path}\n  - {csv_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate policy robustness to degraded cloud forecasts")
    parser.add_argument("--model", type=str, required=True, help="Path to PPO zip model")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes per accuracy level")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation random seed")
    args = parser.parse_args()
    
    run_robustness_test(args.model, args.episodes, args.seed)
