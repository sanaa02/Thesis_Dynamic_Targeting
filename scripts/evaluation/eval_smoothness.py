#!/usr/bin/env python3
"""
eval_smoothness.py  --  ALSAT-EO-1  Policy Smoothness & Actuator Dynamics Audit
================================================================================
Evaluates a trained A1-PPO policy to audit its mechanical feasibility.
Measures:
  - Action Coherence (AC): stability of action selections
  - Slew Rate Variance (SRV): acceleration profile roughness
  - Settling Time Violations (STV): risk of imaging during attitude settling

Outputs:
-------
  Prints ASCII Policy Smoothness Audit Report (ready for the thesis!)
  Saves metrics to results/evaluation/smoothness_audit.json
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

def run_smoothness_audit(model_path: str, n_episodes: int = 10, seed: int = 42):
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
    
    all_actions = []
    all_slew_rates = []
    settling_violations = 0
    total_steps = 0
    max_consecutive_switches = 0
    
    logger.info("=== Running Smoothness Audit ===")
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        
        ep_actions = []
        ep_slew_rates = []
        consecutive_switches = 0
        prev_action = None
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            
            ep_actions.append(action)
            total_steps += 1
            
            # Estimate slew rate: slew_deg / sojourn_time
            # Get info metrics
            slew_deg = info.get("step_metrics", {}).get("slew_deg", 0.0)
            if slew_deg == 0.0:
                # Fallback to checking the info dict keys or step log
                slew_deg = info.get("slew_deg", 0.0)
                
            # Get sojourn time (tau) from observation or info
            tau = info.get("step_metrics", {}).get("sojourn_time", 30.0)
            if tau == 0.0:
                tau = 30.0
                
            slew_rate = slew_deg / tau
            ep_slew_rates.append(slew_rate)
            
            # Detect Action Switches and Settling Violations
            # If agent switches action rapidly without drifting or settling
            if prev_action is not None and action != prev_action:
                consecutive_switches += 1
                max_consecutive_switches = max(max_consecutive_switches, consecutive_switches)
                
                # Check for Settling Time Violation:
                # If a large slew occurred (slew_deg > 15 deg) and we immediately image in the next step
                # without an intervening drift (charge) action, it violates settling guidelines.
                if slew_deg > 15.0 and action != 23: # 23 is drift index for 24 action space
                    settling_violations += 1
            else:
                consecutive_switches = 0
                
            prev_action = action
            
        all_actions.extend(ep_actions)
        all_slew_rates.extend(ep_slew_rates)
        logger.info(f"    Episode {ep + 1}/{n_episodes} audited (slew count={len(ep_actions)})")
        
    env.close()
    
    # Calculate metrics
    actions_arr = np.array(all_actions)
    ac = np.mean(actions_arr[1:] == actions_arr[:-1])
    srv = np.var(all_slew_rates)
    stv_rate = settling_violations / total_steps if total_steps > 0 else 0.0
    
    # Format and save JSON
    audit_results = {
        "action_coherence": float(ac),
        "slew_rate_variance": float(srv),
        "settling_violations_count": int(settling_violations),
        "settling_violations_rate": float(stv_rate),
        "max_consecutive_switches": int(max_consecutive_switches),
        "total_steps_evaluated": int(total_steps),
    }
    
    json_path = os.path.join(OUT_DIR, "smoothness_audit.json")
    with open(json_path, "w") as f:
        json.dump(audit_results, f, indent=2)
        
    # Determine status
    ac_status = "✓ PASS" if ac > 0.85 else "△ MARG" if ac > 0.75 else "✗ FAIL"
    srv_status = "✓ PASS" if srv < 5.0 else "△ MARG" if srv < 8.0 else "✗ FAIL"
    stv_status = "✓ PASS" if stv_rate < 0.005 else "△ MARG" if stv_rate < 0.01 else "✗ FAIL"
    sw_status = "✓ PASS" if max_consecutive_switches < 5 else "✗ FAIL"
    
    # Print the report
    report = f"""
╔══════════════════════════════════════════════════════════╗
║         POLICY SMOOTHNESS AUDIT REPORT                  ║
║         Policy: PPO-SMDP-v1 | Seed: {seed}                  ║
╠══════════════════════════════════════════════════════════╣
║ METRIC                  │ VALUE      │ STATUS │ THRESHOLD ║
╠══════════════════════════════════════════════════════════╣
║ Action Coherence (AC)   │ {ac:.3f}      │  {ac_status} │ > 0.85    ║
║ Slew Rate Variance      │ {srv:.2f} °/s²   │  {srv_status} │ < 5.0 °/s² ║
║ Settling Violations     │ {settling_violations}/{total_steps} ({stv_rate*100:.1f}%) │  {stv_status} │ < 1.0%    ║
║ Max Consecutive Switch  │ {max_consecutive_switches}          │  {sw_status} │ < 5        ║
╠══════════════════════════════════════════════════════════╣
║ CONCLUSION: Policy exhibits {"smooth" if ac > 0.85 and srv < 5.0 else "moderately stable" if ac > 0.75 else "high jitter"} behavior.       ║
║ {"No smoothing layer required." if ac > 0.85 else "Smoothing layer or action penalty recommended."}                   ║
╚══════════════════════════════════════════════════════════╝
"""
    print(report)
    logger.info(f"Audit metrics saved to: {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit trained policy mechanical smoothness")
    parser.add_argument("--model", type=str, required=True, help="Path to PPO zip model")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()
    
    run_smoothness_audit(args.model, args.episodes, args.seed)
