#!/usr/bin/env python3
"""
setup_project.py  --  ALSAT-EO-1  Project Setup Script
======================================================
Initialises the full project structure and verifies the environment.

Run once after cloning to:
  1. Create required directories
  2. Generate synthetic config files if missing
  3. Verify Python package availability
  4. Print a run-guide for all 20 improvements

Usage
-----
    python setup_project.py
    python setup_project.py --no-deps  (skip pip install suggestion)
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

DIRS = [
    "scripts/core", "scripts/training", "scripts/models",
    "scripts/data_fetchers", "scripts/wrappers", "scripts/tests",
    "scripts/evaluation", "scripts/plots",
    "config/targets", "config/cloud_reality", "config/orbit",
    "results/ablation/dynamic_bonus",
    "results/ablation/smdp_vs_flat",
    "results/ablation/entropy",
    "results/evaluation", "results/evaluation/real_data",
    "results/evaluation/attention",
    "results/sensitivity",
    "results/transfer_learning",
    "results/curriculum_fixed",
    "results/benchmark",
    "results/figures",
    "models",
    "data/demos", "data/modis_patches", "data/real_events",
]


def make_dirs():
    for d in DIRS:
        path = os.path.join(ROOT, d)
        os.makedirs(path, exist_ok=True)
    print("✓ Directories created")


def verify_packages():
    required = [
        ("numpy",          "numpy"),
        ("gymnasium",      "gymnasium"),
        ("stable_baselines3", "stable-baselines3"),
        ("torch",          "torch"),
    ]
    optional = [
        ("bsk_rl",         "bsk_rl (Basilisk) -- optional, needed for full training"),
        ("matplotlib",     "matplotlib -- optional, needed for plots"),
        ("pytest",         "pytest -- optional, needed for IMP-20 tests"),
        ("sgp4",           "sgp4 -- optional, needed for IMP-13 TLE"),
    ]

    missing_req  = []
    missing_opt  = []

    for mod, pkg in required:
        try:
            __import__(mod)
        except ImportError:
            missing_req.append(pkg)

    for mod, pkg in optional:
        try:
            __import__(mod)
        except ImportError:
            missing_opt.append(pkg)

    if missing_req:
        print(f"\n⚠  Missing REQUIRED packages:\n   pip install {' '.join(missing_req)}\n")
    else:
        print("✓ All required packages present")

    if missing_opt:
        print(f"ℹ  Missing optional packages (install for full functionality):")
        for p in missing_opt:
            print(f"     {p}")
    return not missing_req


def generate_synthetic_configs():
    """Create placeholder configs if real files missing."""
    import json

    # Synthetic event file
    events_path = os.path.join(ROOT, "data/real_events/firms_gdacs_algeria.json")
    if not os.path.exists(events_path):
        from scripts.data_fetchers.fetch_real_events import build_event_database
        try:
            build_event_database(synthetic=True, out_path=events_path)
            print(f"✓ Synthetic event data → {events_path}")
        except Exception as exc:
            print(f"  [skip] event data: {exc}")

    # Synthetic MODIS patches for first 3 targets
    targets_path = os.path.join(ROOT, "config/targets/global_45_targets.json")
    modis_dir    = os.path.join(ROOT, "data/modis_patches")
    if os.path.exists(targets_path):
        try:
            from scripts.data_fetchers.fetch_modis_patches import (
                download_patches_for_targets)
            n = download_patches_for_targets(
                targets_path=targets_path,
                dates=["2024-01-01"],
                out_dir=modis_dir,
                username="", password="",
            )
            print(f"✓ Synthetic MODIS patches → {modis_dir} ({n} files)")
        except Exception as exc:
            print(f"  [skip] MODIS patches: {exc}")


def print_run_guide():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║   ALSAT-EO-1 A1 System  --  Quick Run Guide                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

All 20 improvements are implemented.  Here's how to run each one:

IMP-01  Attention Policy:
  → Use attention_policy.py in scripts/models/  (requires bsk_rl)
  → python scripts/evaluation/attention_analysis.py --model models/ppo.zip

IMP-02  BC Obs-Action Fix (TargetIDObsWrapper):
  → Wrap env before BC collection:
      from scripts.wrappers.target_id_obs_wrapper import TargetIDObsWrapper
      env = TargetIDObsWrapper(env)

IMP-03  Domain Randomisation:
  → Already in env_alsat_debug.py (domain_randomization_wrapper.py)

IMP-04  Real Cloud Model (CNN):
  → Use Config.DYN_VISION in env_dynamic_factory.make_env()

IMP-05  Reward Shaping (DynamicRewardShaper):
  → Already in reward_shaping.py

IMP-06  Entropy Annealing Ablation:
  → python scripts/training/entropy_ablation.py --seeds 42 123 456

IMP-07  Oracle Cloud Ablation:
  → python scripts/training/cloud_uncertainty_ablation.py --quick

IMP-08  SMDP vs Flat MDP Ablation:
  → python scripts/training/smdp_ablation.py --quick

IMP-09  Dynamic Bonus Grid Search:
  → python scripts/training/dynamic_bonus_ablation.py --quick

IMP-10  Safety Monitor:
  → Already in safety_monitor.py (with_safety=True in make_env)

IMP-11  Fixed Curriculum n_steps:
  → python scripts/training/fixed_curriculum.py --seed 42

IMP-12  Transfer Learning:
  → python scripts/training/transfer_learning.py --mode both

IMP-13  Real Data (FIRMS/GDACS/ERA5/MODIS):
  → python scripts/data_fetchers/fetch_real_events.py --synthetic
  → python scripts/data_fetchers/fetch_modis_patches.py --synthetic
  → python scripts/evaluation/evaluate_real_data.py --model models/ppo.zip

IMP-14  Multi-satellite ClaimRegistry:
  → env_multi_satellite.py (requires bsk_rl)

IMP-15  Rate Sensitivity (zero-shot generalisation):
  → python scripts/training/sensitivity_analysis.py --model models/ppo.zip

IMP-16  BC pretrain (existing bc_pretrain.py):
  → python scripts/core/bc_pretrain.py

IMP-17  Full Evaluation:
  → python scripts/evaluation/evaluate_full_system.py --model models/ppo.zip

IMP-18  CNN Speedup Benchmark:
  → python scripts/training/cnn_speedup_benchmark.py --episodes 50

IMP-19  CNN Speedup (BatchedCachedCloudModel in vision_cloud_model.py)

IMP-20  Unit Tests:
  → pytest scripts/tests/test_core.py -v

Plot all results:
  → python scripts/plots/plot_all_results.py

""")


if __name__ == "__main__":
    print("ALSAT-EO-1 Project Setup")
    print("=" * 50)
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "scripts"))

    make_dirs()
    ok = verify_packages()
    generate_synthetic_configs()
    print_run_guide()

    if not ok:
        sys.exit(1)
    print("Setup complete!")
