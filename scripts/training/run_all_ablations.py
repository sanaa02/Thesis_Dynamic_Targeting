#!/usr/bin/env python3
"""
run_all_ablations.py  --  ALSAT-EO-1  Master Ablation Runner
=============================================================
Runs all ablation studies in sequence (or selectively by name).

Ablations covered:
  smdp        IMP-08 SMDP vs Flat MDP
  entropy     IMP-06 entropy coefficient annealing
  cloud       IMP-07 cloud uncertainty (oracle vs standard)
  bonus       IMP-09 dynamic bonus sensitivity grid
  transfer    IMP-12 static pretrain vs scratch
  curriculum  IMP-11 fixed n_steps curriculum
  sensitivity IMP-15 event-rate zero-shot generalisation

Usage
-----
    python scripts/training/run_all_ablations.py --quick
    python scripts/training/run_all_ablations.py --only smdp entropy
    python scripts/training/run_all_ablations.py --seeds 42 123 456
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import path_setup  # noqa

ROOT = path_setup.root_path()
for _d in ["scripts/core", "scripts/training", "scripts/wrappers", "scripts"]:
    _p = os.path.join(ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


ALL_ABLATIONS = ["smdp", "entropy", "cloud", "bonus",
                 "transfer", "curriculum", "sensitivity"]


def _run_smdp(seeds, quick):
    from smdp_ablation import run_ablation
    run_ablation(seeds=seeds, quick=quick)


def _run_entropy(seeds, quick):
    from entropy_ablation import run_ablation
    run_ablation(seeds=seeds, quick=quick)


def _run_cloud(seeds, quick):
    from cloud_uncertainty_ablation import run_ablation
    run_ablation(seeds=seeds, quick=quick)


def _run_bonus(seeds, quick):
    from dynamic_bonus_ablation import run_grid
    run_grid(seeds=seeds, quick=quick)


def _run_transfer(seeds, quick):
    from transfer_learning import train_transfer, train_scratch
    for seed in seeds:
        train_transfer(seed=seed)
        train_scratch(seed=seed)


def _run_curriculum(seeds, quick):
    from fixed_curriculum import run_fixed_curriculum
    for seed in seeds:
        run_fixed_curriculum(seed=seed, use_fixed_n_steps=True)


def _run_sensitivity(seeds, quick):
    """Sensitivity analysis requires a trained model."""
    model_dir = os.path.join(ROOT, "models")
    model_paths = [os.path.join(model_dir, f)
                   for f in os.listdir(model_dir)
                   if f.endswith(".zip") and "scratch" in f]
    if not model_paths:
        logger.warning(
            "[sensitivity] No trained model found in models/. "
            "Run 'transfer' or 'scratch' training first."
        )
        return
    from sensitivity_analysis import run_sensitivity
    model_path = sorted(model_paths)[0]
    logger.info(f"[sensitivity] Using model: {model_path}")
    rates = [0.5, 1.0, 2.0] if quick else [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
    run_sensitivity(model_path, rates=rates, seed=seeds[0],
                    n_episodes=5 if quick else 30)


RUNNERS = {
    "smdp":        _run_smdp,
    "entropy":     _run_entropy,
    "cloud":       _run_cloud,
    "bonus":       _run_bonus,
    "transfer":    _run_transfer,
    "curriculum":  _run_curriculum,
    "sensitivity": _run_sensitivity,
}


def run_all(
    ablations: list[str] = ALL_ABLATIONS,
    seeds:     list[int] = None,
    quick:     bool      = False,
) -> dict:
    if seeds is None:
        seeds = [42] if quick else [42, 123, 456]

    results = {}
    total_t0 = time.time()

    for name in ablations:
        if name not in RUNNERS:
            logger.warning(f"Unknown ablation: {name}. Skipping.")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"ABLATION: {name.upper()}  (seeds={seeds}  quick={quick})")
        logger.info(f"{'='*60}")

        t0 = time.time()
        try:
            RUNNERS[name](seeds, quick)
            elapsed = time.time() - t0
            results[name] = {"status": "OK", "elapsed_min": round(elapsed / 60, 2)}
            logger.info(f"  [{name}] DONE in {elapsed/60:.1f} min")
        except Exception as exc:
            elapsed = time.time() - t0
            logger.error(f"  [{name}] FAILED: {exc}")
            logger.debug(traceback.format_exc())
            results[name] = {"status": "FAILED", "error": str(exc),
                             "elapsed_min": round(elapsed / 60, 2)}

    total_elapsed = time.time() - total_t0
    print("\n" + "=" * 60)
    print("ALL ABLATIONS COMPLETE")
    print(f"  Total: {total_elapsed/60:.1f} min")
    print("-" * 60)
    for name, r in results.items():
        status_sym = "✓" if r["status"] == "OK" else "✗"
        print(f"  {status_sym} {name:<15} {r['status']}  ({r['elapsed_min']:.1f} min)")
    print("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all ablation studies")
    parser.add_argument("--only",  nargs="+", choices=ALL_ABLATIONS,
                        default=ALL_ABLATIONS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--quick", action="store_true",
                        help="Quick run: 1 seed, fewer episodes per ablation")
    args = parser.parse_args()
    run_all(ablations=args.only, seeds=args.seeds, quick=args.quick)
