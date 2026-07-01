#!/usr/bin/env python3
"""
cnn_speedup_benchmark.py  --  ALSAT-EO-1  IMP-19  CNN Speedup Benchmark
========================================================================
Benchmarks wall-clock time per training episode for three configurations:
  (1) Serial CNN (original -- one inference per step)
  (2) Batched without cache (FIX-CC-1 only)
  (3) Batched with cache (FIX-CC-1+2+3 = BatchedCachedCloudModel)

Records per-episode time in seconds on whatever hardware is available.
Reports GPU model, memory, CUDA version if available.

Usage
-----
    python scripts/training/cnn_speedup_benchmark.py
    python scripts/training/cnn_speedup_benchmark.py --episodes 50

Output
------
  results/benchmark/cnn_speedup_results.json
  results/benchmark/cnn_speedup_plot.png
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

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

TARGETS   = os.path.join(ROOT, "config/targets/global_45_targets.json")
CLOUD     = os.path.join(ROOT, "config/cloud_reality/global_45_clouds.json")
CNN_PATH  = os.path.join(ROOT, "models/cloud_cnn_real.pt")
OUT_DIR   = os.path.join(ROOT, "results/benchmark")

DEFAULT_EPISODES = 100
DEFAULT_SEED     = 42


def _get_hardware_info() -> dict:
    info = {"device": "cpu", "cuda_available": False}
    try:
        import torch
        info["torch_version"]    = torch.__version__
        info["cuda_available"]   = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["device"]        = "cuda"
            info["gpu_name"]      = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
            info["cuda_version"]  = torch.version.cuda
    except ImportError:
        pass
    return info


def _time_episodes(env_factory, n_episodes: int, label: str) -> list[float]:
    """Run n_episodes and return list of per-episode wall-clock times."""
    episode_times = []
    env = env_factory()
    seed = DEFAULT_SEED

    for ep in range(n_episodes):
        t_start = time.perf_counter()
        obs, _  = env.reset(seed=seed + ep)
        done    = False
        while not done:
            action = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
        t_end = time.perf_counter()
        episode_times.append(t_end - t_start)

        if (ep + 1) % 10 == 0:
            logger.info(
                f"  [{label}] ep {ep+1}/{n_episodes}  "
                f"avg={np.mean(episode_times):.2f}s"
            )

    env.close()
    return episode_times


def run_benchmark(n_episodes: int = DEFAULT_EPISODES) -> dict:
    from env_dynamic_factory import make_env, Config

    os.makedirs(OUT_DIR, exist_ok=True)
    hw = _get_hardware_info()
    logger.info(f"Hardware: {hw}")

    results = {"hardware": hw, "n_episodes": n_episodes, "configs": {}}

    # ── Config 1: Serial (DYN_MODIS as serial baseline) ───────────────────────
    def _factory_serial():
        return make_env(Config.DYN_MODIS, TARGETS, CLOUD,
                        event_rate=1.0, seed=DEFAULT_SEED, with_safety=False)

    logger.info(f"\nConfig 1: Serial CNN (Gaussian noise baseline)")
    times_serial = _time_episodes(_factory_serial, n_episodes, "serial")
    results["configs"]["serial"] = {
        "mean_s": float(np.mean(times_serial)),
        "std_s":  float(np.std(times_serial)),
        "episodes": times_serial,
    }

    # ── Config 2: Batched without cache (DYN_VISION if CNN exists) ───────────
    if os.path.exists(CNN_PATH):
        def _factory_batched():
            return make_env(Config.DYN_VISION, TARGETS, CLOUD,
                            event_rate=1.0, seed=DEFAULT_SEED,
                            cnn_path=CNN_PATH, with_safety=False)

        logger.info(f"\nConfig 2: Batched CNN (no cache)")
        times_batched = _time_episodes(_factory_batched, n_episodes, "batched")
        results["configs"]["batched"] = {
            "mean_s": float(np.mean(times_batched)),
            "std_s":  float(np.std(times_batched)),
            "episodes": times_batched,
        }
    else:
        logger.info(f"  [SKIP] Config 2: {CNN_PATH} not found")
        times_batched = None

    # ── Config 3: Batched + cache (DYN_REAL_VISION) ───────────────────────────
    if os.path.exists(CNN_PATH):
        def _factory_cached():
            return make_env(Config.DYN_REAL_VISION, TARGETS, CLOUD,
                            event_rate=1.0, seed=DEFAULT_SEED,
                            cnn_path=CNN_PATH, with_safety=False)

        logger.info(f"\nConfig 3: Batched + cached CNN (BatchedCachedCloudModel)")
        times_cached = _time_episodes(_factory_cached, n_episodes, "cached")
        results["configs"]["batched_cached"] = {
            "mean_s": float(np.mean(times_cached)),
            "std_s":  float(np.std(times_cached)),
            "episodes": times_cached,
        }
    else:
        logger.info(f"  [SKIP] Config 3: {CNN_PATH} not found")
        times_cached = None

    # ── Compute speedup factors ───────────────────────────────────────────────
    base_mean = results["configs"]["serial"]["mean_s"]
    for cfg_name, cfg_data in results["configs"].items():
        if cfg_name != "serial":
            speedup = base_mean / max(cfg_data["mean_s"], 1e-9)
            cfg_data["speedup_vs_serial"] = round(speedup, 1)
            logger.info(
                f"  {cfg_name}: {cfg_data['mean_s']:.3f}s/ep  "
                f"({cfg_data['speedup_vs_serial']}x speedup vs serial)"
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CNN SPEEDUP BENCHMARK SUMMARY")
    print("=" * 60)
    for name, data in results["configs"].items():
        speedup_str = (f"  {data.get('speedup_vs_serial', 1.0):.1f}x"
                       if name != "serial" else "  (baseline)")
        print(f"  {name:20s}  {data['mean_s']:6.3f} ± {data['std_s']:.3f} s/ep"
              f"{speedup_str}")
    print("=" * 60)

    # ── Save results ──────────────────────────────────────────────────────────
    out_json = os.path.join(OUT_DIR, "cnn_speedup_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=float)
    logger.info(f"\nResults saved → {out_json}")

    _plot_benchmark(results)
    return results


def _plot_benchmark(results: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        configs  = results["configs"]
        names    = list(configs.keys())
        means    = [configs[n]["mean_s"] for n in names]
        stds     = [configs[n]["std_s"]  for n in names]
        labels   = {"serial": "Serial", "batched": "Batched\n(no cache)",
                    "batched_cached": "Batched\n+ Cache"}

        fig, ax = plt.subplots(figsize=(7, 5))
        colors  = ["#d62728", "#ff7f0e", "#2ca02c"]
        bars    = ax.bar([labels.get(n, n) for n in names], means,
                         yerr=stds, capsize=4, color=colors[:len(names)],
                         edgecolor="black", linewidth=0.8)
        ax.set_ylabel("Wall-clock time per episode (s)")
        ax.set_title("CNN Cloud Model: Per-Episode Inference Time")
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.02,
                    f"{m:.2f}s", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        out_png = os.path.join(OUT_DIR, "cnn_speedup_plot.png")
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        logger.info(f"Plot saved → {out_png}")
    except ImportError:
        logger.info("  [SKIP] matplotlib not available")
    except Exception as exc:
        logger.warning(f"  Plot error: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    args = parser.parse_args()
    run_benchmark(n_episodes=args.episodes)