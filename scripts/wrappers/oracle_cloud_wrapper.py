#!/usr/bin/env python3
"""
oracle_cloud_wrapper.py  --  ALSAT-EO-1  IMP-07  Oracle Cloud Ablation
=======================================================================
OracleCloudWrapper implements the `oracle_cloud=True` mode for the
cloud-uncertainty ablation study (IMP-07).

When active, it sets cloud_cover_forecast = cloud_cover (ground truth)
for every target and dynamic event in the environment, eliminating CNN
forecast noise entirely.

This supports three ablation conditions:
  (a) Standard policy: CNN forecast with sigma=0.05 (normal training)
  (b) Oracle policy:   cloud_cover_forecast = cloud_cover (sigma=0)
  (c) Standard trained + oracle at test time

The wrapper patches the cloud forecast at each reset and step so that
the policy always observes perfect cloud information.

Scientific basis: IMP-07 (ALSAT Roadmap)
  "Add an oracle_cloud=True flag to DynamicObsWrapper that sets
   cloud_cover_forecast = cloud_cover (no noise)."

Usage
-----
    from oracle_cloud_wrapper import OracleCloudWrapper

    env = make_env(...)
    oracle_env = OracleCloudWrapper(env, oracle_cloud=True)

    # For comparison: standard policy with CNN noise
    standard_env = OracleCloudWrapper(env, oracle_cloud=False)  # no-op
"""
from __future__ import annotations

import logging
from typing import Optional

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)


class OracleCloudWrapper(gym.Wrapper):
    """
    Patches cloud_cover_forecast = cloud_cover (ground truth) for all
    targets and dynamic events, eliminating CNN forecast uncertainty.

    Parameters
    ----------
    env : gym.Env
        The underlying DynamicObsWrapper-based environment.
    oracle_cloud : bool
        If True, set forecast = truth (oracle mode).
        If False, this wrapper is a passthrough (no change).
    add_noise_std : float
        Optional noise to add even in oracle mode (default 0.0).
        Set to CNN_NOISE_STD to test partial oracle conditions.
    """

    def __init__(
        self,
        env: gym.Env,
        oracle_cloud: bool = True,
        add_noise_std: float = 0.0,
        seed: Optional[int] = None,
    ):
        super().__init__(env)
        self._oracle     = oracle_cloud
        self._noise_std  = add_noise_std
        self._rng        = np.random.default_rng(seed or 0)
        self._n_patched  = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self._oracle:
            self._patch_forecasts()
        return obs, info

    def step(self, action: int):
        obs, r, term, trunc, info = self.env.step(action)
        if self._oracle:
            self._patch_forecasts()
        return obs, r, term, trunc, info

    # ── internal ────────────────────────────────────────────────────────────

    def _patch_forecasts(self) -> None:
        """Walk wrapper stack and patch all cloud forecasts to ground truth."""
        patched = 0
        try:
            inner = self.env
            while hasattr(inner, "env"):
                inner = inner.env
            base = getattr(inner, "unwrapped", inner)
            sat  = base.satellites[0]

            # Static targets
            scenario = getattr(sat, "scenario", None)
            if scenario is not None:
                for tgt in getattr(scenario, "targets", []):
                    truth = float(getattr(tgt, "cloud_cover", 0.0))
                    noise = (self._rng.normal(0, self._noise_std)
                             if self._noise_std > 0 else 0.0)
                    tgt.cloud_cover_forecast = float(
                        np.clip(truth + noise, 0.0, 1.0))
                    patched += 1

            # Dynamic events
            mgr = getattr(sat, "_event_manager", None)
            if mgr is not None:
                for evt in getattr(mgr, "_events", []):
                    truth = float(getattr(evt, "cloud_cover", 0.0))
                    noise = (self._rng.normal(0, self._noise_std)
                             if self._noise_std > 0 else 0.0)
                    evt.cloud_cover_forecast = float(
                        np.clip(truth + noise, 0.0, 1.0))
                    patched += 1

        except Exception as exc:
            logger.debug(f"[OracleCloud] patch error: {exc}")

        self._n_patched += patched
        logger.debug(f"[OracleCloud] patched {patched} forecasts")

    def get_stats(self) -> dict:
        return {
            "oracle_mode": self._oracle,
            "noise_std":   self._noise_std,
            "n_patched":   self._n_patched,
        }
