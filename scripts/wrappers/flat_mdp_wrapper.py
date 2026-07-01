#!/usr/bin/env python3
"""
flat_mdp_wrapper.py  --  ALSAT-EO-1  IMP-08  Flat MDP Baseline
================================================================
FlatMDPWrapper converts the variable-length SMDP steps into fixed-length
MDP steps by distributing the SMDP reward uniformly across sub-steps
with standard discount gamma^1 per sub-step.

This is used as the ablation baseline for IMP-08:
  "SMDP Discounting vs Standard MDP Fixed Timestep"

The wrapper intercepts the step() return and replaces the SMDP-discounted
total_r with reward_per_substep * n_sub, then applies gamma^1 per sub.

Usage
-----
    from flat_mdp_wrapper import FlatMDPWrapper

    smdp_env = make_dynamic_env(...)       # DynamicObsWrapper
    flat_env  = FlatMDPWrapper(smdp_env)   # flat MDP baseline

    # Use exactly like smdp_env — same obs/action space
    obs, info = flat_env.reset()
    obs, r, done, trunc, info = flat_env.step(action)

Scientific basis
----------------
Eddy & Kochenderfer (2019) show that proper semi-Markov discounting (gamma^tau)
leads to better long-horizon policies than treating every sub-step identically.
FlatMDPWrapper implements the "flat MDP" baseline for direct comparison.

In the flat MDP formulation:
  R_flat(a) = total_r / n_sub        (uniform per sub-step)
  discount   = gamma^1 per sub-step  (not gamma^(tau/STEP_REF_S))

This flattens out the advantage of short slews, removing the incentive to pick
geometrically close events.
"""
from __future__ import annotations

import logging
from typing import Optional

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)


class FlatMDPWrapper(gym.Wrapper):
    """
    Converts SMDP variable-length steps to fixed-step MDP transitions.

    The underlying DynamicObsWrapper already computes a SMDP-discounted
    total_r = sum_i gamma_sub^i * r_i.  This wrapper divides out the
    SMDP structure and replaces it with uniform reward distribution.

    Parameters
    ----------
    env : DynamicObsWrapper
        The SMDP environment to wrap.
    gamma : float
        Standard MDP discount factor per sub-step (default 0.99).
    redistribute : bool
        If True (default), divide total_r by n_sub and apply gamma^1.
        If False, keep total_r unchanged but re-discount with gamma^1
        (partial fix — use True for clean ablation).
    """

    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        redistribute: bool = True,
    ):
        super().__init__(env)
        self._gamma        = gamma
        self._redistribute = redistribute
        self._n_flat_steps = 0
        self._n_smdp_steps = 0

    def step(self, action: int):
        obs, smdp_r, term, trunc, info = self.env.step(action)

        n_sub   = int(info.get("smdp_n_sub", 1))
        tau_s   = float(info.get("smdp_tau_s", 30.0))

        if self._redistribute and n_sub > 0:
            # Flat MDP: uniform reward per sub-step, standard gamma^1
            r_per_sub  = smdp_r / n_sub
            flat_r = sum(r_per_sub * (self._gamma ** i) for i in range(n_sub))
        else:
            flat_r = smdp_r   # passthrough

        self._n_flat_steps += n_sub
        self._n_smdp_steps += 1

        info["flat_mdp_r"]     = float(flat_r)
        info["smdp_r_orig"]    = float(smdp_r)
        info["flat_n_sub"]     = n_sub
        info["flat_tau_s"]     = tau_s

        logger.debug(
            f"[FlatMDP] action={action}  n_sub={n_sub}  "
            f"smdp_r={smdp_r:+.4f}  flat_r={flat_r:+.4f}  "
            f"tau={tau_s:.0f}s"
        )

        return obs, flat_r, term, trunc, info

    def reset(self, **kwargs):
        self._n_flat_steps = 0
        self._n_smdp_steps = 0
        return self.env.reset(**kwargs)

    def get_stats(self) -> dict:
        return {
            "n_flat_steps": self._n_flat_steps,
            "n_smdp_steps": self._n_smdp_steps,
            "avg_substeps": (self._n_flat_steps / max(1, self._n_smdp_steps)),
        }
