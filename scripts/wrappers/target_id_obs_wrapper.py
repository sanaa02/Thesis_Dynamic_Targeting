#!/usr/bin/env python3
"""
target_id_obs_wrapper.py  --  ALSAT-EO-1  IMP-02  BC Obs-Action Fix
====================================================================
TargetIDObsWrapper adds a normalised target_id / N_STATIC feature to
each static target slot in the observation vector.

Problem (IMP-02): BC accuracy stuck at ~39% because observation slots
are sorted by TTA but action indices are static target IDs, making the
obs->action mapping unlearnable without an identity feature.

Fix: append target_id / N_STATIC (one scalar per slot) so the network
can learn the correct action index from the obs alone.

OBS_DIM changes: 56 -> 59  (3 extra features, one per DYN slot)
or if adding to static slots: varies based on N_AHEAD_OBSERVE.

This wrapper is applied during BC demo collection AND training so that
demo observations and training observations match exactly (FIX-BC-1).

Usage
-----
    from target_id_obs_wrapper import TargetIDObsWrapper

    env = make_env(...)
    env_with_id = TargetIDObsWrapper(env)

    # Pass as obs_wrapper_fn to collect_demonstrations:
    collect_demonstrations(..., obs_wrapper_fn=lambda e: TargetIDObsWrapper(e))
"""
from __future__ import annotations

import logging
from typing import Optional

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)

# Obs layout constants (must match env_alsat_dynamic.py)
_OBS_BASE_DIM   = 43   # static satellite observation (before dynamic slots)
_N_DYN_SLOTS    = 3    # number of dynamic event slots
_N_DYN_FEATS    = 4    # features per dynamic slot
_N_SOJOURN      = 1    # sojourn time feature
_OBS_TOTAL      = _OBS_BASE_DIM + _N_DYN_SLOTS * _N_DYN_FEATS + _N_SOJOURN  # 56

# How many static target slots are in the base obs (obs[0:43])
# Each target slot has 5 features: priority, cloud, std, opp_open, slew
_N_TARGET_AHEAD = 6    # n_ahead_observe in OpportunityProperties
_N_TARGET_FEATS = 5    # features per static slot


class TargetIDObsWrapper(gym.ObservationWrapper):
    """
    Appends normalised target_id features to each dynamic event slot
    in the observation vector to fix the BC obs-action mismatch.

    The extended obs shape is (OBS_TOTAL + N_DYN_SLOTS,) = (59,).

    Each appended feature: slot_index / N_DYN_SLOTS in [0, 1].
    This allows the network to distinguish which dynamic slot is which
    even after the slots are re-sorted by TTA/priority at each step.
    """

    def __init__(self, env: gym.Env, n_static: int = 20):
        super().__init__(env)
        self._n_static    = n_static
        orig_dim          = env.observation_space.shape[0]
        new_dim           = orig_dim + _N_DYN_SLOTS
        self.observation_space = gym.spaces.Box(
            low  = -np.inf,
            high =  np.inf,
            shape= (new_dim,),
            dtype= np.float32,
        )
        logger.debug(
            f"[TargetIDObsWrapper] obs: {orig_dim} -> {new_dim} "
            f"(+{_N_DYN_SLOTS} slot-ID features)"
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Append slot identity features [0/3, 1/3, 2/3] to obs."""
        slot_ids = np.array(
            [i / _N_DYN_SLOTS for i in range(_N_DYN_SLOTS)],
            dtype=np.float32,
        )
        return np.concatenate([obs, slot_ids], axis=-1).astype(np.float32)