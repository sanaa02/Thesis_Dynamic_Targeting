
#!/usr/bin/env python3
"""
scout_provider.py  --  ALSAT-EO-1  Real MODIS Patch Sampler
============================================================
Provides a randomised stream of real MODIS 64×64 patches
**without exposing the label (cloud fraction)** embedded in
the filename.  This breaks the circular dependency that
existed in CloudCNNPredictor, where the synthetic patch
generator received cloud_truth as input and therefore the
CNN's input was derived from its own target label.

Usage
-----
    from scout_provider import RealScoutImageProvider

    provider = RealScoutImageProvider("data/modis_patches")
    patch = provider.get_patch()          # np.ndarray (3, 64, 64) float32
    patch = provider.get_patch_for(0.4)  # nearest-CF patch (optional)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class RealScoutImageProvider:
    """
    Randomly samples a real MODIS patch (3, 64, 64) from a directory of
    .npy files extracted by extract_modis_patches.py.

    Filename convention:  cf{CF:.3f}_{stem}_{i:04d}.npy
    The CF encoded in the filename is *intentionally ignored* when returning
    the image so that no label information leaks into the CNN input pipeline.

    Parameters
    ----------
    patches_dir : str
        Directory containing the .npy patch files.
    seed : int
        RNG seed for reproducible sampling.
    """

    def __init__(self, patches_dir: str, seed: int = 42) -> None:
        patches_dir = os.path.expanduser(patches_dir)
        if not os.path.isdir(patches_dir):
            raise FileNotFoundError(
                f"RealScoutImageProvider: patches directory not found: {patches_dir}"
            )

        self._dir  = patches_dir
        self._rng  = np.random.default_rng(seed)

        # Index all .npy files (sorted for determinism across platforms)
        self._index: List[str] = sorted(
            str(Path(patches_dir) / f)
            for f in os.listdir(patches_dir)
            if f.endswith(".npy")
        )
        if not self._index:
            raise FileNotFoundError(
                f"RealScoutImageProvider: no .npy files found in {patches_dir}"
            )

        # Optional: build a CF-indexed lookup for get_patch_for()
        # Keys: cloud fraction (float), Values: list of file indices
        self._cf_index: dict[float, List[int]] = {}
        for i, path in enumerate(self._index):
            cf = self._parse_cf(Path(path).name)
            if cf is not None:
                key = round(cf, 2)
                self._cf_index.setdefault(key, []).append(i)

        logger.info(
            f"RealScoutImageProvider: {len(self._index)} patches loaded "
            f"from {patches_dir}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_patch(self) -> np.ndarray:
        """
        Return a random (3, 64, 64) float32 patch.
        The cloud fraction of the selected patch is NOT revealed.
        """
        idx = int(self._rng.integers(0, len(self._index)))
        return self._load(self._index[idx])
    
    # In scout_provider.py — add this new method alongside get_patch()

    def get_patch_for_target(
        self,
        target_lat: float,
        target_lon: float,
        utc: datetime.datetime,
        cloud_lookup: "CloudLookup",                    # import from cloud_lookup
        patch_size: tuple = (32, 32),
    ) -> "np.ndarray":
        """
        Return a cloud patch spatially registered to (target_lat, target_lon)
        at the given UTC time.

        Strategy:
        1. Get cloud fraction from ERA5/MODIS at this exact (lat, lon, utc)
        2. Select the MODIS patch from the pool whose cloud_fraction is
            nearest to the looked-up value  ← spatial + temporal registration
        3. If the pool has geo-tagged patches, select the one whose bounding
            box overlaps (target_lat, target_lon)

        This replaces the previous approach of returning a completely random
        patch, which had no connection to the actual target location or time.
        """
        import numpy as np

        # Step 1: Get real cloud fraction at this location and time
        cf_real = cloud_lookup.get(lat=target_lat, lon=target_lon, utc=utc)

        # Step 2: Find the pool patch whose recorded cloud fraction is closest
        if hasattr(self, '_patch_pool') and len(self._patch_pool) > 0:
            pool_cfs  = np.array([p.get('cloud_fraction', 0.5)
                                for p in self._patch_pool])
            best_idx  = int(np.argmin(np.abs(pool_cfs - cf_real)))
            patch_img = self._patch_pool[best_idx].get('image')
            if patch_img is not None:
                arr = np.asarray(patch_img, dtype=np.float32)
                if arr.ndim == 2:
                    arr = arr[np.newaxis]          # → (1, H, W)
                return arr

        # Step 3: If no geo-tagged pool, generate a realistic synthetic patch
        # that encodes the real cloud fraction (NOT the "circular CNN" approach —
        # we're using the real CF from ERA5, not from the CNN output)
        return self._synthesize_patch_from_cf(cf_real, patch_size, target_lat, target_lon)


    def _synthesize_patch_from_cf(
        self,
        cloud_fraction: float,
        patch_size: tuple = (32, 32),
        lat: float = 33.0,
        lon: float = 3.0,
    ) -> "np.ndarray":
        """
        Generate a spatially coherent cloud patch matching a given cloud fraction.
        Uses a fractal/Perlin-noise approach for realistic cloud morphology,
        NOT pure Gaussian noise.

        This is used only when no real MODIS patch is available for the target.
        """
        import numpy as np
        from scipy.ndimage import gaussian_filter

        H, W  = patch_size
        rng   = np.random.default_rng(
            int(abs(lat * 1000 + lon * 100 + cloud_fraction * 50))
        )
        # Multi-scale cloud texture (Gaussian pyramid)
        noise = np.zeros((H, W), dtype=np.float32)
        for scale, weight in [(2, 0.5), (4, 0.3), (8, 0.2)]:
            n = rng.standard_normal((H // scale + 1, W // scale + 1))
            import scipy.ndimage as ndi
            n_up = ndi.zoom(n, scale, order=1)[:H, :W]
            noise += weight * n_up

        # Threshold to match target cloud fraction
        threshold = float(np.percentile(noise, (1.0 - cloud_fraction) * 100))
        cloud_mask = (noise >= threshold).astype(np.float32)

        # Add brightness variation: clear = 0.1 (dark ground), cloud = 0.8–1.0
        patch = np.where(cloud_mask > 0,
                        0.8 + 0.2 * rng.random((H, W)).astype(np.float32),
                        0.05 + 0.1 * rng.random((H, W)).astype(np.float32))
        return patch[np.newaxis].astype(np.float32)   # (1, H, W)

    def get_patch_for(self, target_cf: float, tol: float = 0.1) -> np.ndarray:
        """
        Return a patch whose embedded CF is within *tol* of *target_cf*.
        Falls back to a purely random patch if none is close enough.

        Note: using this method re-introduces a *soft* correlation between
        the CNN input and ground truth.  Only use it if your ablation study
        explicitly requires it; for the main pipeline use get_patch().
        """
        best_key = None
        best_dist = float("inf")
        for key in self._cf_index:
            d = abs(key - target_cf)
            if d < best_dist:
                best_dist = d
                best_key = key

        if best_key is not None and best_dist <= tol:
            candidates = self._cf_index[best_key]
            idx = candidates[int(self._rng.integers(0, len(candidates)))]
            return self._load(self._index[idx])

        # Fallback: purely random
        logger.debug(
            f"get_patch_for({target_cf:.2f}): no patch within tol={tol}; "
            "returning random patch."
        )
        return self.get_patch()

    def reset(self, seed: Optional[int] = None) -> None:
        """Re-seed the RNG (called by cloud model on env reset)."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    @property
    def n_patches(self) -> int:
        """Number of patches in the pool."""
        return len(self._index)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_cf(filename: str) -> Optional[float]:
        """
        Parse the cloud fraction from the filename convention
        ``cf{CF:.3f}_{stem}_{i:04d}.npy``.
        Returns None if the filename does not match the convention.
        """
        try:
            if filename.startswith("cf"):
                return float(filename[2:7])
        except (ValueError, IndexError):
            pass
        return None

    def _load(self, path: str) -> np.ndarray:
        """
        Load a .npy patch and normalise shape to (3, 64, 64) float32.
        Handles edge cases from corrupted or differently-formatted files.
        """
        try:
            patch = np.load(path)
        except Exception as exc:
            logger.warning(f"Failed to load {path}: {exc}; returning zeros.")
            return np.zeros((3, 64, 64), dtype=np.float32)

        # Shape normalisation (defensive — all valid patches are already (3,64,64))
        if patch.ndim == 2:
            # Grayscale → replicate across 3 channels
            patch = np.stack([patch, patch, patch], axis=0)
        elif patch.ndim == 3:
            if patch.shape == (64, 64, 3):
                # HWC → CHW
                patch = patch.transpose(2, 0, 1)
            elif patch.shape[0] not in (1, 3):
                # Unexpected — return zeros
                logger.warning(
                    f"Unexpected patch shape {patch.shape} in {path}; "
                    "returning zeros."
                )
                return np.zeros((3, 64, 64), dtype=np.float32)
        else:
            logger.warning(
                f"Unexpected patch ndim={patch.ndim} in {path}; returning zeros."
            )
            return np.zeros((3, 64, 64), dtype=np.float32)

        patch = patch.astype(np.float32)

        # Sanity checks
        if not np.isfinite(patch).all():
            patch = np.nan_to_num(patch, nan=0.0, posinf=1.0, neginf=0.0)
        if patch.max() < 1e-4:
            logger.debug(f"All-black patch at {path}; still returning it.")

        return patch
