
#!/usr/bin/env python3
"""
fetch_modis_patches.py  --  Download real MODIS L1B cloud patches for Algeria
==============================================================================
Downloads MODIS Terra MOD02QKM (250 m, Band 1 reflectance) HDF4 files via
the NASA Earthdata earthaccess library, extracts 32×32-pixel patches centred
on each Algeria target, and saves them as .npy files.

Output directory structure
--------------------------
    data/modis_patches/
        <target_name>/
            2024-01-15.npy    # float32 array (32, 32), reflectance [0, 1]
            2024-01-16.npy
            ...

These patches are consumed by:
  - cloud_cnn.py  (CloudCNNTrainer uses them for CNN training)
  - env_alsat_debug.py  ModisCloudModel._get_patch()  (runtime inference)

Prerequisites
-------------
    pip install earthaccess pyhdf numpy

    Then authenticate once:
        python -c "import earthaccess; earthaccess.login(strategy='interactive')"
    This writes ~/.netrc with your NASA Earthdata credentials.
    Register free at: https://urs.earthdata.nasa.gov/

IMPORTANT: MOD02QKM files are HDF4 format, not HDF5.
    Primary reader: pyhdf   (pip install pyhdf)
    Fallback reader: h5py   (only works if NASA provides HDF5 variant)

Usage
-----
    python scripts/data_fetchers/fetch_modis_patches.py
    python scripts/data_fetchers/fetch_modis_patches.py --days 30 --patch-size 32
    python scripts/data_fetchers/fetch_modis_patches.py --targets config/targets/global_45_targets.json
    python scripts/data_fetchers/fetch_modis_patches.py --dry-run   # list granules only

Product: MOD02QKM  (MODIS Terra Level-1B Calibrated Radiances 250m)
Band 1: 620-670 nm (red) -- cloud/snow/ice bright, dense vegetation dark
The script converts DN to reflectance using the embedded scale/offset
attributes in the HDF4 file structure.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import date, timedelta
from typing import Any

import numpy as np

ROOT            = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_TARGETS = os.path.join(ROOT, "config", "targets", "global_45_targets.json")
DEFAULT_OUTDIR  = os.path.join(ROOT, "data", "modis_patches")
DEFAULT_DAYS    = 14
DEFAULT_PATCH   = 32   # pixels (250 m/px -> 32x32 = 8 km x 8 km)

MODIS_PX_M    = 250.0
EARTH_R_M     = 6_378_137.0
MAX_GRANULES_PER_DAY = 2


# ─────────────────────────────────────────────────────────────────────────────
# Target loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_targets(path: str) -> list[dict]:
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "targets" in raw:
        raw = raw["targets"]
    out = []
    for t in raw:
        name = t.get("name", t.get("id", f"target_{len(out)}"))
        lat  = float(t.get("lat_deg", t.get("lat", t.get("latitude",  0.0))))
        lon  = float(t.get("lon_deg", t.get("lon", t.get("longitude", 0.0))))
        out.append({"name": name, "lat": lat, "lon": lon})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# HDF4 extraction  (primary — MOD02QKM is HDF4 format)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_patch_hdf4(hdf_path: str,
                         lat: float, lon: float,
                         patch_size: int = DEFAULT_PATCH) -> np.ndarray | None:
    """
    Read MOD02QKM HDF4 file using pyhdf and extract a patch at (lat, lon).

    HDF4 SDS names in MOD02QKM:
        Latitude                 -- (n_scan//5, 1354)  subsampled 5x
        Longitude                -- same
        EV_250_Aggr500_RefSB     -- (2, n_scan, 2708)  Bands 1-2 aggregated to 500m
    or:
        EV_250_RefSB             -- (2, n_scan, 5416)  Bands 1-2 at native 250m

    Returns float32 (patch_size, patch_size) reflectance [0, 1.5] or None.
    """
    try:
        from pyhdf.SD import SD, SDC
    except ImportError:
        raise ImportError(
            "pyhdf not installed. Fix with:\n"
            "    pip install pyhdf\n"
            "pyhdf is required to read MOD02QKM HDF4 files."
        )

    sd = None
    try:
        sd = SD(str(hdf_path), SDC.READ)
        datasets = sd.datasets()

        # ── Geolocation (subsampled 5x) ──────────────────────────────────────
        if "Latitude" not in datasets or "Longitude" not in datasets:
            print(f"    [WARN] {os.path.basename(hdf_path)}: Latitude/Longitude SDS not found. "
                  f"Available: {list(datasets.keys())[:8]}")
            return None

        lat_arr = sd.select("Latitude")[:]
        lon_arr = sd.select("Longitude")[:]

        # Distance in degrees — find nearest geo pixel
        dist2 = (lat_arr - lat) ** 2 + (lon_arr - lon) ** 2
        r5, c5 = np.unravel_index(dist2.argmin(), dist2.shape)

        # Scale back to full-resolution row/col
        # geo is every 5th scan line, every 1st pixel across-track
        r_full = r5 * 5
        c_full = c5   # across-track: full pixel count in geo already (1354 px)

        # ── Reflectance data ─────────────────────────────────────────────────
        # Try 500m-aggregated first (smaller file), then native 250m
        ev_key = None
        for candidate in ("EV_250_Aggr500_RefSB", "EV_250_RefSB"):
            if candidate in datasets:
                ev_key = candidate
                break

        if ev_key is None:
            print(f"    [WARN] {os.path.basename(hdf_path)}: No EV_250 dataset. "
                  f"Available: {list(datasets.keys())[:10]}")
            return None

        ds = sd.select(ev_key)
        attrs = ds.attributes()

        # Scale / offset: stored as lists (one per band)
        scale  = attrs.get("reflectance_scales",  [1.0])
        offset = attrs.get("reflectance_offsets", [0.0])
        if hasattr(scale,  "__iter__"): scale  = float(scale[0])
        else:                           scale  = float(scale)
        if hasattr(offset, "__iter__"): offset = float(offset[0])
        else:                           offset = float(offset)

        fill_val = float(attrs.get("_FillValue", 65535))

        # Read Band 1 (index 0): shape is (n_bands, n_lines, n_pixels)
        raw_data = ds[0, :, :]   # Band 1, all lines, all pixels — shape (n_lines, n_pixels)
        n_lines, n_pixels = raw_data.shape

        # Scale across-track pixel index if 500m aggregated (half the across-track pixels)
        if ev_key == "EV_250_Aggr500_RefSB":
            # 500m aggregated: geo at 1354 px, data at 2708 px -> col_full = c5 * 2
            c_full_data = c5 * 2
        else:
            # Native 250m: geo at 1354 px, data at 5416 px -> col_full = c5 * 4
            c_full_data = c5 * 4

        # ── Patch extraction ─────────────────────────────────────────────────
        half = patch_size // 2
        r0 = max(0, r_full - half);  r1 = min(n_lines,  r0 + patch_size)
        c0 = max(0, c_full_data - half); c1 = min(n_pixels, c0 + patch_size)

        raw = raw_data[r0:r1, c0:c1].astype(np.float32)

        # Mask fill values
        raw[raw == fill_val] = offset   # will become 0.0 after scale

        # DN -> reflectance
        refl = (raw - offset) * scale

        # Pad to patch_size if at boundary
        if r1 <= r0 or c1 <= c0 or refl.size == 0:
            return None  # target outside swath — try next granule
        pad_r = patch_size - (r1 - r0)
        pad_c = patch_size - (c1 - c0)
        if pad_r > 0 or pad_c > 0:
            refl = np.pad(refl, ((0, pad_r), (0, pad_c)), mode="constant", constant_values=0.0)

        return np.clip(refl, 0.0, 1.5).astype(np.float32)[:patch_size, :patch_size]

    except Exception as exc:
        print(f"    [WARN] HDF4 extraction error ({os.path.basename(hdf_path)}): {exc}")
        return None
    finally:
        if sd is not None:
            try:
                sd.end()
            except Exception:
                pass

def _extract_patch_nc(hdf_path: str,
                      lat: float, lon: float,
                      patch_size: int = DEFAULT_PATCH) -> np.ndarray | None:
    """Read collection 007 .nc files using the netCDF4 library."""
    try:
        import netCDF4 as nc4
    except ImportError:
        print("    [WARN] netCDF4 not installed. Fix: pip install netCDF4")
        return None
    try:
        ds = nc4.Dataset(str(hdf_path), "r")
        # Variable names may be at root or inside a group
        def _get_var(name):
            if name in ds.variables:
                return np.array(ds.variables[name][:])
            for g in ds.groups.values():
                if name in g.variables:
                    return np.array(g.variables[name][:])
            return None
        lat_arr = _get_var("Latitude")
        lon_arr = _get_var("Longitude")
        if lat_arr is None or lon_arr is None:

            ds.close(); return None
        dist2 = (lat_arr - lat) ** 2 + (lon_arr - lon) ** 2
        r5, c5 = np.unravel_index(dist2.argmin(), dist2.shape)
        r_full, c_full = r5 * 5, c5 * 5
        ev_ds = _get_var("EV_250_Aggr500_RefSB") or _get_var("EV_250_RefSB")
        if ev_ds is None:
            print(f"    [WARN] No EV_250 dataset in {os.path.basename(hdf_path)}")
            ds.close(); return None
        # ev_ds shape: (n_bands, n_lines, n_pixels)
        band1 = np.array(ev_ds[0]).astype(np.float32)
        n_lines, n_pixels = band1.shape
        half = patch_size // 2
        r0 = max(0, r_full - half); r1 = min(n_lines,  r0 + patch_size)
        c0 = max(0, c_full - half); c1 = min(n_pixels, c0 + patch_size)
        raw = band1[r0:r1, c0:c1]
        pad_r = patch_size - (r1 - r0)
        pad_c = patch_size - (c1 - c0)
        if r1 <= r0 or c1 <= c0 or raw.size == 0:
            ds.close(); return None
        if pad_r > 0 or pad_c > 0:
            raw = np.pad(raw, ((0, pad_r), (0, pad_c)), mode="constant", constant_values=0.0)
        ds.close()
        return np.clip(raw / 10000.0, 0.0, 1.5).astype(np.float32)[:patch_size, :patch_size]
    
    except Exception as exc:
        print(f"    [WARN] netCDF4 extraction error ({os.path.basename(hdf_path)}): {exc}")
        return None
# ─────────────────────────────────────────────────────────────────────────────
# HDF5 extraction  (fallback — some derived products are HDF5)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_patch_hdf5(hdf_path: str,
                         lat: float, lon: float,
                         patch_size: int = DEFAULT_PATCH) -> np.ndarray | None:
    """
    Fallback reader for HDF5-format MODIS files (e.g., MOD021KM from LAADS).
    """
    try:
        import h5py
    except ImportError:
        return None

    try:
        with h5py.File(hdf_path, "r") as f:

            # ── FLEXIBLE GEOLOCATION HANDLING ─────────────────────────────
            # collection 007 .nc: /MODIS_SWATH_Type_L1B/Geolocation Fields/Latitude
            # collection 061 .hdf via h5py (rare): /Geolocation Fields/Latitude
            swath = f.get("MODIS_SWATH_Type_L1B", f)
            if "Geolocation Fields" in swath:
                geo = swath["Geolocation Fields"]
                lat_arr = geo["Latitude"][:]
                lon_arr = geo["Longitude"][:]
            elif "Latitude" in swath and "Longitude" in swath:
                lat_arr = swath["Latitude"][:]
                lon_arr = swath["Longitude"][:]
            else:
                print(f"    [WARN] No geolocation fields in {os.path.basename(hdf_path)}")
                return None

            dist2 = (lat_arr - lat) ** 2 + (lon_arr - lon) ** 2
            r5, c5 = np.unravel_index(dist2.argmin(), dist2.shape)
            r_full, c_full = r5 * 5, c5 * 5

            data_grp = swath.get("Data Fields", swath)
            ev_key   = None
            for candidate in ("EV_250_Aggr500_RefSB", "EV_250_RefSB"):
                if candidate in data_grp:
                    ev_key = candidate
                    break
            if ev_key is None:
                return None

            ds    = data_grp[ev_key]
            attrs = dict(ds.attrs)
            scale  = float(np.atleast_1d(attrs.get("reflectance_scales",  [1.0]))[0])
            offset = float(np.atleast_1d(attrs.get("reflectance_offsets", [0.0]))[0])

            n_bands, n_lines, n_pixels = ds.shape
            half = patch_size // 2
            r0 = max(0, r_full - half);  r1 = min(n_lines,  r0 + patch_size)
            c0 = max(0, c_full - half);  c1 = min(n_pixels, c0 + patch_size)

            raw  = ds[0, r0:r1, c0:c1].astype(np.float32)
            refl = (raw - offset) * scale

            pad_r = patch_size - (r1 - r0)
            pad_c = patch_size - (c1 - c0)
            if pad_r > 0 or pad_c > 0:
                refl = np.pad(refl, ((0, pad_r), (0, pad_c)), mode="edge")

            return np.clip(refl, 0.0, 1.5).astype(np.float32)[:patch_size, :patch_size]

    except Exception as exc:
        print(f"    [WARN] HDF5 extraction error ({os.path.basename(hdf_path)}): {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Unified extraction: try HDF4 first, then HDF5
# ─────────────────────────────────────────────────────────────────────────────

def _extract_patch(hdf_path: str,
                   lat: float, lon: float,
                   patch_size: int = DEFAULT_PATCH) -> np.ndarray | None:
    """
    Route to correct reader based on file extension.
    .hdf / .he4  -> pyhdf  (HDF4 — standard MOD02QKM collection 061)
    .nc / .nc4   -> h5py   (netCDF4/HDF5 — collection 007 files)
    """
    ext = str(hdf_path).lower().rsplit(".", 1)[-1]

    if ext in ("nc", "nc4"):
        return _extract_patch_nc(hdf_path, lat, lon, patch_size)
    if ext in ("h5", "he5"):
        return _extract_patch_hdf5(hdf_path, lat, lon, patch_size)

    try:
        import pyhdf  # noqa
        return _extract_patch_hdf4(hdf_path, lat, lon, patch_size)
    except ImportError:
        print("    [WARN] pyhdf not installed. Fix: pip install pyhdf")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# earthaccess search + download
# ─────────────────────────────────────────────────────────────────────────────

def _search_granules(lat: float, lon: float,
                     day: date,
                     max_results: int = MAX_GRANULES_PER_DAY) -> list[Any]:
    """Search for MOD02QKM granules covering (lat, lon) on *day*."""
    import earthaccess
    bbox     = (lon - 0.5, lat - 0.5, lon + 0.5, lat + 0.5)
    date_str = day.strftime("%Y-%m-%d")
    return earthaccess.search_data(
        short_name   = "MOD02QKM",
        temporal     = (date_str, date_str),
        bounding_box = bbox,
        count        = max_results,
    )


def _download_granule(granule_list: list, tmp_dir: str) -> str | None:
    """Download granule files to tmp_dir. Returns the path of the best radiance file."""
    import earthaccess
    try:
        paths = earthaccess.download(granule_list, local_path=tmp_dir)
        if not paths:
            return None
        str_paths = [str(p) for p in paths]
        # Prefer .hdf files — these are the actual L1B radiance data.
        # .nc files from the same granule are calibration/QA sidecars with no reflectance data.
        for p in str_paths:
            if p.lower().endswith((".hdf", ".he4")):
                return p
        return str_paths[0]  # fallback: return whatever we got
    except Exception as exc:
        print(f"    [WARN] Download failed: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic fallback patch
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_patch(lat: float, lon: float,
                     day_ord: int,
                     patch_size: int = DEFAULT_PATCH,
                     cloud_frac: float | None = None) -> np.ndarray:
    """Seasonal-climatology patch for Algeria when MODIS data is unavailable."""
    rng = np.random.default_rng(int(abs(lat * 100) + abs(lon * 100)) + day_ord)
    month = ((day_ord % 365) // 30) + 1
    if cloud_frac is None:
        seasonal = 0.28 - 0.18 * math.cos(2 * math.pi * (month - 1) / 12)
        cloud_frac = float(np.clip(seasonal + rng.uniform(-0.10, 0.10), 0.05, 0.85))

    n_cloud_px = int(patch_size * patch_size * cloud_frac)
    patch = rng.uniform(0.05, 0.20, size=(patch_size, patch_size)).astype(np.float32)

    if n_cloud_px > 0:
        flat      = patch.ravel()
        cloud_idx = rng.choice(len(flat), size=n_cloud_px, replace=False)
        flat[cloud_idx] = rng.uniform(0.60, 1.00, size=n_cloud_px).astype(np.float32)
        patch = flat.reshape(patch_size, patch_size)

    return patch


# ─────────────────────────────────────────────────────────────────────────────
# Per-target fetch
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_patches_for_target(target: dict,
                               days: int,
                               out_dir: str,
                               patch_size: int,
                               dry_run: bool,
                               verbose: bool,
                               tmp_dir: str) -> int:
    name = target["name"]
    lat  = target["lat"]
    lon  = target["lon"]

    tgt_dir = os.path.join(out_dir, name)
    os.makedirs(tgt_dir, exist_ok=True)

    today = date.today()
    saved = 0

    for i in range(days):
        day     = today - timedelta(days=days - i - 1)
        day_str = day.strftime("%Y-%m-%d")
        out_path = os.path.join(tgt_dir, f"{day_str}.npy")

        if os.path.exists(out_path):
            continue   # already saved

        patch: np.ndarray | None = None
        granule_found = False

        # Try earthaccess
        try:
            import earthaccess
            granules = _search_granules(lat, lon, day, max_results=4)
            granule_found = len(granules) > 0

          

            if dry_run:
                continue

            if granule_found:
                for granule in granules[:4]:
                    hdf_path = _download_granule([granule], tmp_dir)
                    if hdf_path:
                        patch = _extract_patch(hdf_path, lat, lon, patch_size)
                        try:
                            os.remove(hdf_path)
                        except OSError:
                            pass
                        if patch is not None:
                            break   # got a good patch, stop trying

        except ImportError:
            pass   # earthaccess not installed
        except Exception as exc:
            if verbose:
                print(f"    [WARN] Search/download {name} {day_str}: {exc}")

        # Use synthetic fallback if extraction failed or no granules found
        reason = ""
        if patch is None:
            patch  = _synthetic_patch(lat, lon, day.toordinal(), patch_size)
            if granule_found:
                reason = "(granule found but extraction failed — check pyhdf install)"
            else:
                reason = "(no granule in region for this date)"
            if verbose:
                print(f"    {name:<25} {day_str}  <- synthetic fallback {reason}")
        else:
            if verbose:
                mean_refl = float(patch.mean())
                est_cloud = min(1.0, max(0.0, mean_refl - 0.15) / 0.7)
                print(f"    {name:<25} {day_str}  "
                      f"refl_mean={mean_refl:.3f}  cloud_est={est_cloud:.0%}  [MODIS]")

        np.save(out_path, patch.astype(np.float32))
        saved += 1

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Top-level function
# ─────────────────────────────────────────────────────────────────────────────

def fetch_modis_patches(targets_path: str  = DEFAULT_TARGETS,
                        out_dir:      str  = DEFAULT_OUTDIR,
                        days:         int  = DEFAULT_DAYS,
                        patch_size:   int  = DEFAULT_PATCH,
                        dry_run:      bool = False,
                        verbose:      bool = True) -> str:
    """Download MODIS patches for all targets."""
    targets = _load_targets(targets_path)
    if verbose:
        print(f"\n  [MODIS] Fetching patches for {len(targets)} targets x {days} days")
        print(f"          Patch size: {patch_size}x{patch_size} px  "
              f"({patch_size * 0.25:.0f}x{patch_size * 0.25:.0f} km)")

        # Check pyhdf availability upfront
        try:
            import pyhdf
            print("  [MODIS] pyhdf available -- will read HDF4 files correctly")
        except ImportError:
            print("  [MODIS] WARNING: pyhdf not installed.")
            print("          MOD02QKM files are HDF4 format and CANNOT be read without pyhdf.")
            print("          All downloads will fall back to synthetic patches.")
            print("          Fix:  pip install pyhdf")

        try:
            import earthaccess
            earthaccess.login(strategy="netrc")
            print("  [MODIS] Authenticated with NASA Earthdata (netrc)")
        except ImportError:
            print("  [MODIS] earthaccess not installed -> synthetic patches.")
            print("          Install:  pip install earthaccess pyhdf")
        except Exception as exc:
            print(f"  [MODIS] earthaccess auth failed ({exc}) -> synthetic patches.")

    tmp_dir = os.path.join(out_dir, "_tmp")
    os.makedirs(tmp_dir,  exist_ok=True)
    os.makedirs(out_dir,  exist_ok=True)

    total_saved = 0
    for tgt in targets:
        if verbose:
            print(f"\n  -> {tgt['name']}  ({tgt['lat']:+.2f}N  {tgt['lon']:+.2f}E)")
        n = _fetch_patches_for_target(
            tgt, days, out_dir, patch_size, dry_run, verbose, tmp_dir
        )
        total_saved += n
        time.sleep(0.2)

    try:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    if verbose and not dry_run:
        print(f"\n  [MODIS] Done -- {total_saved} new patches saved to {out_dir}/")
        print(f"          Layout:  {out_dir}/<target_name>/<YYYY-MM-DD>.npy")

    return out_dir


# ─────────────────────────────────────────────────────────────────────────────
# Patch reader (used by env_alsat_real.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_patch(target_name: str,
               day_str:     str,
               patch_dir:   str = DEFAULT_OUTDIR,
               patch_size:  int = DEFAULT_PATCH) -> np.ndarray:
    """Load saved .npy patch. Returns synthetic patch if file not found."""
    path = os.path.join(patch_dir, target_name, f"{day_str}.npy")
    if os.path.exists(path):
        try:
            return np.load(path).astype(np.float32)
        except Exception:
            pass
    import datetime as _dt
    try:
        day = _dt.date.fromisoformat(day_str)
        ord_val = day.toordinal()
    except ValueError:
        ord_val = 0
    lat = hash(target_name) % 100 / 100.0 * 18 + 19
    lon = hash(target_name[::-1]) % 100 / 100.0 * 21 - 9
    return _synthetic_patch(lat, lon, ord_val, patch_size)


def cloud_fraction_from_patch(patch: np.ndarray,
                               cloud_thresh_refl: float = 0.40) -> float:
    """Estimate cloud fraction from a MODIS Band 1 reflectance patch."""
    return float(np.mean(patch > cloud_thresh_refl))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download real MODIS cloud patches for Algeria targets"
    )
    ap.add_argument("--targets",    default=DEFAULT_TARGETS, help="Targets JSON path")
    ap.add_argument("--out",        default=DEFAULT_OUTDIR,  help="Output directory")
    ap.add_argument("--days",       type=int, default=DEFAULT_DAYS,
                    help=f"Days of patches to download (default {DEFAULT_DAYS})")
    ap.add_argument("--patch-size", type=int, default=DEFAULT_PATCH,
                    help=f"Patch dimension in pixels (default {DEFAULT_PATCH})")
    ap.add_argument("--dry-run",    action="store_true",
                    help="List available granules without downloading")
    ap.add_argument("--quiet",      action="store_true", help="Suppress progress output")
    args = ap.parse_args()

    fetch_modis_patches(
        targets_path = args.targets,
        out_dir      = args.out,
        days         = args.days,
        patch_size   = args.patch_size,
        dry_run      = args.dry_run,
        verbose      = not args.quiet,
    )


if __name__ == "__main__":
    main()
