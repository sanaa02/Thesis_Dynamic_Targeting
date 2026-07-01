
#!/usr/bin/env python3
"""
fetch_era5_clouds.py  --  Real cloud cover data for Algeria targets
====================================================================
Downloads cloud-fraction time-series for every target in
config/targets/global_45_targets.json using the Open-Meteo API
(ERA5-reanalysis, free, no API key required).

Writes  config/cloud_reality/era5_clouds_algeria.json  in the exact
format consumed by the existing env_alsat_debug.py cloud model:

    {
      "target_name": {
        "YYYY-MM-DD": 0.35,
        "YYYY-MM-DD": 0.78,
        ...
      },
      ...
    }

This file is a DROP-IN replacement for the existing JSON — just pass it
with --cloud to train_ppo_smdp_full_fixed.py.  No code changes required.

Usage
-----
    # Fetch last 30 days for all 20 Algeria targets
    python scripts/data_fetchers/fetch_era5_clouds.py

    # Custom range
    python scripts/data_fetchers/fetch_era5_clouds.py --days 90

    # Specific targets JSON (if yours lives elsewhere)
    python scripts/data_fetchers/fetch_era5_clouds.py \\
        --targets config/targets/global_45_targets.json

    # Force re-fetch even if fresh
    python scripts/data_fetchers/fetch_era5_clouds.py --force

Open-Meteo API endpoint used
-----------------------------
    https://archive-api.open-meteo.com/v1/archive
    Variable: cloud_cover  (total cloud cover in %, hourly)
    Model:    ERA5  (reanalysis, available from 1940 to ~5 days ago)

For near-real-time data (last 5 days) the script falls back to the
Open-Meteo forecast API (cloud_cover, GFS model).
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
import urllib.parse
from datetime import date, datetime, timedelta, timezone
from typing import Any

ROOT          = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_TARGETS = os.path.join(ROOT, "config", "targets", "global_45_targets.json")
DEFAULT_OUT     = os.path.join(ROOT, "config", "cloud_reality", "era5_clouds_algeria.json")

MAX_AGE_HOURS   = 12
DEFAULT_DAYS    = 30          # days of history to fetch
RATE_LIMIT_S    = 0.5         # courtesy pause between API calls

# Open-Meteo archive endpoint (ERA5)
OM_ARCHIVE = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude={lat}&longitude={lon}"
    "&start_date={start}&end_date={end}"
    "&hourly=cloud_cover"
    "&timezone=UTC"
)
# Open-Meteo forecast (fallback for last ~5 days)
OM_FORECAST = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude={lat}&longitude={lon}"
    "&hourly=cloud_cover"
    "&past_days=5"
    "&forecast_days=1"
    "&timezone=UTC"
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_json(url: str, timeout: int = 30) -> Any:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ALSAT-RL-fetcher/1.0 (research)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _load_targets(targets_path: str) -> list[dict]:
    """
    Load targets JSON.  Supports both formats seen in the codebase:
      - bsk_rl GeneralSatelliteTasking:  list of {"name": ..., "lat": ..., "lon": ...}
      - Older format: {"targets": [...]}
    """
    with open(targets_path) as f:
        raw = json.load(f)

    # Handle nested {"targets": [...]} wrapper
    if isinstance(raw, dict) and "targets" in raw:
        targets = raw["targets"]
    elif isinstance(raw, list):
        targets = raw
    else:
        raise ValueError(f"Unexpected targets JSON format in {targets_path}")

    parsed = []
    for t in targets:
        name = t.get("name", t.get("id", f"target_{len(parsed)}"))
        # lat/lon can be stored as "lat_deg", "lat", "latitude"
        lat  = float(t.get("lat_deg", t.get("lat", t.get("latitude",  0.0))))
        lon  = float(t.get("lon_deg", t.get("lon", t.get("longitude", 0.0))))
        parsed.append({"name": name, "lat": lat, "lon": lon})

    return parsed


def _hourly_to_daily_mean(times: list[str], values: list[float | None]) -> dict[str, float]:
    """
    Aggregate hourly cloud_cover (%) → daily mean fraction [0, 1].
    Missing values (None) are ignored.  Days with < 3 valid hours are skipped.
    """
    daily: dict[str, list[float]] = {}
    for t_str, val in zip(times, values):
        if val is None:
            continue
        day = t_str[:10]   # "YYYY-MM-DD"
        daily.setdefault(day, []).append(float(val))

    result = {}
    for day, vals in daily.items():
        if len(vals) >= 3:
            result[day] = round(sum(vals) / len(vals) / 100.0, 4)   # % → fraction
    return result


def _fetch_cloud_for_target(name: str, lat: float, lon: float,
                             start: str, end: str,
                             verbose: bool = True) -> dict[str, float]:
    """
    Fetch cloud cover for one target using Open-Meteo ERA5.
    Falls back to forecast API for the last 5 days if archive fails.
    """
    url = OM_ARCHIVE.format(lat=lat, lon=lon, start=start, end=end)
    data: dict[str, float] = {}

    try:
        resp   = _get_json(url)
        hourly = resp.get("hourly", {})
        times  = hourly.get("time",        [])
        values = hourly.get("cloud_cover", [])
        data   = _hourly_to_daily_mean(times, values)
        if verbose:
            print(f"    {name:<25} lat={lat:+.2f} lon={lon:+.2f}  "
                  f"→ {len(data)} days  [ERA5 archive]")
    except Exception as exc:
        if verbose:
            print(f"    {name:<25} ERA5 archive failed: {exc}  "
                  f"→ trying forecast fallback...")

    # Supplement with near-real-time forecast for the most recent days
    try:
        resp_fc  = _get_json(OM_FORECAST.format(lat=lat, lon=lon))
        hourly_f = resp_fc.get("hourly", {})
        times_f  = hourly_f.get("time",        [])
        values_f = hourly_f.get("cloud_cover", [])
        fc_data  = _hourly_to_daily_mean(times_f, values_f)
        # Merge: forecast fills gaps and adds recent days
        for day, val in fc_data.items():
            data.setdefault(day, val)
    except Exception:
        pass   # silent — archive data is sufficient

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic fallback
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_cloud_series(days: int = 30) -> dict[str, float]:
    """
    Seasonal climatology for Algeria (summer dry, winter wet).
    Used if both ERA5 archive and forecast API fail.
    """
    import math
    today = date.today()
    result = {}
    for i in range(days):
        d = today - timedelta(days=days - i - 1)
        # Month-based climatology:  July/Aug ≈ 0.10, Dec/Jan ≈ 0.45
        month = d.month
        seasonal = 0.28 - 0.18 * math.cos(2 * math.pi * (month - 1) / 12)
        # Small pseudo-random daily variation (deterministic from date)
        noise = 0.08 * math.sin(d.toordinal() * 1.618)
        val   = round(max(0.0, min(1.0, seasonal + noise)), 4)
        result[d.strftime("%Y-%m-%d")] = val
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def fetch_era5_clouds(targets_path:  str  = DEFAULT_TARGETS,
                      out_path:      str  = DEFAULT_OUT,
                      days:          int  = DEFAULT_DAYS,
                      force:         bool = False,
                      verbose:       bool = True) -> str:
    """
    Fetch ERA5 cloud cover for all targets and write the output JSON.

    Parameters
    ----------
    targets_path : Path to global_45_targets.json
    out_path     : Destination JSON file
    days         : Number of historical days to fetch
    force        : Re-fetch even if output is fresh
    verbose      : Print progress

    Returns
    -------
    str  — path to written JSON file
    """
    # Skip if fresh
    if not force and os.path.exists(out_path):
        age_h = (time.time() - os.path.getmtime(out_path)) / 3600.0
        if age_h < MAX_AGE_HOURS:
            if verbose:
                print(f"  [ERA5] {out_path} is {age_h:.1f}h old — skipping fetch.")
            return out_path

    targets = _load_targets(targets_path)
    if verbose:
        print(f"  [ERA5] Fetching cloud cover for {len(targets)} targets "
              f"(last {days} days)...")

    end_date   = date.today() - timedelta(days=5)   # ERA5 archive lag ~5 days
    start_date = end_date - timedelta(days=days - 1)
    start_str  = start_date.strftime("%Y-%m-%d")
    end_str    = end_date.strftime("%Y-%m-%d")

    if verbose:
        print(f"  [ERA5] ERA5 archive range: {start_str} → {end_str}")

    cloud_json: dict[str, dict[str, float]] = {}

    for i, tgt in enumerate(targets):
        name = tgt["name"]
        lat  = tgt["lat"]
        lon  = tgt["lon"]

        data = _fetch_cloud_for_target(
            name, lat, lon, start_str, end_str, verbose=verbose
        )

        # Fallback if API returned nothing
        if not data:
            if verbose:
                print(f"    {name:<25} ← using seasonal climatology fallback")
            data = _synthetic_cloud_series(days)

        cloud_json[name] = data
        time.sleep(RATE_LIMIT_S)   # respect Open-Meteo rate limit

    # Metadata header (ignored by env_alsat_debug.py — it only reads target keys)
    cloud_json["_meta"] = {
        "_fetched_utc": datetime.now(timezone.utc).isoformat(),
        "_source":      "Open-Meteo ERA5 reanalysis",
        "_days":        days,
        "_start":       start_str,
        "_end":         end_str,
        "_n_targets":   len(targets),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cloud_json, f, indent=2, ensure_ascii=False)

    if verbose:
        total_days = sum(len(v) for k, v in cloud_json.items() if not k.startswith("_"))
        print(f"\n  [ERA5] Saved {len(targets)} targets × ~{days} days "
              f"({total_days} data-points) → {out_path}")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fetch ERA5 cloud cover for Algeria targets from Open-Meteo"
    )
    ap.add_argument("--targets", default=DEFAULT_TARGETS,
                    help="Path to global_45_targets.json")
    ap.add_argument("--out",     default=DEFAULT_OUT,
                    help="Output JSON path")
    ap.add_argument("--days",    type=int, default=DEFAULT_DAYS,
                    help=f"Days of ERA5 history to fetch (default {DEFAULT_DAYS})")
    ap.add_argument("--force",   action="store_true",
                    help="Re-fetch even if output is fresh")
    ap.add_argument("--quiet",   action="store_true",
                    help="Suppress progress output")
    args = ap.parse_args()

    fetch_era5_clouds(
        targets_path = args.targets,
        out_path     = args.out,
        days         = args.days,
        force        = args.force,
        verbose      = not args.quiet,
    )


if __name__ == "__main__":
    main()