
#!/usr/bin/env python3
"""
fetch_tle.py  --  Download real TLE for ALSAT-2A from Celestrak
================================================================
Downloads the current Two-Line Element set for ALSAT-2A (NORAD ID 39500)
from Celestrak and writes it to  config/orbit/alsat2a.tle

Usage
-----
    python scripts/data_fetchers/fetch_tle.py
    python scripts/data_fetchers/fetch_tle.py --norad 39500 --out config/orbit/alsat2a.tle
    python scripts/data_fetchers/fetch_tle.py --force   # overwrite even if fresh

The script is completely ADDITIVE — it writes only to config/orbit/ and
never touches any existing training file.

Output format (plain-text TLE, 3 lines):
    ALSAT-2A
    1 39500U 14009B   24...
    2 39500  98.1...

Basilisk injection:  see env_alsat_real.py which reads this file and calls
    sgp4 to get r,v at the episode epoch, then sets
    scenario.satellites[0].scObject.hub.mHub.r_CN_NInit and v_CN_NInit.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ALSAT-2A NORAD catalogue number
DEFAULT_NORAD = 39500
DEFAULT_OUT   = os.path.join(ROOT, "config", "orbit", "alsat2a.tle")

# Celestrak GP data endpoint (JSON → reliable, no login needed)
# Falls back to legacy TLE text endpoint if JSON fails.
CELESTRAK_GP_URL   = "https://celestrak.org/SATCAT/groups/active.txt"
CELESTRAK_TLE_URL  = "https://celestrak.org/satcat/tle.php?CATNR={norad}"
CELESTRAK_GP_JSON  = "https://celestrak.org/SATCAT/records.php?CATNR={norad}&FORMAT=JSON"
SPACETRACK_TLE_URL = "https://www.space-track.org/basicspacedata/query/class/gp/NORAD_CAT_ID/{norad}/orderby/EPOCH%20desc/limit/1/format/tle"

# Age threshold: if existing TLE is fresher than this, skip download
MAX_AGE_HOURS = 24


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tle_age_hours(tle_path: str) -> float:
    """Return age of existing TLE file in hours (-1 if not present)."""
    if not os.path.exists(tle_path):
        return -1.0
    mtime = os.path.getmtime(tle_path)
    age_s = time.time() - mtime
    return age_s / 3600.0


def _parse_tle_epoch(line1: str) -> str:
    """Extract human-readable epoch from TLE line 1 (columns 18-32)."""
    try:
        epoch_raw = line1[18:32].strip()
        yr2 = int(epoch_raw[:2])
        doy = float(epoch_raw[2:])
        year = 2000 + yr2 if yr2 < 57 else 1900 + yr2
        import datetime as dt
        jan1 = dt.date(year, 1, 1)
        epoch_date = jan1 + dt.timedelta(days=doy - 1)
        return epoch_date.strftime("%Y-%m-%d")
    except Exception:
        return "unknown"


def _fetch_url(url: str, timeout: int = 20) -> str:
    """HTTP GET → str (raises on failure)."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ALSAT-RL-fetcher/1.0 (research; bensaid@esi-sba.dz)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _extract_tle_from_text(text: str, norad: int) -> tuple[str, str, str] | None:
    """
    Scan a block of TLE text for a satellite with the given NORAD id.
    Returns (name_line, line1, line2) or None.
    """
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    for i in range(len(lines) - 2):
        l1, l2 = lines[i + 1], lines[i + 2]
        if l1.startswith("1 ") and l2.startswith("2 "):
            try:
                cat = int(l1[2:7])
            except ValueError:
                continue
            if cat == norad:
                return lines[i], l1, l2
    return None


def _verify_tle(line1: str, line2: str) -> bool:
    """Sanity-check: correct format and matching NORAD IDs."""
    if not (line1.startswith("1 ") and line2.startswith("2 ")):
        return False
    try:
        n1 = int(line1[2:7])
        n2 = int(line2[2:7])
        return n1 == n2
    except ValueError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Download strategies (tried in order)
# ─────────────────────────────────────────────────────────────────────────────

def _try_celestrak_tle(norad: int) -> tuple[str, str, str] | None:
    """Strategy 1: Celestrak single-satellite TLE endpoint."""
    url = CELESTRAK_TLE_URL.format(norad=norad)
    try:
        text = _fetch_url(url)
        return _extract_tle_from_text(text, norad)
    except Exception as exc:
        print(f"  [WARN] Celestrak TLE endpoint: {exc}")
        return None


def _try_celestrak_active(norad: int) -> tuple[str, str, str] | None:
    """Strategy 2: Celestrak active satellite list (larger download ~2 MB)."""
    try:
        text = _fetch_url(CELESTRAK_GP_URL, timeout=40)
        return _extract_tle_from_text(text, norad)
    except Exception as exc:
        print(f"  [WARN] Celestrak active list: {exc}")
        return None


def _try_space_track(norad: int) -> tuple[str, str, str] | None:
    """
    Strategy 3: Space-Track.org (requires free account; uses env vars
    SPACETRACK_USER and SPACETRACK_PASS).  Skipped silently if not set.
    """
    user = os.environ.get("SPACETRACK_USER", "")
    pwd  = os.environ.get("SPACETRACK_PASS", "")
    if not user or not pwd:
        return None

    import urllib.parse
    login_url  = "https://www.space-track.org/ajaxauth/login"
    login_data = urllib.parse.urlencode({"identity": user, "password": pwd}).encode()

    try:
        import http.cookiejar
        cj      = http.cookiejar.CookieJar()
        opener  = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
        urllib.request.install_opener(opener)

        req = urllib.request.Request(login_url, data=login_data)
        opener.open(req, timeout=20)

        tle_url = SPACETRACK_TLE_URL.format(norad=norad)
        with opener.open(tle_url, timeout=20) as r:
            text = r.read().decode("utf-8", errors="replace")
        return _extract_tle_from_text(text, norad)
    except Exception as exc:
        print(f"  [WARN] Space-Track: {exc}")
        return None


def _synthetic_fallback(norad: int) -> tuple[str, str, str]:
    """
    Last resort: return a static reference TLE for ALSAT-2A.
    Epoch is approximate (2024); update periodically from Celestrak manually.
    This guarantees the orbit data is at least the correct orbit class.
    """
    print("  [WARN] All download strategies failed. Using static reference TLE.")
    print("         Run this script again when internet is available for a fresh TLE.")
    name = "ALSAT-2A"
    l1   = "1 39500U 14009B   24001.50000000  .00000030  00000-0  24135-4 0  9991"
    l2   = "2 39500  98.0900  10.2500 0001200  90.0000 270.0000 14.76431579534200"
    return name, l1, l2


# ─────────────────────────────────────────────────────────────────────────────
# Main download function
# ─────────────────────────────────────────────────────────────────────────────

def fetch_tle(norad: int = DEFAULT_NORAD,
              out_path: str = DEFAULT_OUT,
              force: bool = False,
              verbose: bool = True) -> str:
    """
    Download TLE for *norad* and save to *out_path*.

    Parameters
    ----------
    norad    : NORAD catalogue number (default 39500 = ALSAT-2A)
    out_path : destination file (created with parents if needed)
    force    : if False and file is < MAX_AGE_HOURS old, skip download
    verbose  : print progress

    Returns
    -------
    str  — path to written TLE file
    """
    age = _tle_age_hours(out_path)
    if not force and 0 <= age < MAX_AGE_HOURS:
        if verbose:
            print(f"  [TLE] {out_path} is {age:.1f}h old (< {MAX_AGE_HOURS}h) — skipping download.")
        return out_path

    if verbose:
        print(f"  [TLE] Downloading TLE for NORAD {norad}...")

    result = (
        _try_celestrak_tle(norad)
        or _try_celestrak_active(norad)
        or _try_space_track(norad)
    )

    if result is None:
        name, l1, l2 = _synthetic_fallback(norad)
    else:
        name, l1, l2 = result

    if not _verify_tle(l1, l2):
        print(f"  [WARN] TLE verification failed — using fallback.")
        name, l1, l2 = _synthetic_fallback(norad)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"{name}\n{l1}\n{l2}\n")

    epoch = _parse_tle_epoch(l1)
    if verbose:
        print(f"  [TLE] Saved → {out_path}")
        print(f"        Satellite : {name.strip()}")
        print(f"        NORAD     : {int(l1[2:7])}")
        print(f"        Epoch     : {epoch}")
        print(f"        Line 1    : {l1}")
        print(f"        Line 2    : {l2}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Orbit state vector from TLE (for Basilisk injection)
# ─────────────────────────────────────────────────────────────────────────────

def tle_to_state_at_epoch(tle_path: str,
                           epoch_jd: float | None = None
                           ) -> dict:
    """
    Use SGP4 to propagate the TLE to *epoch_jd* (Julian date).
    Returns a dict with ECI position r_m (m) and velocity v_ms (m/s).

    If epoch_jd is None, propagates to the TLE epoch itself (t=0).

    Requires:  pip install sgp4
    """
    try:
        from sgp4.api import Satrec, jday
    except ImportError:
        raise ImportError(
            "sgp4 not installed. Run:  pip install sgp4\n"
            "Or install from conda-forge:  conda install -c conda-forge sgp4"
        )

    with open(tle_path) as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    if len(lines) < 3:
        raise ValueError(f"TLE file {tle_path} must have 3 lines (name, L1, L2)")

    _name, l1, l2 = lines[0], lines[1], lines[2]
    sat = Satrec.twoline2rv(l1, l2)

    if epoch_jd is None:
        # Propagate at TLE epoch (t=0 from epoch)
        yr2  = int(l1[18:20])
        doy  = float(l1[20:32])
        year = 2000 + yr2 if yr2 < 57 else 1900 + yr2
        import datetime as dt
        jan1 = dt.datetime(year, 1, 1, tzinfo=timezone.utc)
        epoch_dt = jan1 + dt.timedelta(days=doy - 1)
        epoch_jd_val, epoch_fr = jday(epoch_dt.year, epoch_dt.month,
                                       epoch_dt.day, epoch_dt.hour,
                                       epoch_dt.minute,
                                       epoch_dt.second + epoch_dt.microsecond * 1e-6)
    else:
        epoch_jd_val = int(epoch_jd)
        epoch_fr     = epoch_jd - epoch_jd_val

    err, r_km, v_kms = sat.sgp4(epoch_jd_val, epoch_fr)
    if err != 0:
        raise RuntimeError(f"SGP4 error code {err} — TLE may be expired or corrupt.")

    import numpy as np
    r_m  = np.array(r_km,  dtype=float) * 1e3
    v_ms = np.array(v_kms, dtype=float) * 1e3

    return {
        "r_ECI_m":   r_m.tolist(),
        "v_ECI_ms":  v_ms.tolist(),
        "norad":     int(l1[2:7]),
        "name":      _name,
        "epoch_jd":  epoch_jd_val + epoch_fr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download TLE for ALSAT-2A and save to config/orbit/alsat2a.tle"
    )
    ap.add_argument("--norad",   type=int, default=DEFAULT_NORAD,
                    help=f"NORAD catalogue number (default: {DEFAULT_NORAD} = ALSAT-2A)")
    ap.add_argument("--out",     default=DEFAULT_OUT,
                    help="Output TLE file path")
    ap.add_argument("--force",   action="store_true",
                    help="Re-download even if existing TLE is fresh")
    ap.add_argument("--verify",  action="store_true",
                    help="After download, propagate TLE with SGP4 and print state vector")
    args = ap.parse_args()

    path = fetch_tle(norad=args.norad, out_path=args.out,
                     force=args.force, verbose=True)

    if args.verify:
        try:
            sv = tle_to_state_at_epoch(path)
            import numpy as np
            r = np.array(sv["r_ECI_m"])
            v = np.array(sv["v_ECI_ms"])
            alt_km = (np.linalg.norm(r) - 6378.137e3) / 1e3
            print(f"\n  SGP4 state at TLE epoch:")
            print(f"    r = [{r[0]/1e3:.3f}, {r[1]/1e3:.3f}, {r[2]/1e3:.3f}] km")
            print(f"    v = [{v[0]/1e3:.4f}, {v[1]/1e3:.4f}, {v[2]/1e3:.4f}] km/s")
            print(f"    altitude ≈ {alt_km:.1f} km  (expected ~686 km for ALSAT-2A)")
        except ImportError as exc:
            print(f"  [SKIP] SGP4 not installed: {exc}")


if __name__ == "__main__":
    main()
