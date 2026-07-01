
#!/usr/bin/env python3
"""
fetch_real_events.py  --  Download real disaster events from FIRMS + GDACS
===========================================================================
Fetches active fire/flood/earthquake events for Algeria and writes them
to  data/real_events/firms_gdacs_algeria.json

FIRMS and GDACS results are ALWAYS combined into one file. If one source
fails, the other still contributes. The file is never empty -- static
fallback events fill any gaps.

Sources
-------
  FIRMS  (Fire Information for Resource Management System, NASA)
         https://firms.modaps.eosdis.nasa.gov/api/
         Provides VIIRS/MODIS active fire detections (last 24-48 h).
         Free API key required. Set env var FIRMS_API_KEY before running.

  GDACS  (Global Disaster Alert and Coordination System, UN)
         https://www.gdacs.org/xml/rss.xml  (RSS feed, no key required)
         Provides floods, earthquakes, tropical cyclones, tsunamis.
         Uses RSS XML feed (stable) -- the JSON API endpoint is unreliable.

Algeria bounding box: lat 18.9-37.1 N, lon -8.7-11.9 E

Usage
-----
    # GDACS only (no API key needed):
    python scripts/data_fetchers/fetch_real_events.py

    # FIRMS + GDACS:
    FIRMS_API_KEY=<your_key> python scripts/data_fetchers/fetch_real_events.py

    # Force re-fetch:
    python scripts/data_fetchers/fetch_real_events.py --force

    # Custom output path:
    python scripts/data_fetchers/fetch_real_events.py --out data/real_events/my_events.json

Output JSON schema
------------------
{
  "meta": { "_fetched_utc": "...", "_n_events": 80, ... },
  "events": [
    {
      "name":          "wildfire_001",
      "event_type":    "wildfire",
      "lat_deg":       35.12,
      "lon_deg":        2.44,
      "priority":      0.92,
      "source":        "FIRMS_VIIRS",
      "detected_utc":  "2024-01-15T08:22Z",
      "confidence":    "high"
    },
    ...
  ]
}
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_OUT = os.path.join(ROOT, "data", "real_events", "firms_gdacs_algeria.json")

# Algeria bounding box
ALG_LAT_MIN, ALG_LAT_MAX = 18.9, 37.1
ALG_LON_MIN, ALG_LON_MAX = -8.7, 11.9

MAX_AGE_HOURS = 6

# Deduplication: same source same location within this distance is a duplicate
DEDUP_SAME_SOURCE_DEG  = 0.05   # ~5.5 km -- only merge very close same-source points
DEDUP_CROSS_SOURCE_DEG = 0.30   # ~33 km -- merge across sources (fire + flood same area)


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helper
# ─────────────────────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ALSAT-RL-fetcher/1.0 (research)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _get_json(url: str, timeout: int = 30) -> Any:
    text = _get(url, timeout)
    if not text.strip():
        raise ValueError("Empty response from server")
    return json.loads(text)


# ─────────────────────────────────────────────────────────────────────────────
# FIRMS  (active fires via CSV API)
# ─────────────────────────────────────────────────────────────────────────────

FIRMS_CSV_URL = (
    "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
    "/{api_key}/VIIRS_SNPP_NRT/{wlon},{slat},{elon},{nlat}/10"
)
FIRMS_NOAA20_URL = (
    "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
    "/{api_key}/VIIRS_NOAA20_NRT/{wlon},{slat},{elon},{nlat}/10"
)
FIRMS_MODIS_URL = (
    "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
    "/{api_key}/MODIS_NRT/{wlon},{slat},{elon},{nlat}/10"
)

def _fetch_firms(api_key: str,
                 lat_min: float = ALG_LAT_MIN, lat_max: float = ALG_LAT_MAX,
                 lon_min: float = ALG_LON_MIN, lon_max: float = ALG_LON_MAX,
                 verbose: bool = True) -> list[dict]:
    """Fetch fire detections from VIIRS SNPP + NOAA-20 + MODIS NRT for the last 10 days."""
    all_lines = []
    sensors = [
        ("VIIRS_SNPP",  FIRMS_CSV_URL),
        ("VIIRS_NOAA20", FIRMS_NOAA20_URL),
        ("MODIS_NRT",   FIRMS_MODIS_URL),
    ]
    header = None
    for sensor_name, url_template in sensors:
        url = url_template.format(
            api_key=api_key,
            slat=lat_min, nlat=lat_max,
            wlon=lon_min, elon=lon_max,
        )
        if verbose:
            print(f"  [FIRMS] Fetching {sensor_name}... {url[:80]}...")
        try:
            text = _get(url)
            raw_lines = text.strip().splitlines()
            if len(raw_lines) >= 2:
                if header is None:
                    header = raw_lines[0]
                    all_lines.extend(raw_lines[1:])
                else:
                    all_lines.extend(raw_lines[1:])   # skip repeated header
                if verbose:
                    print(f"  [FIRMS] {sensor_name}: {len(raw_lines)-1} detections")
        except Exception as exc:
            if verbose:
                print(f"  [FIRMS] {sensor_name} failed: {exc}")
    if not all_lines or header is None:
        print("  [FIRMS] No fire detections from any sensor.")
        return []
    lines = [header] + all_lines

    header = [h.strip().lower() for h in lines[0].split(",")]
    events = []
    for raw in lines[1:]:
        cols = raw.split(",")
        if len(cols) < len(header):
            continue
        row = dict(zip(header, cols))

        try:
            lat = float(row.get("latitude", 0))
            lon = float(row.get("longitude", 0))
        except ValueError:
            continue

        # Filter to bounding box (FIRMS should already do this, but double-check)
        if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
            continue

        conf_raw = row.get("confidence", "n").strip().lower()
        conf_str = {"h": "high", "n": "nominal", "l": "low"}.get(conf_raw, "nominal")
        priority = {"high": 0.90, "nominal": 0.65, "low": 0.35}.get(conf_str, 0.5)

        try:
            frp = float(row.get("frp", 0))
            if frp > 100:
                priority = min(1.0, priority + 0.08)
        except ValueError:
            pass

        acq_date = row.get("acq_date", "").strip()
        acq_time = row.get("acq_time", "0000").strip().zfill(4)
        detected = (f"{acq_date}T{acq_time[:2]}:{acq_time[2:]}Z"
                    if acq_date else "unknown")

        events.append({
            "name":         f"FIRMS_fire_{len(events)+1:03d}",
            "event_type":   "wildfire",
            "lat_deg":      round(lat, 4),
            "lon_deg":      round(lon, 4),
            "priority":     round(priority, 3),
            "source":       "FIRMS_VIIRS",
            "detected_utc": detected,
            "confidence":   conf_str,
        })

    if verbose:
        print(f"  [FIRMS] {len(events)} fire detections found.")
    return events


# ─────────────────────────────────────────────────────────────────────────────
# GDACS  (floods, earthquakes, cyclones -- via RSS XML feed)
# ─────────────────────────────────────────────────────────────────────────────

# These RSS endpoints are stable -- the JSON /geteventlist/ API is unreliable
GDACS_RSS_URLS = [
    "https://www.gdacs.org/xml/rss.xml",           # global all-events feed
    "https://www.gdacs.org/xml/rss_24h.xml",        # last 24 h
    "https://www.gdacs.org/xml/rss_7d.xml",         # last 7 days (more events)
]

# XML namespace used by GDACS RSS
GDACS_NS = {
    "geo":   "http://www.w3.org/2003/01/geo/wgs84_pos#",
    "gdacs": "http://www.gdacs.org",
    "dc":    "http://purl.org/dc/elements/1.1/",
}

_GDACS_TYPE_MAP = {
    "EQ": ("earthquake", 0.80),
    "FL": ("flood",      0.75),
    "TC": ("cyclone",    0.85),
    "TS": ("tsunami",    0.95),
    "VO": ("eruption",   0.70),
    "DR": ("drought",    0.55),
    "WF": ("wildfire",   0.80),
}
_GDACS_ALERT_BOOST = {"green": 0.0, "orange": 0.10, "red": 0.20}


def _parse_gdacs_rss(xml_text: str,
                     lat_min: float, lat_max: float,
                     lon_min: float, lon_max: float) -> list[dict]:
    """
    Parse GDACS RSS XML and return events within the bounding box.

    GDACS RSS item structure (example):
        <item>
          <title>...</title>
          <pubDate>Mon, 15 Jan 2024 08:00:00 GMT</pubDate>
          <geo:lat>36.5</geo:lat>
          <geo:long>3.0</geo:long>
          <gdacs:eventtype>EQ</gdacs:eventtype>
          <gdacs:alertlevel>Orange</gdacs:alertlevel>
          <gdacs:country>Algeria</gdacs:country>
          <gdacs:eventid>1234567</gdacs:eventid>
        </item>
    """
    # Strip XML declaration if encoding differs (ElementTree handles UTF-8 only)
    if xml_text.startswith("<?xml"):
        xml_text = xml_text[xml_text.index("?>") + 2:]

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"GDACS RSS XML parse error: {exc}")

    events = []
    for item in root.iter("item"):
        # Extract lat/lon -- GDACS uses both geo:lat/long and georss:point
        lat = lon = None

        for tag in ("geo:lat", "{http://www.w3.org/2003/01/geo/wgs84_pos#}lat"):
            el = item.find(tag)
            if el is not None and el.text:
                try:
                    lat = float(el.text.strip())
                    break
                except ValueError:
                    pass

        for tag in ("geo:long", "{http://www.w3.org/2003/01/geo/wgs84_pos#}long"):
            el = item.find(tag)
            if el is not None and el.text:
                try:
                    lon = float(el.text.strip())
                    break
                except ValueError:
                    pass

        # Try georss:point as fallback ("lat lon" space-separated)
        if lat is None or lon is None:
            for tag in ("georss:point",
                        "{http://www.georss.org/georss}point"):
                el = item.find(tag)
                if el is not None and el.text:
                    parts = el.text.strip().split()
                    if len(parts) == 2:
                        try:
                            lat, lon = float(parts[0]), float(parts[1])
                            break
                        except ValueError:
                            pass

        if lat is None or lon is None:
            continue

        # Filter to Algeria bounding box
        if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
            continue

        # Event type
        etype_raw = ""
        for tag in ("gdacs:eventtype",
                    "{http://www.gdacs.org}eventtype"):
            el = item.find(tag)
            if el is not None and el.text:
                etype_raw = el.text.strip().upper()
                break
        etype, base_prio = _GDACS_TYPE_MAP.get(etype_raw, ("disaster", 0.60))

        # Alert level
        alert = "green"
        for tag in ("gdacs:alertlevel",
                    "{http://www.gdacs.org}alertlevel"):
            el = item.find(tag)
            if el is not None and el.text:
                alert = el.text.strip().lower()
                break
        boost    = _GDACS_ALERT_BOOST.get(alert, 0.0)
        priority = min(1.0, base_prio + boost)

        # Event ID
        eid = ""
        for tag in ("gdacs:eventid",
                    "{http://www.gdacs.org}eventid"):
            el = item.find(tag)
            if el is not None and el.text:
                eid = el.text.strip()
                break
        if not eid:
            eid = str(len(events) + 1)

        # Date
        pub_el = item.find("pubDate")
        date_str = pub_el.text.strip() if pub_el is not None and pub_el.text else "unknown"

        events.append({
            "name":         f"GDACS_{etype}_{eid}",
            "event_type":   etype,
            "lat_deg":      round(lat, 4),
            "lon_deg":      round(lon, 4),
            "priority":     round(priority, 3),
            "source":       "GDACS",
            "detected_utc": date_str,
            "confidence":   alert,
        })

    return events


def _fetch_gdacs(lat_min: float = ALG_LAT_MIN, lat_max: float = ALG_LAT_MAX,
                 lon_min: float = ALG_LON_MIN, lon_max: float = ALG_LON_MAX,
                 verbose: bool = True) -> list[dict]:
    """
    Fetch GDACS alerts for Algeria via RSS XML feed.
    Tries multiple RSS URLs in order; returns list of event dicts.
    """
    if verbose:
        print("  [GDACS] Fetching disaster alerts via RSS feed...")

    last_error = ""
    for url in GDACS_RSS_URLS:
        try:
            xml_text = _get(url, timeout=30)
            if not xml_text.strip():
                last_error = f"Empty response from {url}"
                continue
            events = _parse_gdacs_rss(xml_text, lat_min, lat_max, lon_min, lon_max)
            if verbose:
                print(f"  [GDACS] {len(events)} alerts found in Algeria bounding box "
                      f"(from {url.split('/')[-1]})")
            return events
        except Exception as exc:
            last_error = str(exc)
            if verbose:
                print(f"  [GDACS] {url.split('/')[-1]} failed: {exc}")

    if verbose:
        print(f"  [GDACS] All RSS feeds failed ({last_error}) -- "
              f"using static fallback.")
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Historical Algeria hotspots (permanent fallback -- always merged in)
# ─────────────────────────────────────────────────────────────────────────────

_ALGERIA_HISTORICAL = [
    # Northern Algeria wildfire hotspots (fires occur every summer)
    {"name": "hist_fire_kabylie_1",   "event_type": "wildfire",   "lat_deg": 36.55, "lon_deg":  4.05, "priority": 0.85},
    {"name": "hist_fire_kabylie_2",   "event_type": "wildfire",   "lat_deg": 36.70, "lon_deg":  3.85, "priority": 0.82},
    {"name": "hist_fire_jijel",       "event_type": "wildfire",   "lat_deg": 36.82, "lon_deg":  5.77, "priority": 0.80},
    {"name": "hist_fire_skikda",      "event_type": "wildfire",   "lat_deg": 36.88, "lon_deg":  6.90, "priority": 0.78},
    {"name": "hist_fire_bejaia",      "event_type": "wildfire",   "lat_deg": 36.75, "lon_deg":  5.08, "priority": 0.82},
    {"name": "hist_fire_tlemcen",     "event_type": "wildfire",   "lat_deg": 34.88, "lon_deg": -1.31, "priority": 0.65},
    {"name": "hist_fire_setif",       "event_type": "wildfire",   "lat_deg": 36.19, "lon_deg":  5.41, "priority": 0.70},
    {"name": "hist_fire_chlef",       "event_type": "wildfire",   "lat_deg": 36.16, "lon_deg":  1.34, "priority": 0.68},
    {"name": "hist_fire_constantine", "event_type": "wildfire",   "lat_deg": 36.36, "lon_deg":  6.62, "priority": 0.72},
    {"name": "hist_fire_souk_ahras",  "event_type": "wildfire",   "lat_deg": 36.28, "lon_deg":  7.95, "priority": 0.69},
    # Flooding zones (Cheliff river basin, Seybouse river)
    {"name": "hist_flood_cheliff",    "event_type": "flood",      "lat_deg": 36.17, "lon_deg":  1.33, "priority": 0.72},
    {"name": "hist_flood_seybouse",   "event_type": "flood",      "lat_deg": 36.45, "lon_deg":  7.60, "priority": 0.68},
    {"name": "hist_flood_hodna",      "event_type": "flood",      "lat_deg": 35.40, "lon_deg":  4.70, "priority": 0.60},
    {"name": "hist_flood_macta",      "event_type": "flood",      "lat_deg": 35.78, "lon_deg": -0.28, "priority": 0.58},
    # Earthquake zones (northern Tell Atlas seismic belt)
    {"name": "hist_quake_oranie",     "event_type": "earthquake", "lat_deg": 35.70, "lon_deg": -0.65, "priority": 0.75},
    {"name": "hist_quake_ain_t",      "event_type": "earthquake", "lat_deg": 35.30, "lon_deg": -1.14, "priority": 0.70},
    {"name": "hist_quake_boumerdes",  "event_type": "earthquake", "lat_deg": 36.76, "lon_deg":  3.48, "priority": 0.80},
    {"name": "hist_quake_chlef",      "event_type": "earthquake", "lat_deg": 36.09, "lon_deg":  1.25, "priority": 0.73},
    # Industrial / infrastructure hazards
    {"name": "hist_industrial_skikda","event_type": "wildfire",   "lat_deg": 36.90, "lon_deg":  6.82, "priority": 0.62},
    {"name": "hist_industrial_arzew", "event_type": "wildfire",   "lat_deg": 35.84, "lon_deg": -0.30, "priority": 0.60},
]


def _static_fallback(verbose: bool = True) -> list[dict]:
    """Return historical Algeria hotspots with current timestamp."""
    if verbose:
        print("  [FALLBACK] Using 20 historical Algeria hazard locations.")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    return [
        {**ev, "source": "historical_fallback", "detected_utc": now, "confidence": "nominal"}
        for ev in _ALGERIA_HISTORICAL
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication (source-aware)
# ─────────────────────────────────────────────────────────────────────────────

def _deduplicate(events: list[dict]) -> list[dict]:
    """
    Remove near-duplicate events using source-aware thresholds:
    - Same source within 0.05 deg (~5 km): merge, keep higher priority
    - Different sources within 0.30 deg (~33 km): merge, keep higher priority
    Historical fallback events are only removed if a real event is within 0.30 deg.
    """
    kept: list[dict] = []

    for ev in events:
        lat, lon = ev["lat_deg"], ev["lon_deg"]
        src = ev.get("source", "")
        is_historical = "fallback" in src or "historical" in src

        merged = False
        for k in kept:
            k_lat, k_lon = k["lat_deg"], k["lon_deg"]
            k_src = k.get("source", "")
            k_hist = "fallback" in k_src or "historical" in k_src

            dist = math.hypot(lat - k_lat, lon - k_lon)

            # Threshold: same source = tight, cross source = looser
            if src == k_src:
                thresh = DEDUP_SAME_SOURCE_DEG
            else:
                thresh = DEDUP_CROSS_SOURCE_DEG

            if dist < thresh:
                # Keep higher priority; prefer real data over historical
                ev_prio = ev.get("priority", 0)
                k_prio  = k.get("priority", 0)
                if (not k_hist and is_historical):
                    pass   # keep existing real event, discard historical
                elif ev_prio > k_prio:
                    k.update(ev)
                merged = True
                break

        if not merged:
            kept.append(dict(ev))

    return kept


import math


# ─────────────────────────────────────────────────────────────────────────────
# Main fetch function
# ─────────────────────────────────────────────────────────────────────────────

def fetch_events(out_path: str = DEFAULT_OUT,
                 force:    bool = False,
                 verbose:  bool = True) -> str:
    """
    Fetch real events from FIRMS + GDACS and save to *out_path*.
    Always merges all available sources. Falls back to historical data if APIs fail.

    Reads FIRMS_API_KEY from environment (optional).
    Returns path to the written JSON file.
    """
    # Skip if fresh
    if not force and os.path.exists(out_path):
        age_h = (time.time() - os.path.getmtime(out_path)) / 3600.0
        if age_h < MAX_AGE_HOURS:
            if verbose:
                print(f"  [EVENTS] {out_path} is {age_h:.1f}h old -- skipping fetch.")
            return out_path

    firms_events: list[dict] = []
    gdacs_events: list[dict] = []

    # ── FIRMS (fires) ──────────────────────────────────────────────────────
    api_key = os.environ.get("FIRMS_API_KEY", "").strip()
    if api_key:
        firms_events = _fetch_firms(api_key, verbose=verbose)
        if not firms_events and verbose:
            print("  [FIRMS] No events returned (check API key or try --force).")
    else:
        if verbose:
            print("  [FIRMS] FIRMS_API_KEY not set -- skipping fire data.")
            print("          Register free at: https://firms.modaps.eosdis.nasa.gov/api/")

    # ── GDACS (floods, quakes, etc.) ───────────────────────────────────────
    gdacs_events = _fetch_gdacs(verbose=verbose)

    # ── Merge FIRMS + GDACS ────────────────────────────────────────────────
    combined = firms_events + gdacs_events

    if verbose:
        print(f"  [MERGE] FIRMS: {len(firms_events)}  GDACS: {len(gdacs_events)}  "
              f"combined before dedup: {len(combined)}")

    # Deduplicate
    combined = _deduplicate(combined)

    if verbose:
        print(f"  [MERGE] After deduplication: {len(combined)} events")

    # ── Always append historical fallback (adds variety for underrepresented types) ─
    # Historical events fill gaps (earthquake zones, flood plains not in real-time data)
    hist_events = _static_fallback(verbose=False)
    combined_with_hist = _deduplicate(combined + hist_events)

    n_real      = len(combined)
    n_with_hist = len(combined_with_hist)
    if verbose and n_with_hist > n_real:
        print(f"  [MERGE] Added {n_with_hist - n_real} historical fallback events "
              f"(no real-time equivalent in region)")

    events = combined_with_hist

    # Assign clean sequential names for logging
    for i, ev in enumerate(events):
        if not ev.get("name", "").startswith("hist_"):
            ev["name"] = f"{ev['event_type']}_{i+1:03d}"

    # ── Write output ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    meta = {
        "_fetched_utc":  datetime.now(timezone.utc).isoformat(),
        "_n_events":     len(events),
        "_n_firms":      len(firms_events),
        "_n_gdacs":      len(gdacs_events),
        "_n_historical": n_with_hist - n_real,
        "_bbox":         [ALG_LAT_MIN, ALG_LAT_MAX, ALG_LON_MIN, ALG_LON_MAX],
    }
    output = {"meta": meta, "events": events}

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    if verbose:
        by_type: dict[str, int] = {}
        by_src:  dict[str, int] = {}
        for ev in events:
            t = ev.get("event_type", "unknown")
            s = ev.get("source",     "unknown")
            by_type[t] = by_type.get(t, 0) + 1
            by_src[s]  = by_src.get(s,  0) + 1

        print(f"\n  [EVENTS] Saved {len(events)} events -> {out_path}")
        print(f"  By type:")
        for t, n in sorted(by_type.items()):
            print(f"    {t:<15}: {n}")
        print(f"  By source:")
        for s, n in sorted(by_src.items()):
            print(f"    {s:<25}: {n}")

    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fetch real disaster events for Algeria from FIRMS + GDACS"
    )
    ap.add_argument("--out",   default=DEFAULT_OUT, help="Output JSON path")
    ap.add_argument("--force", action="store_true",  help="Re-fetch even if file is fresh")
    ap.add_argument("--quiet", action="store_true",  help="Suppress progress output")
    args = ap.parse_args()

    fetch_events(out_path=args.out, force=args.force, verbose=not args.quiet)


if __name__ == "__main__":
    main()

