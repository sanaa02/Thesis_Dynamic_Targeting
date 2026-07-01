
"""
cloud_lookup.py  —  Unified cloud cover lookup for Algeria
==========================================================
Returns cloud fraction ∈ [0,1] for any (lat, lon, utc_datetime) query.

Priority order:
  1. ERA5 NetCDF lookup  (exact hourly, 0.25° resolution)
  2. MODIS daily average (from existing global_45_clouds.json)
  3. Target-day climatological mean

All lookups are nearest-neighbor in both space and time.
"""
from __future__ import annotations
import datetime, os, json, pickle
from typing import Optional
import numpy as np


class CloudLookup:
    """
    Parameters
    ----------
    era5_lookup_pkl : path to era5_tcc_lookup.pkl (optional)
    modis_json      : path to global_45_clouds.json (fallback)
    """
    def __init__(
        self,
        era5_lookup_pkl: Optional[str] = None,
        modis_json: Optional[str]      = None,
    ):
        self._era5 = None
        self._era5_meta = {}
        if era5_lookup_pkl and os.path.exists(era5_lookup_pkl):
            with open(era5_lookup_pkl, 'rb') as f:
                data = pickle.load(f)
            self._era5 = data['lookup']
            print(f"[CloudLookup] ERA5 loaded: {len(self._era5)} hourly snapshots")
        else:
            print("[CloudLookup] ERA5 not found, using MODIS fallback")
        
        self._modis: dict = {}
        if modis_json and os.path.exists(modis_json):
            with open(modis_json) as f:
                self._modis = json.load(f)
        
        # Climatological fallback (monthly mean for Algeria ≈ 20%)
        self._climatology = {m: 0.15 + 0.05 * abs(m - 7) / 6
                              for m in range(1, 13)}

    def get(self, lat: float, lon: float, utc: datetime.datetime) -> float:
        """
        Return cloud fraction [0,1] at (lat, lon) at utc.
        Tries ERA5 first, then MODIS daily, then climatology.
        """
        if self._era5 is not None:
            val = self._query_era5(lat, lon, utc)
            if val is not None:
                return float(val)
        
        if self._modis:
            val = self._query_modis(lat, lon, utc)
            if val is not None:
                return float(val)
        
        # Climatological fallback
        return self._climatology.get(utc.month, 0.20)

    # ── ERA5 lookup ───────────────────────────────────────────────────────────

    def _query_era5(
        self, lat: float, lon: float, utc: datetime.datetime
    ) -> Optional[float]:
        # Round down to nearest hour
        key = utc.strftime("%Y-%m-%d %H")
        if key not in self._era5:
            # Try nearest available hour (±3h search)
            for dh in range(1, 4):
                for sign in (+1, -1):
                    alt = utc + datetime.timedelta(hours=dh * sign)
                    k2  = alt.strftime("%Y-%m-%d %H")
                    if k2 in self._era5:
                        key = k2
                        break
                else:
                    continue
                break
            else:
                return None
        
        snap = self._era5[key]
        lats = snap['lats']
        lons = snap['lons']
        tcc  = snap['tcc']
        
        # Nearest-neighbor in lat/lon
        i = int(np.argmin(np.abs(lats - lat)))
        j = int(np.argmin(np.abs(lons - lon)))
        val = float(np.clip(tcc[i, j], 0.0, 1.0))
        return val

    # ── MODIS daily fallback ──────────────────────────────────────────────────

    def _query_modis(
        self, lat: float, lon: float, utc: datetime.datetime
    ) -> Optional[float]:
        """
        global_45_clouds.json format:
          { "target_name": { "dates": [...], "cloud_fractions": [...] } }
        Find the target nearest to (lat, lon), then find the date nearest to utc.
        """
        if not self._modis:
            return None
        
        # Find the nearest target by geographic distance
        best_target = None
        best_dist   = float('inf')
        for tname, tdata in self._modis.items():
            tlat = tdata.get('lat', 33.0)
            tlon = tdata.get('lon',  3.0)
            d = (tlat - lat)**2 + (tlon - lon)**2
            if d < best_dist:
                best_dist   = d
                best_target = tname
        
        if best_target is None:
            return None
        
        td   = self._modis[best_target]
        dates = td.get('dates', [])
        cfs   = td.get('cloud_fractions', [])
        if not dates:
            return None
        
        # Find nearest date
        date_str = utc.strftime("%Y-%m-%d")
        try:
            idx = dates.index(date_str)
        except ValueError:
            # Find nearest by day difference
            target_day = utc.toordinal()
            diffs = [abs(datetime.datetime.strptime(d, "%Y-%m-%d").toordinal()
                         - target_day) for d in dates]
            idx = int(np.argmin(diffs))
        
        return float(np.clip(cfs[idx], 0.0, 1.0))