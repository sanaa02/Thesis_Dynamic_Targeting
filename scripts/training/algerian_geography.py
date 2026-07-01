#!/usr/bin/env python3
"""
algerian_geography.py  --  ALSAT-EO-1  Algeria/Wilaya Lookup
=============================================================
Given (lat, lon) returns the Algerian wilaya name (or "outside Algeria").
Uses simple bounding-box lookup — no external dependencies.

Algeria bounding box:  lat [18.97, 37.09]  lon [-8.68, 11.99]
"""
from __future__ import annotations

# Each entry: (wilaya_number, name, lat_min, lat_max, lon_min, lon_max)
# Boxes are approximate — good enough for orbit verification logs.
_WILAYAS = [
    (1,  "Adrar",          20.0, 29.5, -2.9,  2.9),
    (2,  "Chlef",          35.8, 36.7,  0.9,  2.0),
    (3,  "Laghouat",       32.5, 34.5,  2.0,  3.5),
    (4,  "Oum El Bouaghi", 35.5, 36.3,  6.5,  8.0),
    (5,  "Batna",          35.0, 36.2,  5.7,  7.0),
    (6,  "Béjaïa",         36.4, 37.1,  4.5,  5.8),
    (7,  "Biskra",         33.5, 35.5,  5.5,  7.2),
    (8,  "Béchar",         27.0, 32.5, -2.9,  2.2),
    (9,  "Blida",          36.1, 36.7,  2.3,  3.1),
    (10, "Bouira",         35.9, 36.7,  3.5,  4.5),
    (11, "Tamanrasset",    18.9, 25.5,  4.5,  9.5),
    (12, "Tébessa",        34.5, 36.0,  7.5,  8.5),
    (13, "Tlemcen",        34.5, 35.5, -1.8,  0.3),
    (14, "Tiaret",         34.8, 35.9,  1.0,  2.5),
    (15, "Tizi Ouzou",     36.4, 37.1,  3.8,  4.9),
    (16, "Alger",          36.5, 37.1,  2.8,  3.4),
    (17, "Djelfa",         33.5, 35.5,  2.0,  4.0),
    (18, "Jijel",          36.5, 37.1,  5.4,  6.2),
    (19, "Sétif",          35.6, 36.7,  5.0,  6.3),
    (20, "Saïda",          34.5, 35.5, -0.3,  0.8),
    (21, "Skikda",         36.5, 37.1,  6.5,  7.3),
    (22, "Sidi Bel Abbès", 34.5, 35.5, -1.2,  0.2),
    (23, "Annaba",         36.5, 37.1,  7.2,  8.3),
    (24, "Guelma",         36.3, 37.0,  7.0,  8.0),
    (25, "Constantine",    36.0, 36.6,  6.4,  7.0),
    (26, "Médéa",          35.6, 36.6,  2.3,  3.6),
    (27, "Mostaganem",     35.8, 36.4,  0.0,  0.9),
    (28, "M'Sila",         34.5, 36.0,  4.0,  5.5),
    (29, "Mascara",        35.0, 35.9,  0.0,  1.0),
    (30, "Ouargla",        29.5, 33.0,  5.0,  9.5),
    (31, "Oran",           35.3, 36.1, -1.2,  0.5),
    (32, "El Bayadh",      32.5, 34.5, -0.2,  2.0),
    (33, "Illizi",         22.0, 29.5,  7.0, 12.0),
    (34, "Bordj Bou Arréridj", 35.5, 36.6, 4.5, 5.6),
    (35, "Boumerdès",      36.5, 37.1,  3.4,  4.0),
    (36, "El Tarf",        36.5, 37.1,  8.0,  8.8),
    (37, "Tindouf",        19.0, 29.0, -8.7, -2.5),
    (38, "Tissemsilt",     35.4, 36.2,  1.5,  2.3),
    (39, "El Oued",        32.0, 34.5,  6.7,  8.5),
    (40, "Khenchela",      35.0, 36.0,  6.9,  8.0),
    (41, "Souk Ahras",     36.0, 37.0,  7.8,  8.5),
    (42, "Tipasa",         36.2, 36.8,  2.0,  2.8),
    (43, "Mila",           36.0, 36.7,  6.0,  6.8),
    (44, "Aïn Defla",      35.9, 36.7,  1.7,  2.5),
    (45, "Naâma",          32.5, 34.5, -1.3,  0.5),
    (46, "Aïn Témouchent", 35.2, 35.8, -1.5, -0.3),
    (47, "Ghardaïa",       31.5, 33.0,  3.0,  4.5),
    (48, "Relizane",       35.5, 36.5,  0.5,  1.3),
    (49, "Timimoun",       27.5, 30.5,  0.0,  2.5),
    (50, "Bordj Badji Mokhtar", 19.0, 25.0, -0.3, 2.5),
    (51, "Ouled Djellal",  33.5, 35.0,  4.5,  6.0),
    (52, "Béni Abbès",     28.0, 32.0, -2.5, -0.2),
    (53, "In Salah",       26.0, 29.5,  2.3,  4.8),
    (54, "In Guezzam",     19.5, 22.5,  5.0,  8.5),
    (55, "Touggourt",      33.0, 34.5,  5.8,  7.3),
    (56, "Djanet",         23.0, 26.5,  8.5, 10.0),
    (57, "El M'Ghair",     33.5, 34.5,  5.5,  6.5),
    (58, "El Meniaa",      28.5, 31.0,  2.5,  4.5),
]

# Neighbouring country bounding boxes (approximate)
_NEIGHBOURS = [
    ("Libya",     23.0, 37.4,  9.5, 25.2),
    ("Tunisia",   30.5, 37.5,  8.1, 11.6),
    ("Morocco",   27.7, 35.9, -8.7, -1.0),
    ("Mauritania",14.7, 26.8,-17.1, -4.8),
    ("Mali",       9.5, 24.9,-12.2,  4.3),
    ("Niger",     11.7, 23.5,  2.8, 16.0),
    ("Western Sahara", 20.8, 27.7, -17.1, -8.7),
    ("Mediterranean Sea", 37.0, 43.0, -5.0, 15.0),
]


def lookup_location(lat: float, lon: float) -> dict:
    """
    Return a location dict for (lat, lon):
      {
        'country':     'Algeria' | 'Libya' | ...
        'wilaya':      'Alger' | None       (None when not in Algeria)
        'wilaya_num':  16 | None
        'label':       'Alger (16)' | 'Mediterranean Sea'
      }
    """
    # Check Algeria wilayas first (most specific)
    if 18.9 <= lat <= 37.1 and -8.7 <= lon <= 12.0:
        for wnum, wname, la0, la1, lo0, lo1 in _WILAYAS:
            if la0 <= lat <= la1 and lo0 <= lon <= lo1:
                return {
                    "country":    "Algeria",
                    "wilaya":     wname,
                    "wilaya_num": wnum,
                    "label":      f"{wname} (W{wnum:02d})",
                }
        # In Algeria bounding box but no wilaya matched
        return {"country": "Algeria", "wilaya": "unknown", "wilaya_num": None,
                "label": "Algeria (unknown wilaya)"}

    # Neighbouring countries
    for name, la0, la1, lo0, lo1 in _NEIGHBOURS:
        if la0 <= lat <= la1 and lo0 <= lon <= lo1:
            return {"country": name, "wilaya": None, "wilaya_num": None,
                    "label": name}

    return {"country": "International", "wilaya": None, "wilaya_num": None,
            "label": f"International ({lat:+.1f}°, {lon:+.1f}°)"}


def ecef_to_latlon(r_ecef: "np.ndarray") -> tuple[float, float, float]:
    """
    Convert ECEF position vector (m) → (lat_deg, lon_deg, alt_km).
    Uses WGS-84 oblate Earth approximation.
    """
    import math
    import numpy as np
    x, y, z = float(r_ecef[0]), float(r_ecef[1]), float(r_ecef[2])
    a  = 6_378_137.0        # WGS-84 semi-major axis (m)
    b  = 6_356_752.3142     # WGS-84 semi-minor axis (m)
    e2 = 1.0 - (b/a)**2    # eccentricity squared
    p  = math.sqrt(x**2 + y**2)
    lon_deg = math.degrees(math.atan2(y, x))
    lat_rad = math.atan2(z, p * (1 - e2))   # initial estimate
    for _ in range(5):
        N = a / math.sqrt(1 - e2 * math.sin(lat_rad)**2)
        lat_rad = math.atan2(z + e2 * N * math.sin(lat_rad), p)
    N   = a / math.sqrt(1 - e2 * math.sin(lat_rad)**2)
    alt = p / math.cos(lat_rad) - N if abs(math.cos(lat_rad)) > 1e-6 else abs(z)/math.sin(lat_rad) - N*(1-e2)
    return math.degrees(lat_rad), lon_deg, alt / 1000.0   # → km


if __name__ == "__main__":
    # Quick test
    tests = [
        (36.74, 3.06,  "Alger capital"),
        (35.70, 0.63,  "Mostaganem"),
        (19.10, 5.50,  "Tamanrasset"),
        (38.0,  9.0,   "Mediterranean"),
        (-10.0, 20.0,  "Sub-Saharan Africa"),
    ]
    for lat, lon, desc in tests:
        loc = lookup_location(lat, lon)
        print(f"  {desc:25s}  ({lat:+6.2f}, {lon:+7.2f})  →  {loc['label']}")