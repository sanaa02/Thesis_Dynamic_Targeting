#!/usr/bin/env python3
"""
make_global_targets.py  --  generate a globally-distributed target set that
mirrors the schema of config/targets/global_45_targets.json.

Usage:
    python scripts/make_global_targets.py --n 20  --out config/targets/global_20_targets.json
    python scripts/make_global_targets.py --n 45  --out config/targets/global_45_targets.json

It reads your existing Algeria file to copy its exact key structure, then
replaces lat/lon with points sampled UNIFORMLY on the sphere (equal-area),
restricted to |lat| <= 55 deg so they stay within an agile sun-sync access band.
Priorities are sampled in the same range as your existing targets.
"""
import argparse, json, os, math, random

def load_schema(ref_path):
    with open(ref_path) as f:
        ref = json.load(f)
    # support either {"targets":[...]} or [...]
    items = ref["targets"] if isinstance(ref, dict) and "targets" in ref else ref
    wrapper_key = "targets" if isinstance(ref, dict) and "targets" in ref else None
    return items, wrapper_key, ref

def detect_keys(item):
    # find the lat/lon/name/priority field names actually used
    keys = {k.lower(): k for k in item.keys()}
    lat = next((keys[k] for k in keys if k in ("lat","latitude","lat_deg")), None)
    lon = next((keys[k] for k in keys if k in ("lon","lng","longitude","lon_deg")), None)
    name= next((keys[k] for k in keys if k in ("name","id","target","label")), None)
    prio= next((keys[k] for k in keys if "prior" in k), None)
    return lat, lon, name, prio

def sample_uniform_sphere(rng, lat_limit_deg=55.0):
    # equal-area: lat = arcsin(uniform(sin(-L), sin(L)))
    L = math.radians(lat_limit_deg)
    u = rng.uniform(math.sin(-L), math.sin(L))
    lat = math.degrees(math.asin(u))
    lon = rng.uniform(-180.0, 180.0)
    return lat, lon

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--ref", default="config/targets/global_45_targets.json",
                    help="existing file whose schema is copied")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lat-limit", type=float, default=55.0)
    args = ap.parse_args()

    items, wrapper_key, ref = load_schema(args.ref)
    template = dict(items[0])                      # copy ALL keys/defaults from a real entry
    lat_k, lon_k, name_k, prio_k = detect_keys(template)
    if lat_k is None or lon_k is None:
        raise SystemExit(f"Could not find lat/lon keys in {args.ref}: keys={list(template.keys())}")

    rng = random.Random(args.seed)
    # priority range from existing targets (fallback 0.5..1.0)
    if prio_k:
        ps = [float(it.get(prio_k, 0.7)) for it in items]
        pmin, pmax = min(ps), max(ps)
    else:
        pmin, pmax = 0.5, 1.0

    out_items = []
    for i in range(args.n):
        e = dict(template)                         # keep every other field as in the template
        lat, lon = sample_uniform_sphere(rng, args.lat_limit)
        e[lat_k] = round(lat, 4)
        e[lon_k] = round(lon, 4)
        if name_k: e[name_k] = f"global_{i:03d}"
        if prio_k: e[prio_k] = round(rng.uniform(pmin, pmax), 3)
        out_items.append(e)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if wrapper_key:
        payload = dict(ref); payload[wrapper_key] = out_items
    else:
        payload = out_items
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {args.n} global targets -> {args.out}")
    print(f"  schema keys copied from {args.ref}: {list(template.keys())}")
    print(f"  lat/lon keys: {lat_k}/{lon_k}  name:{name_k}  priority:{prio_k}")

if __name__ == "__main__":
    main()
