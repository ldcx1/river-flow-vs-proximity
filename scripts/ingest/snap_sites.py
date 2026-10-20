#!/usr/bin/env python3
"""Snap EEA Waterbase German monitoring sites to the nearest reach (<= 1 km).

Writes the snapped-site CSV to stdout. The shipped file is
data/de/national_sites_all_snapped.csv.

    python scripts/ingest/snap_sites.py \\
        --sites /path/Waterbase_..._S_WISE6_SpatialObject_DerivedData.csv \\
        > national_sites_all_snapped.csv
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from experiments.ingest import waterbase as wb  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data" / "de"
RADIUS_M = 1000.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", required=True, help="Waterbase SpatialObject CSV")
    ap.add_argument("--graph", default=str(DATA / "segment_graph_de.json"),
                    help="reach graph JSON")
    args = ap.parse_args()
    sites_csv = args.sites
    if not os.path.exists(sites_csv):
        print(f"Waterbase SpatialObject CSV not found: {sites_csv}", file=sys.stderr)
        return 1
    graph = json.load(open(args.graph, encoding="utf-8"))
    idx = wb.GridIndex(graph)
    w = csv.writer(sys.stdout)
    w.writerow(["site", "name", "basin", "segment", "river", "dist_m", "lat", "lon"])
    n_de = n = 0
    with open(sites_csv, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r.get("countryCode") != "DE":
                continue
            n_de += 1
            try:
                lat, lon = float(r["lat"]), float(r["lon"])
            except (TypeError, ValueError, KeyError):
                continue
            oid, d = idx.nearest(lat, lon)
            if oid and d <= RADIUS_M:
                w.writerow([r["monitoringSiteIdentifier"], r.get("monitoringSiteName", ""),
                            r.get("rbdName", ""), oid, graph[oid].get("river_name"),
                            round(d), lat, lon])
                n += 1
    print(f"DE sites: {n_de}; snapped (<= {RADIUS_M:.0f} m): {n}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
