"""Facility density (paper Sec. V-B and Limitations: 49% of German reaches and 18%
of Romanian reaches lie within 1.5 km of an OSM facility).

Counts reaches whose centroid is within --radius of any industrial or wastewater
facility, per country, and the share of German exceedance months on such reaches.

    python scripts/facility_shares.py > results/national/facility_shares.txt   # seconds
"""
import argparse
import csv
import json

import numpy as np
from scipy.spatial import cKDTree

import _bootstrap

R_EARTH_M = 6_371_000.0


def unit_xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def near_facility(graph_path, osm_path, radius_m):
    with open(graph_path, encoding="utf-8") as fh:
        g = json.load(fh)
    with open(osm_path, encoding="utf-8") as fh:
        fac = json.load(fh)
    ids = [o for o, v in g.items() if v.get("lat") is not None]
    reaches = unit_xyz([g[o]["lat"] for o in ids], [g[o]["lon"] for o in ids])
    tree = cKDTree(unit_xyz([f["lat"] for f in fac], [f["lon"] for f in fac]))
    chord = 2 * np.sin(radius_m / R_EARTH_M / 2)
    d, _ = tree.query(reaches, k=1, distance_upper_bound=chord)
    return dict(zip(ids, np.isfinite(d))), len(fac)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--radius", type=float, default=1500.0, help="metres")
    args = ap.parse_args()

    for country, data, gname, oname in (
            ("Germany", _bootstrap.data_de, "segment_graph_de.json", "osm_industrial_de.json"),
            ("Romania", _bootstrap.data_ro, "segment_graph_ro.json", "osm_industrial_ro.json")):
        near, n_fac = near_facility(data(gname), data(oname), args.radius)
        share = np.mean(list(near.values()))
        print(f"{country}: {n_fac} facilities, {len(near)} reaches, "
              f"{share:.2%} within {args.radius:.0f} m of a facility")
        if country == "Germany":
            pos = near_pos = 0
            with _bootstrap.open_text(data("features_national_de.csv")) as fh:
                for row in csv.DictReader(fh):
                    if row["label"] == "1":
                        pos += 1
                        near_pos += near.get(row["object_id"], False)
            print(f"  exceedance months on such reaches: {near_pos}/{pos} = {near_pos / pos:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
