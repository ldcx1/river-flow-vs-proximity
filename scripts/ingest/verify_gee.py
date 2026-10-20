#!/usr/bin/env python3
"""Live Earth Engine round-trip: re-fetch a few Sentinel-2 passes and confirm
they correspond to the shipped bundle (``data/de/pass_obs_national.csv``).

For a random sample of ``(object_id, date)`` rows it re-fetches that date's S2
indices over the reach, compares TURBIDITY / NDCI to the committed values, and
recomputes ``risk_score`` with the ported pure formula. Credentials come from the
environment (GEE_SERVICE_ACCOUNT_KEY, GEE_PROJECT). If --asset (an EE
FeatureCollection of the reach geometries, keyed by ``OBJECT_ID``) is given the
reduction is over the true reach geometry (exact match expected). Otherwise a
small buffer around the reach centroid is used, which corresponds closely but
not bit-for-bit.

    GEE_SERVICE_ACCOUNT_KEY=/path/key.json GEE_PROJECT=proj \\
        python scripts/ingest/verify_gee.py --n 8 --tol 0.06
"""
import argparse
import csv
import json
import random
import statistics
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from experiments.ingest import gee, indices  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data" / "de"
CENTROID_BUFFER_M = 150


def _region(ee, oid, graph, asset):
    if asset:
        return ee.FeatureCollection(asset).filter(ee.Filter.eq("OBJECT_ID", oid)).geometry()
    n = graph[oid]
    return ee.Geometry.Point([n["lon"], n["lat"]]).buffer(CENTROID_BUFFER_M)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=8, help="passes to sample")
    ap.add_argument("--tol", type=float, default=0.06, help="max median |diff| to pass")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--asset", default=None,
                    help="EE FeatureCollection of reach geometries keyed by OBJECT_ID "
                         "(default: a centroid buffer)")
    ap.add_argument("--graph", default=str(DATA / "segment_graph_de.json"))
    ap.add_argument("--passes", default=str(DATA / "pass_obs_national.csv"),
                    help="shipped per-pass CSV to check")
    a = ap.parse_args()

    import ee  # noqa: F401
    print("earth-engine:", gee.init_ee())
    asset = a.asset
    print(f"region source: {'asset ' + asset if asset else 'centroid buffer (%dm)' % CENTROID_BUFFER_M}")

    graph = json.load(open(a.graph, encoding="utf-8"))
    rows = [r for r in csv.DictReader(open(a.passes, encoding="utf-8"))
            if r["object_id"] in graph and r.get("TURBIDITY") not in (None, "")]
    random.Random(a.seed).shuffle(rows)

    print(f"\n{'object_id':<14}{'date':<12}{'TURB ship/gee':>22}{'NDCI ship/gee':>22}{'risk s/g':>10}")
    tdiff, ndiff, risk_ok = [], [], 0
    for r in rows[:a.n]:
        oid, d = r["object_id"], r["date"]
        d1 = (date.fromisoformat(d) + timedelta(days=1)).isoformat()
        geom = _region(ee, oid, graph, asset)
        means = gee.reduce_means(gee.indices_image(d, d1, geom), geom)
        if means.get("TURBIDITY") is None:
            print(f"{oid:<14}{d:<12}  (no cloud-free pixels for this date/region)")
            continue
        ts, tg = float(r["TURBIDITY"]), float(means["TURBIDITY"])
        ns, ng = float(r["NDCI"]), float(means["NDCI"])
        rs, rg = float(r["risk_score"]), indices.risk_score(means)["risk_score"]
        tdiff.append(abs(ts - tg)); ndiff.append(abs(ns - ng)); risk_ok += (round(rs) == rg)
        print(f"{oid:<14}{d:<12}{ts:10.4f}/{tg:<10.4f}{ns:10.4f}/{ng:<10.4f}{rs:6.1f}/{rg:<3d}")

    if not tdiff:
        print("\nno comparable passes fetched (all cloud-covered?), rerun with a larger --n")
        return 1
    mt, mn = statistics.median(tdiff), statistics.median(ndiff)
    print(f"\nmedian |diff|: TURBIDITY {mt:.4f}  NDCI {mn:.4f}  |  risk match {risk_ok}/{len(tdiff)}")
    ok = mt <= a.tol and mn <= a.tol
    print("RESULT: CORRESPONDS" if ok else
          "RESULT: MISMATCH (exceeds tol; pass --asset for exact reach geometry)")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
