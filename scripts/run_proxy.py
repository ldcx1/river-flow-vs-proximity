"""Proxy validity: satellite optical signal vs in-situ chemistry (Germany).

Spearman correlation per (satellite index, in-situ determinand) pair, matched on
(segment, month), with a 95% CI from a segment cluster bootstrap. Each pair is
reported separately. The paper cites turbidity vs ammonium. Optically visible
determinands (turbidity, suspended solids, chlorophyll) are sparse in the
national in-situ data. Needs the ``data`` extra.

    python scripts/run_proxy.py > results/national/proxy.txt
"""
import argparse
import csv
from collections import defaultdict

import numpy as np
import pandas as pd

import _bootstrap
from experiments.metrics import spearman_r  # noqa: E402



def ym(d):                       # YYYYMMDD -> YYYY-MM
    return d[:4] + "-" + d[4:6]


def cluster_boot_spearman(seg, a, b, n_boot=2000, seed=0):
    seg = np.asarray(seg)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    point = spearman_r(a, b)
    uniq = np.array(sorted(set(seg.tolist())))
    rows_by = {g: np.where(seg == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        ri = np.concatenate([rows_by[g] for g in samp])
        r = spearman_r(a[ri], b[ri])
        if np.isfinite(r):
            vals.append(r)
    lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan))
    return float(point), float(lo), float(hi), len(a), len(uniq)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perdate", default=_bootstrap.data_de("turb_perdate_national.csv"),
                    help="per-pass turbidity/NDCI CSV (object_id, acquired_at, turb, ndci)")
    ap.add_argument("--insitu", default=_bootstrap.data_de("insitu_readings_national_de.csv"),
                    help="in-situ readings CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.perdate, parse_dates=["acquired_at"])
    df["ym"] = df["acquired_at"].dt.strftime("%Y-%m")
    sat = df.groupby(["object_id", "ym"]).agg(turb=("turb", "mean"),
                                              ndci=("ndci", "mean")).reset_index()
    satmap = {(r.object_id, r.ym): (r.turb, r.ndci) for r in sat.itertuples()}

    insitu = defaultdict(list)
    for r in csv.DictReader(_bootstrap.open_text(args.insitu)):
        try:
            insitu[(r["determinand"], r["segment"], ym(r["date"]))].append(float(r["value"]))
        except (ValueError, KeyError):
            pass
    det_month = {}
    for (det, seg, m), vs in insitu.items():
        det_month.setdefault(det, {})[(seg, m)] = float(np.mean(vs))

    pairs = [("Chlorophyll a", "ndci"), ("Total suspended solids", "turb"),
             ("Ammonium", "turb"), ("BOD5", "turb"),
             ("Nitrate", "turb"), ("Dissolved oxygen", "turb")]
    sidx = {"turb": 0, "ndci": 1}
    lines = ["NATIONAL proxy correspondence: satellite index vs in-situ determinand,",
             "matched on (segment, month); tie-aware Spearman; segment cluster-bootstrap",
             "95% CI (2000 resamples of whole segments). Strata separate.\n",
             f"  {'in-situ determinand':<24} {'sat':<5} {'Spearman r [95% CI]':<26} {'N pairs':>8} {'segs':>6}"]
    for det, ix in pairs:
        dm = det_month.get(det, {})
        seg, a, b = [], [], []
        for (s, m), v in dm.items():
            if (s, m) in satmap:
                seg.append(s)
                a.append(v)
                b.append(satmap[(s, m)][sidx[ix]])
        if len(a) >= 12:
            pt, lo, hi, n, ns = cluster_boot_spearman(seg, a, b)
            lines.append(f"  {det:<24} {ix:<5} {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]      {n:>8} {ns:>6}")
        else:
            lines.append(f"  {det:<24} {ix:<5} N={len(a)} (too few to estimate)")
    lines.append("\n  Note: turbidity/suspended-solids/chlorophyll are sparse nationally; the")
    lines.append("  abundant in-situ (nutrients/oxygen) are not what the satellite optically sees.")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
