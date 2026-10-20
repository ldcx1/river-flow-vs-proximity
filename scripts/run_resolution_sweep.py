"""Temporal-resolution sweep (paper Sec. V-C, Fig. 5).

Re-runs the temporally resolved experiments on the per-pass Sentinel-2 series
binned at W in {1, 3, 7, 15, 28} days. Records each effect with its CI and the
coverage (usable segments, pairs, NaN rate) per W. Time-collapsed experiments
(autocorrelation, gap-filling, propagation) do not depend on W and are skipped.
The results go to the JSON file given by --out. Progress goes to stderr.

    python scripts/run_resolution_sweep.py --out results/national/resolution_sweep.json
    python scripts/run_resolution_sweep.py --widths 28,7 --out sweep.json

The per-pass series ships as split gzip parts
(data/de/turb_perdate_national_full.csv.gz.part*). They are read in order as
one stream, so no reassembly is needed.
"""
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

import _bootstrap
from experiments.eval import (common_mode_anomaly, incremental_auc, run_residual,  # noqa: E402
                              run_break, run_recovery, run_denoise, run_tracing,
                              temporal_transport)
from experiments.graph import RiverGraph  # noqa: E402

N_MIN = 12


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", default=_bootstrap.data_de("segment_graph_de.json"))
    ap.add_argument("--perdate", default=_bootstrap.data_de("turb_perdate_national_full.csv"),
                    help="per-pass CSV (object_id, acquired_at, turb, ndci); "
                         "plain, .gz or split .gz.part* files")
    ap.add_argument("--widths", default="1,3,7,15,28",
                    help="comma-separated bin widths in days")
    ap.add_argument("--out", required=True, help="output JSON path")
    args = ap.parse_args()
    WS = [int(x) for x in args.widths.split(",") if x.strip()]

    t0 = time.time()
    log("loading graph + per-date series ...")
    g = RiverGraph.from_json(args.graph)
    segset = set(g.nodes)
    with _bootstrap.open_text(args.perdate) as fh:
        df = pd.read_csv(fh, parse_dates=["acquired_at"])
    df = df[df["object_id"].isin(segset)]
    segs = sorted(df["object_id"].unique())
    sidx = {s: i for i, s in enumerate(segs)}
    si = df["object_id"].map(sidx).to_numpy(np.int64)
    day = (df["acquired_at"] - df["acquired_at"].min()).dt.days.to_numpy(np.int64)
    turb = df["turb"].to_numpy(float)
    ndci = df["ndci"].to_numpy(float)
    nseg, maxday = len(segs), int(day.max())
    log(f"{len(df):,} obs | {nseg:,} segs | span {maxday+1} days | load {time.time()-t0:.0f}s")

    def binned(values, W):
        nb = maxday // W + 1
        flat = si * nb + (day // W)
        size = nseg * nb
        s = np.bincount(flat, weights=values, minlength=size)
        c = np.bincount(flat, minlength=size).astype(float)
        with np.errstate(invalid="ignore"):
            m = s / c
        m[c == 0] = np.nan
        return m.reshape(nseg, nb)

    def cid(d):
        return {k: round(float(v), 4) for k, v in d.items()}

    results = {}
    for W in WS:
        tw = time.time()
        Xt, Xn = binned(turb, W), binned(ndci, W)
        nb = Xt.shape[1]
        cov = np.isfinite(Xt).sum(1)
        res = {"coverage": {"W_days": W, "n_bins": nb,
                            "usable_segs_min%d" % N_MIN: int((cov >= N_MIN).sum()),
                            "mean_nonan_per_seg": round(float(cov.mean()), 1),
                            "nan_rate": round(float(np.isnan(Xt).mean()), 3)}}
        maxlag = max(1, round(14 / W))
        for sig, X in [("turb", Xt), ("ndci", Xn)]:
            anom = common_mode_anomaly(X)
            r = temporal_transport(g, segs, anom, name=sig, deseason="cm", n_min=N_MIN,
                                   max_dist_km=60.0, long_km=30.0, max_lag=maxlag,
                                   sample=2000, n_boot=1000, n_perm=0)
            res["cofluctuation_" + sig] = {
                "diff": round(r.diff, 4), "lo": round(r.boot_lo, 4),
                "hi": round(r.boot_hi, 4), "n_pairs": r.n_pairs,
                "c_flow": round(r.c_flow, 4), "c_euc": round(r.c_euc, 4),
                "H2_asym": round(float(r.asym_long), 4), "H2_n_long": r.n_long,
                "H2_maxlag_bins": maxlag}
        a1, _ = run_residual(g, segs, Xt, seed=0)
        res["residual"] = {"vs_euc": cid(a1.boot_delta["spatial_euclidean"]),
                           "vs_node": cid(a1.boot_delta["node_temporal"]),
                           "frac_dropped": round(a1.frac_dropped, 3)}
        a2, _, raw = run_break(g, segs, Xt, return_scores=True, seed=0)
        res["break"] = {"vs_euc": cid(a2.boot_delta["spatial_euclidean"]),
                        "vs_node": cid(a2.boot_delta["node_temporal"])}
        inc = incremental_auc(raw["y"], [raw["scores"]["node_temporal"], raw["scores"]["spatial_euclidean"]],
                              raw["scores"]["graph_neighbors"], raw["groups"], n_boot=300)
        res["conditional_incremental"] = {k: round(float(inc[k]), 4)
                                          for k in ("gain", "lo", "hi", "auc_base", "auc_full")}
        a3 = run_recovery(g, segs, Xt, seed=0); res["recovery"] = {"delta": cid(a3.f1_delta), "n_runs": a3.n_runs}
        a4 = run_denoise(g, segs, Xt, seed=0); res["denoise"] = {"delta": cid(a4.prec_delta), "n_runs": a4.n_runs}
        a5 = run_tracing(g, segs, Xt, n_events=120, seed=0)
        res["tracing"] = {"st_static": cid(a5.top1_delta), "mrr_euc": round(a5.mrr["euclidean"], 3),
                          "mrr_static": round(a5.mrr["static"], 3), "n_events": a5.n_events}
        results[str(W)] = res
        log(f"W={W:>2}d done in {time.time()-tw:.0f}s | {res['coverage']}")
        del Xt, Xn

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=0)
    log(f"\nwrote {args.out} | total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
