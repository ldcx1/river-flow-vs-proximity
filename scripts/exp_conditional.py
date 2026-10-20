"""Conditional and Strahler-stratified analyses (paper Strahler strata, Germany).

Three blocks:
  1. Partial Moran's I (after covariates) and stream vs Euclidean variograms for
     per-segment dissolved oxygen, nitrate and turbidity.
  2. Turbidity co-fluctuation advantage per Strahler bin, with a monotonic trend.
     run_ndci_strata.py does the same for NDCI.
  3. Neighbour-break test: incremental AUC of the graph score over node and
     Euclidean scores, and the graph-vs-Euclidean AUC delta per Strahler bin.

    python scripts/exp_conditional.py > results/national/CONDITIONAL.txt   # ~30 min
"""

import argparse
import csv

import numpy as np
import _bootstrap  # noqa: F401

from experiments.eval import (
    BIN_ORDER,
    common_mode_anomaly,
    incremental_auc,
    load_series,
    monotonic_trend,
    partial_morans,
    run_break,
    segment_means,
    strahler_bin,
    temporal_transport,
    variograms,
)
from experiments.graph import RiverGraph
from experiments.metrics import roc_auc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", default=_bootstrap.data_de("segment_graph_de.json"))
    ap.add_argument("--readings", default=_bootstrap.data_de("insitu_readings_national_de.csv"))
    ap.add_argument("--turb-means", default=_bootstrap.data_de("turbidity_field_national.csv"),
                    help="per-segment turb/ndci CSV")
    ap.add_argument("--turb-series", default=_bootstrap.data_de("turb_series_national.csv"),
                    help="per-(segment,month) turb CSV")
    ap.add_argument("--ag", default=_bootstrap.data_de("ag_fraction_de_full.csv"),
                    help="ag_fraction CSV (object_id, ag_frac)")
    args = ap.parse_args()

    graph = RiverGraph.from_json(args.graph)
    with open(args.readings, encoding="utf-8") as f:
        readings = list(csv.DictReader(f))
    if not readings:
        raise SystemExit(f"empty input readings: {args.readings}")
    ag = None
    if args.ag:
        ag = {}
        with open(args.ag, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    ag[r["object_id"]] = float(r["ag_frac"])
                except (ValueError, KeyError):
                    pass
    turb_mean = {}
    with open(args.turb_means, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                turb_mean[r["object_id"]] = float(r["turb"])
            except (ValueError, KeyError):
                pass

    print("=" * 72 + "\nCC2-Autocorrelation: partial Moran (beyond covariates) + stream variogram\n" + "=" * 72)
    scalars = {"Dissolved oxygen": segment_means(readings, "Dissolved oxygen", graph),
               "Nitrate": segment_means(readings, "Nitrate", graph),
               "TURBIDITY": turb_mean}
    for name, vals in scalars.items():
        pm = partial_morans(graph, vals, None, name=name, ag=ag)
        print("  " + pm.line())
        h, gf, ge = variograms(graph, vals)
        # short-range semivariance ratio, stream / Euclidean (lower = more similar)
        sr = np.nanmean(gf[:2]) / np.nanmean(ge[:2]) if np.nanmean(ge[:2]) else float("nan")
        print(f"    variogram short-range (<= {h[1]:.0f} km) stream/euclidean ratio = {sr:.3f} "
              f"({'stream more similar' if sr < 1 else 'no stream advantage'})")

    print("\n" + "=" * 72 + "\nConditional co-fluctuation: temporal coherence by Strahler bin (monotonic trend)\n" + "=" * 72)
    with _bootstrap.open_text(args.turb_series) as f:
        rows = list(csv.DictReader(f))
    segs, months, X = load_series(rows, "turb", graph)
    anom = common_mode_anomaly(X)
    diffs = []
    for b in BIN_ORDER:
        res = temporal_transport(graph, segs, anom, name="turb", deseason="cm",
                                 sample=2500, n_boot=2000, source_bins={b})
        diffs.append(res.diff)
        print(f"  {b:<18} n={res.n_pairs:<4} C_flow-C_euc diff = {res.diff:+.3f} "
              f"[{res.boot_lo:+.3f},{res.boot_hi:+.3f}]")
    tr = monotonic_trend(BIN_ORDER, diffs)
    print(f"  monotonic trend (advantage vs Strahler): rho={tr['rho']:+.3f} slope={tr['slope']:+.3f}")

    print("\n" + "=" * 72 + "\nConditional value: incremental AUC + Strahler-stratified anomaly\n" + "=" * 72)
    res2, contrast, raw = run_break(graph, segs, X, return_scores=True)
    y, sc, strh = raw["y"], raw["scores"], raw["strahler"]
    inc = incremental_auc(y, [sc["node_temporal"], sc["spatial_euclidean"]],
                          sc["graph_neighbors"], raw["groups"], n_boot=2000)
    print(f"  Conditional incremental AUC (graph beyond node+euclidean): base {inc['auc_base']:.3f} "
          f"-> full {inc['auc_full']:.3f}  gain {inc['gain']:+.3f} [{inc['lo']:+.3f},{inc['hi']:+.3f}]")
    bins = np.array([strahler_bin(s) for s in strh])
    sdiffs = []
    for b in BIN_ORDER:
        m = bins == b
        if m.sum() < 20 or y[m].sum() in (0, m.sum()):
            sdiffs.append(np.nan); print(f"  {b:<18} (too few)"); continue
        d = roc_auc(y[m], sc["graph_neighbors"][m]) - roc_auc(y[m], sc["spatial_euclidean"][m])
        sdiffs.append(d)
        print(f"  {b:<18} n={int(m.sum()):<5} graph-euclidean AUC delta = {d:+.3f}")
    tr2 = monotonic_trend(BIN_ORDER, sdiffs)
    print(f"  monotonic trend: rho={tr2['rho']:+.3f} slope={tr2['slope']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
