"""Basin-accretion scaling (paper Table III, scaling row).

Re-runs the co-fluctuation test (flow minus Euclidean coherence) on the
cumulative k largest outlet basins, to show how adding smaller basins moves the
effect. --random-ref adds random basin orders as a "just more data" reference.

Paper runs (Table III scaling row), with a reduced source sample to keep the
national run tractable:

    python scripts/exp_basins.py --random-ref 3 > results/national/BASINS.txt   # ~40 min
    python scripts/exp_basins.py --graph data/ro/segment_graph_ro.json \\
        --series data/ro/turb_series_ro.csv --k-grid 1,2,3,5,10,20,50,100 \\
        --sample 1500 --n-boot 300 > results/national/ro_basins.txt          # ~5 min

At these sample sizes (~800 pairs per point) the curve's wiggles are mostly
sampling noise. A more precise run uses the headline co-fluctuation settings,
so its full-set point equals Table III's turbidity cell (BASINS_headline.txt,
ro_basins_headline.txt):

    python scripts/exp_basins.py --sample 3000 --n-boot 2000                 # ~30 min
"""

import argparse
import csv

import numpy as np
import _bootstrap  # noqa: F401

from experiments.eval import (
    accretion_curve,
    basins,
    common_mode_anomaly,
    load_series,
    summarize,
    temporal_transport,
)
from experiments.eval.basins import AccretionPoint
from experiments.graph import RiverGraph


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", default=_bootstrap.data_de("segment_graph_de.json"))
    ap.add_argument("--series", default=_bootstrap.data_de("turb_series_national.csv"),
                    help="per-(segment,month) field CSV")
    ap.add_argument("--value-col", default="turb")
    ap.add_argument("--min-segs", type=int, default=10,
                    help="skip cumulative sets smaller than this (CI undefined)")
    ap.add_argument("--random-ref", type=int, default=0,
                    help="N random basin orderings as the more-data reference (0=skip)")
    ap.add_argument("--k-grid", default="1,2,3,5,10,20,50,100,200,400",
                    help="comma-separated cumulative k values; the full set is always added "
                         "last. Each k costs one full co-fluctuation test. 'all' = full set.")
    ap.add_argument("--sample", type=int, default=1200, help="sampled source reaches per point")
    ap.add_argument("--n-boot", type=int, default=200, help="bootstrap resamples per point")
    args = ap.parse_args()

    graph = RiverGraph.from_json(args.graph)
    with _bootstrap.open_text(args.series) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty input series: {args.series}")
    segs, months, X = load_series(rows, args.value_col, graph)
    anom = common_mode_anomaly(X)
    pos = {s: i for i, s in enumerate(segs)}

    def metric_fn(included):
        idx = [pos[s] for s in segs if s in included]
        if len(idx) < args.min_segs:
            return float("nan"), float("nan"), float("nan")
        sub = [segs[i] for i in idx]
        res = temporal_transport(graph, sub, anom[idx], name=args.value_col,
                                 deseason="cm", sample=args.sample, n_boot=args.n_boot)
        return res.diff, res.boot_lo, res.boot_hi

    bl = basins(graph, restrict=set(segs))
    print(f"{len(segs)} monitored segments in {len(bl)} outlet-basins\n")

    def grid_accretion(ordered):
        """Accrete basins cumulatively, evaluating only at --k-grid (empty = every k)."""
        if not args.k_grid:
            return accretion_curve(ordered, metric_fn)
        want = {len(ordered) if t.strip() == "all" else int(t) for t in args.k_grid.split(",")}
        want.add(len(ordered))   # always evaluate the full set
        included, curve = set(), []
        for k, (_, members) in enumerate(ordered, 1):
            included |= set(members)
            if k in want:
                v, lo, hi = metric_fn(included)
                curve.append(AccretionPoint(k=k, n_segments=len(included), value=v, lo=lo, hi=hi))
        return curve

    print("size-ordered accretion (largest basin first; metric = flow - euclidean coherence diff):")
    print(summarize(grid_accretion(bl)))

    if args.random_ref > 0:
        rng = np.random.default_rng(0)
        finals = []
        for _ in range(args.random_ref):
            order = [bl[i] for i in rng.permutation(len(bl))]
            c = grid_accretion(order)
            if c and np.isfinite(c[-1].value):
                finals.append(c[-1].value)
        if finals:
            print(f"\nrandom-order reference (n={args.random_ref}): final-k diff "
                  f"mean {np.nanmean(finals):+.3f} (size-ordered final-k should match - "
                  f"same full set; the curve SHAPE is what differs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
