"""Co-fluctuation test (paper Table III and Fig. 3, label-free).

Do monthly turbidity / NDCI anomalies of flow-connected reach pairs move
together more than distance-matched, non-connected Euclidean pairs (printed as
H1)? Two deseasonings: network common-mode and per-segment seasonal. Also
reports a lag asymmetry on long pairs (printed as H2). Cluster bootstrap by
source reach plus a permutation null. --orient-match adds the bearing-matched
control.

    python scripts/exp_temporal.py > results/national/COFLUCTUATION.txt   # ~15 min
    python scripts/exp_temporal.py --orient-match \\
        > results/national/COFLUCTUATION_orient.txt
    python scripts/exp_temporal.py --graph data/ro/segment_graph_ro.json \\
        --series data/ro/turb_series_ro.csv > results/national/ro_cofluctuation.txt

Add --orient-match to the Romanian command for ro_cofluctuation_orient.txt.
Progress lines go to stderr.
"""

import argparse
import csv
import sys
import time

import _bootstrap  # noqa: F401

from experiments.eval import (
    common_mode_anomaly,
    load_series,
    seasonal_anomaly,
    temporal_transport,
)
from experiments.graph import RiverGraph


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", default=_bootstrap.data_de("segment_graph_de.json"))
    ap.add_argument("--series", default=_bootstrap.data_de("turb_series_national.csv"),
                    help="per-(segment,month) turbidity/NDCI CSV")
    ap.add_argument("--max-dist-km", type=float, default=60.0)
    ap.add_argument("--n-min", type=int, default=12)
    ap.add_argument("--sample", type=int, default=3000)
    ap.add_argument("--perm", type=int, default=2000,
                    help="permutation-null count (>200 to avoid a coarse p-floor)")
    ap.add_argument("--long-km", type=float, default=40.0,
                    help="min flow distance for the long-pair lag test (<= max-dist-km)")
    ap.add_argument("--orient-match", action="store_true",
                    help="bearing-match each Euclidean control pair to its flow pair")
    ap.add_argument("--orient-tol-deg", type=float, default=45.0)
    args = ap.parse_args()

    t0 = time.time()

    def progress(msg):
        # Progress goes to stderr so a redirected log holds only the results.
        print(f"[{time.time() - t0:6.0f}s] {msg}", file=sys.stderr, flush=True)

    progress("loading graph and series")
    graph = RiverGraph.from_json(args.graph)
    with _bootstrap.open_text(args.series) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty input series: {args.series}")

    print(f"graph {len(graph)} | rows {len(rows)} | sample {args.sample} | "
          f"max flow-dist {args.max_dist_km:.0f} km | min obs {args.n_min} | "
          f"orient_match {args.orient_match}\n")
    print("signal    deseason       npairs  C_flow  C_euc   diff[CI]                p    verdict")
    print("-" * 100)
    for col in ["turb", "ndci"]:
        segs, months, X = load_series(rows, col, graph)
        for ds_name, anom in [("common-mode", common_mode_anomaly(X)),
                              ("seasonal", seasonal_anomaly(X, months))]:
            progress(f"{col} {ds_name}: pairs, bootstrap and permutation null")
            res = temporal_transport(graph, segs, anom, name=col, deseason=ds_name,
                                     n_min=args.n_min, max_dist_km=args.max_dist_km,
                                     sample=args.sample, n_perm=args.perm,
                                     long_km=args.long_km, orient_match=args.orient_match,
                                     orient_tol_deg=args.orient_tol_deg)
            print(res.line())
    progress("done")
    print("\nH1: diff>0 with CI excluding 0 => flow co-fluctuation beats geography.")
    print("H2 asym(long): >0 => downstream lags upstream (transport); ~0 likely "
          "unresolvable at monthly cadence (expected).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
