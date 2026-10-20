"""NDCI co-fluctuation advantage by source Strahler bin (paper Strahler strata).

Same procedure as the turbidity block of exp_conditional.py, applied to NDCI.
Checks whether the effect holds on wide reaches, where mixed land/water pixels
are rare.

    python scripts/run_ndci_strata.py > results/national/ndci_strahler.txt
"""

import argparse
import csv

import _bootstrap

from experiments.eval import BIN_ORDER
from experiments.eval.temporal_transport import (
    common_mode_anomaly,
    load_series,
    temporal_transport,
)
from experiments.graph import RiverGraph


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", default=_bootstrap.data_de("segment_graph_de.json"))
    ap.add_argument("--series", default=_bootstrap.data_de("turb_series_national.csv"),
                    help="per-(segment,month) turbidity/NDCI CSV")
    args = ap.parse_args()

    graph = RiverGraph.from_json(args.graph)
    with _bootstrap.open_text(args.series) as f:
        rows = list(csv.DictReader(f))
    segs, months, X = load_series(rows, "ndci", graph)
    anom = common_mode_anomaly(X)
    print(f"NDCI series: {len(segs)} reaches x {len(months)} months")
    print("NDCI co-fluctuation (common-mode) by source Strahler bin:")
    for b in BIN_ORDER:
        res = temporal_transport(graph, segs, anom, name="ndci", deseason="cm",
                                 sample=2500, n_boot=2000, source_bins={b})
        print(f"  {b:<18} n={res.n_pairs:<4} C_flow-C_euc diff = {res.diff:+.3f} "
              f"[{res.boot_lo:+.3f},{res.boot_hi:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
