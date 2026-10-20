"""Static propagation test (paper Table III, label-free).

Does the per-segment mean turbidity / NDCI field correlate more along
flow-downstream pairs than along Euclidean-nearest or random pairs?

    python scripts/exp_propagation.py > results/national/PROPAGATION.txt     # Germany
    python scripts/exp_propagation.py --graph data/ro/segment_graph_ro.json \\
        --field data/ro/turbidity_field_ro.csv > results/national/ro_propagation.txt
"""

import argparse
import csv

import _bootstrap  # noqa: F401

from experiments.eval import propagation_test
from experiments.graph import RiverGraph


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", default=_bootstrap.data_de("segment_graph_de.json"))
    ap.add_argument("--field", default=_bootstrap.data_de("turbidity_field_national.csv"),
                    help="per-segment turbidity/NDCI CSV")
    ap.add_argument("--max-dist-km", type=float, default=50.0)
    ap.add_argument("--sample", type=int, default=3000)
    args = ap.parse_args()

    graph = RiverGraph.from_json(args.graph)
    turb, ndci = {}, {}
    with open(args.field, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for col, d in (("turb", turb), ("ndci", ndci)):
                try:
                    d[r["object_id"]] = float(r[col])
                except (ValueError, KeyError):
                    pass

    for name, vals in [("TURBIDITY", turb), ("NDCI", ndci)]:
        if not vals:
            continue
        res = propagation_test(graph, vals, name=name, max_dist_km=args.max_dist_km,
                               sample=args.sample)
        print(res.summary())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
