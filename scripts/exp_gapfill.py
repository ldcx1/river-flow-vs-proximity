"""Gap-filling at ungauged reaches (Germany).

Predicts each monitored segment's mean dissolved oxygen, nitrate and binary
exceedance flag from the other segments. Compares network propagation with
Euclidean interpolation, the global mean and a label-permuted placebo.

    python scripts/exp_gapfill.py > results/national/GAPFILL.txt
"""

import argparse
import csv

import _bootstrap  # noqa: F401

from experiments.data import DEFAULT_RULES, is_exceedance
from experiments.eval import gapfill_eval, segment_means
from experiments.graph import RiverGraph


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", default=_bootstrap.data_de("segment_graph_de.json"))
    ap.add_argument("--readings", default=_bootstrap.data_de("insitu_readings_national_de.csv"))
    ap.add_argument("--max-dist-km", type=float, default=50.0)
    ap.add_argument("--decay-km", type=float, default=30.0)
    args = ap.parse_args()

    graph = RiverGraph.from_json(args.graph)
    with open(args.readings, encoding="utf-8") as f:
        readings = list(csv.DictReader(f))
    if not readings:
        raise SystemExit(f"empty input readings: {args.readings}")

    do = segment_means(readings, "Dissolved oxygen", graph)
    no3 = segment_means(readings, "Nitrate", graph)
    ever = {}
    for r in readings:
        rule = DEFAULT_RULES.get(r["determinand"])
        if rule is None or r["segment"] not in graph or r["uom"] != rule.uom:
            continue
        try:
            v = float(r["value"])
        except (ValueError, TypeError):
            continue
        ever[r["segment"]] = max(ever.get(r["segment"], 0), int(is_exceedance(rule, v)))
    ever = {s: float(v) for s, v in ever.items()}

    for name, vals, binary in [("Dissolved oxygen", do, False),
                               ("Nitrate", no3, False),
                               ("exceedance flag", ever, True)]:
        res = gapfill_eval(graph, vals, name=name, max_dist_km=args.max_dist_km,
                           decay_km=args.decay_km, binary=binary)
        print(res.summary())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
