"""Network autocorrelation vs geographic proximity (Moran's I, Germany).

For per-segment mean dissolved oxygen, nitrate and the binary "ever exceeded"
flag from in-situ readings, compares Moran's I on flow connectivity with a
degree-matched Euclidean kNN adjacency. Permutation p-values and a rewired-graph
placebo are reported.

    python scripts/exp_autocorr.py > results/national/AUTOCORRELATION.txt
"""

import argparse
import csv

import _bootstrap  # noqa: F401

from experiments.data import DEFAULT_RULES, is_exceedance
from experiments.eval import morans_autocorr, segment_means
from experiments.graph import RiverGraph


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", default=_bootstrap.data_de("segment_graph_de.json"))
    ap.add_argument("--readings", default=_bootstrap.data_de("insitu_readings_national_de.csv"),
                    help="insitu readings CSV")
    ap.add_argument("--max-dist-km", type=float, default=50.0)
    ap.add_argument("--perm", type=int, default=2000)
    args = ap.parse_args()

    graph = RiverGraph.from_json(args.graph)
    with open(args.readings, encoding="utf-8") as f:
        readings = list(csv.DictReader(f))
    if not readings:
        raise SystemExit(f"empty input readings: {args.readings}")

    scalars = {
        "Dissolved oxygen": segment_means(readings, "Dissolved oxygen", graph),
        "Nitrate": segment_means(readings, "Nitrate", graph),
    }
    # binary: did the segment ever exceed any rule
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
    scalars["exceedance flag"] = {s: float(v) for s, v in ever.items()}

    print(f"graph: {len(graph)} segments | max flow-dist {args.max_dist_km:.0f} km | "
          f"{args.perm} permutations\n")
    print("scalar           n      I_flow (p)        I_euc      diff (p)        I_rewired")
    print("-" * 84)
    results = []
    for name, vals in scalars.items():
        res = morans_autocorr(graph, vals, name=name, max_dist_km=args.max_dist_km,
                              n_perm=args.perm)
        print(res.line())
        results.append(res)
    sig = [r for r in results if r.I_diff > 0 and r.p_diff < 0.05]
    if sig:
        print("\nI_flow significantly > I_euc (diff p<0.05) => flow connectivity "
              "organises the data beyond geography for: "
              + ", ".join(r.name for r in sig) + ".")
    else:
        ps = "/".join(f"{r.p_diff:.3f}" for r in results)
        ns = [r.n_segments for r in results]
        print(f"\nNo determinand shows I_flow significantly > I_euc (every diff p>=0.05: "
              f"{ps}) => no evidence the flow network organises these means beyond "
              f"geography (n={min(ns)}-{max(ns)}, underpowered).")
    print("I_rewired near 0 (vs I_flow) would indicate the specific wiring matters, "
          "not just degree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
