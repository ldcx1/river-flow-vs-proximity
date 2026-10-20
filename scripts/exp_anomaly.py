"""Synthetic anomaly suite (paper Table III, label-free).

Injects synthetic anomalies into the monthly turbidity field and asks whether
flow-graph neighbours detect or trace them better than Euclidean neighbours.
Choose sub-experiments with --exp, one name or a comma list (default: all five):
  residual  graph-expectation residual (local injection vs inherited high value)
  break     neighbour-break test (a reach that breaks from its flow neighbours)
  recovery  connected event recovery (flow components vs geographic clusters)
  denoise   false-alarm filter (flow vs geographic corroboration)
  tracing   spatio-temporal source tracing

Main logs:

    python scripts/exp_anomaly.py --exp all > results/national/anomaly.txt
    python scripts/exp_anomaly.py --exp tracing > results/national/tracing_reversed.txt
    python scripts/exp_anomaly.py --exp residual,break --euc-decay-km 30 \\
        > results/national/break_euc_decay.txt
    python scripts/exp_anomaly.py --graph data/ro/segment_graph_ro.json \\
        --series data/ro/turb_series_ro.csv --exp all > results/national/ro_anomaly.txt

Decay robustness (results/national/robustness.txt). --brief prints one line per
break or tracing run. The log is these four runs concatenated:

    (python scripts/exp_anomaly.py --exp break --brief
     python scripts/exp_anomaly.py --exp break --brief --inject-decay-km 15
     python scripts/exp_anomaly.py --exp tracing --brief
     python scripts/exp_anomaly.py --exp tracing --brief --hop-decay 0.6
    ) > results/national/robustness.txt
"""

import argparse
import csv

import _bootstrap  # noqa: F401

from experiments.eval import (load_series, run_residual, run_break, run_recovery,
                              run_denoise, run_tracing)
from experiments.graph import RiverGraph

ALL_EXP = ["residual", "break", "recovery", "denoise", "tracing"]
BREAK_DECAY_KM = 30.0      # run_break's score decay, also its default plume decay
TRACING_HOP_DECAY = 0.3    # run_tracing's default per-hop plume decay


def _exp_list(text):
    names = ALL_EXP if text == "all" else [t.strip() for t in text.split(",") if t.strip()]
    bad = [n for n in names if n not in ALL_EXP]
    if bad or not names:
        raise argparse.ArgumentTypeError(
            f"unknown experiment(s) {bad}; choose from {ALL_EXP + ['all']}")
    return set(names)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", default=_bootstrap.data_de("segment_graph_de.json"))
    ap.add_argument("--series", default=_bootstrap.data_de("turb_series_national.csv"),
                    help="per-(segment,month) field CSV")
    ap.add_argument("--value-col", default="turb")
    ap.add_argument("--exp", default="all", type=_exp_list,
                    help="experiment name, comma list, or 'all' (%s)" % ", ".join(ALL_EXP))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--euc-decay-km", type=float, default=None,
                    help="break: decay-weight the Euclidean control like the flow arm "
                         "(default: unweighted, as published)")
    ap.add_argument("--inject-decay-km", type=float, default=None,
                    help="break: decay of the injected plume, separate from the score's "
                         f"decay (default: the score's {BREAK_DECAY_KM:g} km)")
    ap.add_argument("--hop-decay", type=float, default=TRACING_HOP_DECAY,
                    help="tracing: per-hop decay of the injected plume")
    ap.add_argument("--brief", action="store_true",
                    help="break and tracing: print one robustness line instead of the "
                         "full block, and no header")
    args = ap.parse_args()

    graph = RiverGraph.from_json(args.graph)
    with _bootstrap.open_text(args.series) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty input series: {args.series}")
    segs, months, X = load_series(rows, args.value_col, graph)
    if not args.brief:
        print(f"graph {len(graph)} | field {args.value_col}: "
              f"{len(segs)} segs x {len(months)} months\n")

    want = args.exp

    if "residual" in want:
        res1, c1 = run_residual(graph, segs, X, seed=args.seed)
        print(res1.summary())
        print("  focused contrast (LOCAL injection vs INHERITED-high), ROC-AUC:")
        for m, v in c1.items():
            print(f"    {m:<22}{v:.3f}")
        print()
    if "break" in want:
        res2, c2 = run_break(graph, segs, X, seed=args.seed, decay_km=BREAK_DECAY_KM,
                             inject_decay_km=args.inject_decay_km,
                             euc_decay_km=args.euc_decay_km)
        if args.brief:
            inj = args.inject_decay_km if args.inject_decay_km is not None else BREAK_DECAY_KM
            d = res2.boot_delta["spatial_euclidean"]
            print(f"break inject_decay_km={inj:g}: graph-euc {d['mean']:+.4f} "
                  f"[{d['lo']:+.4f},{d['hi']:+.4f}]")
        else:
            print(res2.summary())
            print("  focused contrast (VIOLATION vs PROPAGATED plume), ROC-AUC:")
            for m, v in c2.items():
                print(f"    {m:<22}{v:.3f}")
            print()
    if "recovery" in want:
        print(run_recovery(graph, segs, X, seed=args.seed).summary())
        print()
    if "denoise" in want:
        print(run_denoise(graph, segs, X, seed=args.seed).summary())
        print()
    if "tracing" in want:
        res5 = run_tracing(graph, segs, X, seed=args.seed, inject_hop_decay=args.hop_decay)
        if args.brief:
            d = res5.top1_delta
            print(f"tracing hop_decay={args.hop_decay:g}: ST-static {d['mean']:+.3f} "
                  f"[{d['lo']:+.3f},{d['hi']:+.3f}]; euc MRR {res5.mrr['euclidean']:.3f}")
        else:
            print(res5.summary())
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
