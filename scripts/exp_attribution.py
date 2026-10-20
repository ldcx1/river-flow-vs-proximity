"""Source ranking against OSM facilities (paper Sec. V-B).

For each exceedance target, ranks candidate source reaches by graph-upstream
tracing and checks whether top-ranked reaches lie near an OSM industrial or
wastewater facility. Controls: Euclidean proximity, random, a permuted field and
reversed (downstream) tracing. The plain Euclidean control may also hit
downstream facilities. See exp_attribution_upstream.py for the fair,
upstream-restricted comparison.

The default field is data/de/pollution_field_national.csv.gz, the monthly mean of
the per-pass risk score (experiments.ingest.indices.risk_score) per reach. It
reproduces the paper's Sec. IV numbers (+0.056 vs reversed, -0.041 vs proximity).
Any per-(reach, month) CSV works via --field and --value-col, for example the
turbidity series (--field data/de/turb_series_national.csv --value-col turb).

    python scripts/exp_attribution.py > results/national/ATTRIBUTION.txt   # 30 s
"""

import argparse
import csv
from collections import defaultdict

import _bootstrap

from experiments.eval import run_attribution
from experiments.graph import RiverGraph
from experiments.osm import FacilityIndex


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", default=_bootstrap.data_de("segment_graph_de.json"))
    ap.add_argument("--strong", default=_bootstrap.data_de("strong_labels_national_de.csv"),
                    help="strong_labels CSV")
    ap.add_argument("--osm", default=_bootstrap.data_de("osm_industrial_de.json"),
                    help="OSM facilities JSON")
    ap.add_argument("--field", default=_bootstrap.data_de("pollution_field_national.csv"),
                    help="per-(reach, month) field CSV with columns object_id, ym and --value-col")
    ap.add_argument("--value-col", default="c")
    ap.add_argument("--radius-m", type=float, default=1500.0)
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    graph = RiverGraph.from_json(args.graph)
    facilities = FacilityIndex.from_json(args.osm)

    targets = defaultdict(list)
    with open(args.strong, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["label"]) == 1:
                targets[r["year_month"]].append(r["object_id"])

    field = defaultdict(dict)
    with _bootstrap.open_text(args.field) as f:
        for r in csv.DictReader(f):
            try:
                field[r["ym"]][r["object_id"]] = float(r[args.value_col])
            except (ValueError, KeyError):
                continue

    print(f"graph {len(graph)} | facilities {len(facilities.facilities)} | "
          f"target-months {len(targets)} | field-months {len(field)}\n")
    res = run_attribution(graph, field, targets, facilities,
                          radius_m=args.radius_m, top_k=args.top_k)
    print(res.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
