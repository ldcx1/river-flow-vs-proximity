"""Build Romanian source-ranking inputs for ``exp_attribution.py``.

Romania has no Sentinel-era in-situ chemistry, so exceedance targets come from
other sources. The traced field is turbidity and the hits are OSM facilities,
as in Germany. Two target sets (both from the shipped RO bundle):
  wfd  reaches in a WFD water body worse than "good" ecological status, traced
       on the static per-reach mean turbidity
  sat  the top --pct percent turbidity reach-months, traced on the monthly series

    python scripts/prep_ro_attribution.py --variant wfd --out build/ro_attr
    python scripts/exp_attribution.py \\
        --graph data/ro/segment_graph_ro.json \\
        --strong build/ro_attr/targets.csv \\
        --osm   data/ro/osm_industrial_ro.json \\
        --field build/ro_attr/field.csv > results/national/ro_attribution_wfd.txt

Use --variant sat (and ro_attribution_sat.txt) for the satellite target set.
"""
import argparse
import csv
import os

import _bootstrap  # noqa: F401


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["wfd", "sat"], default="wfd")
    ap.add_argument("--wfd", default=_bootstrap.data_ro("segment_labels_ro_national.csv"))
    ap.add_argument("--field-mean", default=_bootstrap.data_ro("turbidity_field_ro.csv"))
    ap.add_argument("--series", default=_bootstrap.data_ro("turb_series_ro.csv"))
    ap.add_argument("--pct", type=float, default=99.0,
                    help="sat variant: turbidity percentile above which a reach-month is a target")
    ap.add_argument("--out", required=True, help="output directory for targets.csv + field.csv")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tgt_path = os.path.join(args.out, "targets.csv")
    fld_path = os.path.join(args.out, "field.csv")

    if args.variant == "wfd":
        field = list(csv.DictReader(open(args.field_mean, encoding="utf-8")))
        with open(fld_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["ym", "object_id", "c"])
            for r in field:
                w.writerow(["static", r["object_id"], r["turb"]])
        n = 0
        with open(tgt_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["year_month", "object_id", "label"])
            for r in csv.DictReader(open(args.wfd, encoding="utf-8")):
                if r["label"] == "1":
                    w.writerow(["static", r["object_id"], 1]); n += 1
        print(f"wfd: {n} WFD-degraded targets | field {len(field)} reaches (static mean turbidity)")
    else:
        import numpy as np
        rows = list(csv.DictReader(_bootstrap.open_text(args.series)))
        turb = np.array([float(r["turb"]) for r in rows])
        thr = float(np.percentile(turb, args.pct))
        with open(fld_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["ym", "object_id", "c"])
            for r in rows:
                w.writerow([r["ym"], r["object_id"], r["turb"]])
        n = 0
        with open(tgt_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["year_month", "object_id", "label"])
            for r, t in zip(rows, turb):
                if t >= thr:
                    w.writerow([r["ym"], r["object_id"], 1]); n += 1
        print(f"sat: {n} anomaly targets (turb>=p{args.pct:g}={thr:.0f}) | field {len(rows)} reach-months")
    print(f"wrote {tgt_path}\nwrote {fld_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
