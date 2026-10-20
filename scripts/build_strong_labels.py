"""Build per-(segment, month) strong labels from dated in-situ readings.

    python scripts/build_strong_labels.py \\
        --readings data/de/insitu_readings_national_de.csv --sites data/de/national_sites_all_snapped.csv \\
        --out strong_labels.csv
"""

import argparse
import csv

import _bootstrap  # noqa: F401

from experiments.data import DEFAULT_RULES, build_strong_labels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--readings", required=True, help="dated in-situ readings CSV")
    ap.add_argument("--sites", required=True, help="snapped sites CSV (segment, basin)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--year-from", default="2020")
    ap.add_argument("--year-to", default="2023")
    args = ap.parse_args()

    with open(args.sites, encoding="utf-8") as f:
        site_basin = {}
        for r in csv.DictReader(f):
            site_basin.setdefault(r["segment"], r["basin"])
    with open(args.readings, encoding="utf-8") as f:
        readings = list(csv.DictReader(f))

    rows, stats = build_strong_labels(readings, site_basin, DEFAULT_RULES,
                                      years=(args.year_from, args.year_to))
    if not rows:
        print("no label cells produced (check determinands, units, year range). "
              f"unit mismatches: {dict(stats.uom_mismatch)}")
        return 1
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"strong-label cells: {stats.n_cells} over {stats.n_segments} segments "
          f"-> {args.out}")
    print(f"class balance: at_risk={stats.n_pos} ok={stats.n_cells - stats.n_pos} "
          f"({stats.positive_rate:.1%} positive)")
    print("exceedance by determinand (exceed/seen):")
    for det in DEFAULT_RULES:
        if stats.det_seen[det]:
            print(f"  {det:<18} {stats.det_exceed[det]:>5}/{stats.det_seen[det]:<6} "
                  f"({stats.det_exceed[det] / stats.det_seen[det]:.0%})")
    if stats.uom_mismatch:
        print("WARNING unit mismatches skipped:", dict(stats.uom_mismatch))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
