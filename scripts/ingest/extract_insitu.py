#!/usr/bin/env python3
"""Parse EEA Waterbase disaggregated data into surface-water in-situ readings.

Keeps only rows whose site snapped (--sites-snap), whose determinand is
optically relevant, whose matrix is water, and whose water-body category is
surface water (drops the ~38% groundwater rows). See
:func:`experiments.ingest.waterbase.accept_row`. Writes the readings CSV to
stdout. The shipped file is data/de/insitu_readings_national_de.csv.

    python scripts/ingest/extract_insitu.py \\
        --disagg /path/Waterbase_..._T_WISE6_DisaggregatedData.csv \\
        > insitu_readings_national_de.csv
"""
import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from experiments.ingest import waterbase as wb  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data" / "de"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disagg", required=True, help="Waterbase DisaggregatedData CSV")
    ap.add_argument("--sites-snap", default=str(DATA / "national_sites_all_snapped.csv"),
                    help="snapped-site CSV from snap_sites.py")
    args = ap.parse_args()
    disagg = args.disagg
    if not os.path.exists(disagg):
        print(f"Waterbase DisaggregatedData CSV not found: {disagg}", file=sys.stderr)
        return 1
    sites = args.sites_snap
    site_seg = {}
    with open(sites, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            site_seg.setdefault(r["site"], (r["segment"], r["basin"]))

    w = csv.writer(sys.stdout)
    w.writerow(["site", "segment", "basin", "determinand", "date", "value", "uom"])
    kept = 0
    with open(disagg, encoding="utf-8") as fh:
        rd = csv.reader(fh)
        next(rd, None)                               # skip header
        for row in rd:
            out = wb.accept_row(row, site_seg)
            if out:
                w.writerow(out)
                kept += 1
    print(f"kept {kept} surface-water readings across {len(site_seg)} snapped sites",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
