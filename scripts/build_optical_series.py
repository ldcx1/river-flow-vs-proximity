"""Rebuild the monthly optical inputs from the shipped per-pass archive.

Writes, for the reaches in the graph:
  <prefix>_series.csv   one row per (reach, month): mean turbidity and NDCI
  <prefix>_field.csv    one row per reach: its mean value
Values are rounded to 4 decimals, as in the shipped files. Reproduces
turb_series_{national,ro}.csv and turbidity_field_{national,ro}.csv.

The two shipped fields were averaged differently: Germany over all passes,
Romania over monthly means (each month counts equally). --field-mean
picks one; the default reproduces the shipped file for each country.

    python scripts/build_optical_series.py --region ro --out build/ro
"""
import argparse
import os

import pandas as pd

import _bootstrap
from experiments.graph import RiverGraph

REGIONS = {
    "de": (_bootstrap.data_de, "segment_graph_de.json", "turb_perdate_national_full.csv", "passes"),
    "ro": (_bootstrap.data_ro, "segment_graph_ro.json", "turb_perdate_ro_full.csv", "months"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", choices=sorted(REGIONS), default="ro")
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--field-mean", choices=("passes", "months"),
                    help="average the field over passes or monthly means")
    args = ap.parse_args()

    data, gname, pname, field_mean = REGIONS[args.region]
    field_mean = args.field_mean or field_mean
    nodes = set(RiverGraph.from_json(data(gname)).nodes)
    with _bootstrap.open_text(data(pname)) as fh:
        pp = pd.read_csv(fh, usecols=["object_id", "acquired_at", "turb", "ndci"])
    pp = pp[pp.object_id.isin(nodes)]
    pp["ym"] = pp.acquired_at.str[:7]
    series = pp.groupby(["object_id", "ym"])[["turb", "ndci"]].mean().reset_index()
    by = pp if field_mean == "passes" else series
    field = by.groupby("object_id")[["turb", "ndci"]].mean().reset_index()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    series.round(4).to_csv(args.out + "_series.csv", index=False)
    field.round(4).to_csv(args.out + "_field.csv", index=False)
    print(f"{len(series)} reach-months, {len(field)} reaches -> {args.out}_{{series,field}}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
