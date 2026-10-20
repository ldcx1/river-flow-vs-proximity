#!/usr/bin/env python3
"""Download OpenStreetMap industrial and wastewater facilities from Overpass.

Tiles the Germany bounding box and queries each facility selector per tile so no
single Overpass request times out. Requires ``requests`` (the ``osm`` extra) and
network access. The shipped file is data/de/osm_industrial_de.json.

    python scripts/ingest/fetch_osm.py --out osm_industrial_de.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from experiments.ingest import osm  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output JSON path")
    args = ap.parse_args()
    out = Path(args.out)

    facilities = osm.fetch_facilities(osm.DE_BBOX)
    if not facilities:
        print("nothing fetched (all Overpass mirrors failed?)", file=sys.stderr)
        return 1
    out.write_text(json.dumps(facilities))
    by_kind = {}
    for f in facilities:
        by_kind[f["kind"]] = by_kind.get(f["kind"], 0) + 1
    print(f"wrote {len(facilities)} facilities to {out}")
    print(f"by kind: {by_kind}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
