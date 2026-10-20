#!/usr/bin/env python3
"""Build the reach graph (data/de/segment_graph_de.json) from EU-Hydro River_Net_l.

Reads the geodatabase with geopandas, projects reach centroids to WGS84, and
hands per-reach records to the pure :func:`experiments.ingest.euhydro.assemble_graph`
(which breaks cycles and derives the upstream lists).

    python scripts/ingest/build_graph.py --gdb /path/to/EU-Hydro.gdb \\
        --out segment_graph_de.json

Requires ``geopandas`` (``pip install -e ".[geo]"`` or your GDAL stack).
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from experiments.ingest import euhydro  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gdb", required=True, help="EU-Hydro geodatabase path")
    ap.add_argument("--out", required=True, help="output graph JSON path")
    args = ap.parse_args()
    gdb = args.gdb
    if not os.path.exists(gdb):
        print(f"EU-Hydro geodatabase not found: {gdb}", file=sys.stderr)
        return 1
    import geopandas as gpd
    gdf = gpd.read_file(gdb, layer="River_Net_l",
                        columns=["OBJECT_ID", "STRAHLER", "LENGTH", "NEXTDOWNID", "nameText"])
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    cent = gpd.GeoSeries(gdf.geometry.centroid, crs=gdf.crs).to_crs(4326)
    gdf["lon"], gdf["lat"] = cent.x.values, cent.y.values

    def name(v):
        return v if isinstance(v, str) and v.strip() else None

    def nd(v):
        return v if isinstance(v, str) and v.strip() else None

    recs = [{"object_id": r.OBJECT_ID,
             "strahler": r.STRAHLER if r.STRAHLER == r.STRAHLER else 1,
             "length_m": r.LENGTH if r.LENGTH == r.LENGTH else 0.0,
             "lat": r.lat, "lon": r.lon,
             "river_name": name(r.nameText), "next_down": nd(r.NEXTDOWNID)}
            for r in gdf.itertuples(index=False)]
    broken = []
    graph = euhydro.assemble_graph(recs, broken=broken)
    Path(args.out).write_text(json.dumps(graph))
    outlets = sum(1 for v in graph.values() if not v["next_down"])
    print(f"wrote {len(graph)} reaches ({outlets} outlets, {len(broken)} cycles broken) "
          f"to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
