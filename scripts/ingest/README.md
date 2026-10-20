# Data acquisition (`scripts/ingest/`)

How the files in [`data/`](../../data/) are built from public raw sources. You do not
need this to reproduce the paper: the derived bundle is committed. The parsing and
index logic lives in `experiments.ingest` and is unit-tested in
[`tests/ingest/`](../../tests/ingest). Run the commands from the repository root.

## Credentials

Only the Earth Engine steps need them. Both options are free and need no billing:

```bash
export GEE_SERVICE_ACCOUNT_KEY=/path/to/service-account-key.json   # recommended
export GEE_PROJECT=your-ee-project
# or run `earthengine authenticate` once and set only GEE_PROJECT
```

## Sources to files

| Source | Command | Output |
|--------|---------|--------|
| EU-Hydro river network (geodatabase) | `python scripts/ingest/build_graph.py --gdb <gdb> --out segment_graph_de.json` | `segment_graph_de.json` |
| EEA Waterbase sites (WISE-6 SpatialObject) | `python scripts/ingest/snap_sites.py --sites <csv> > national_sites_all_snapped.csv` | `national_sites_all_snapped.csv` |
| EEA Waterbase chemistry (WISE-6 Disaggregated) | `python scripts/ingest/extract_insitu.py --disagg <csv> > insitu_readings_national_de.csv` | `insitu_readings_national_de.csv` |
| OpenStreetMap (Overpass) | `python scripts/ingest/fetch_osm.py --out osm_industrial_de.json` | `osm_industrial_de.json` |
| Sentinel-2 (Earth Engine) | per-pass export, see below | per-pass archives |
| Per-pass archives | `python scripts/build_optical_series.py --region de --out <prefix>` | monthly series and per-reach fields |
| CORINE land cover (Earth Engine) | static export, see below | `ag_fraction_de_full.csv` |

Extras: `build_graph.py` needs `geo` (geopandas), `fetch_osm.py` needs `osm` (requests),
and the Earth Engine steps need `gee`, for example `pip install -e ".[geo]"`.

## Sentinel-2 per-pass archive

Each usable Sentinel-2 pass is cloud-masked with the scene classification layer, and
the seven indices are averaged over the 20 m pixels on each reach's line geometry
(`experiments.ingest.gee`, index formulas in `experiments.ingest.indices`). The national
run was one Earth Engine export per month, loaded into a database and dumped as the
shipped per-pass CSVs. It needs a Cloud Storage bucket and hours of Earth Engine
compute, so it is described here rather than scripted.

`verify_gee.py` re-fetches a random sample of passes and checks them against the
shipped `pass_obs_national.csv`, including its `risk_score` column:

```bash
python scripts/ingest/verify_gee.py --n 8
# --asset <FeatureCollection of reach geometries> gives an exact per-reach reduction;
# without it a centroid buffer is used (close, not exact).
```

## CORINE agricultural fraction

`ag_fraction_de_full.csv` is one static Earth Engine export: CORINE `landcover`
classes 211-244 masked to agriculture, averaged over each reach buffered by 1 km.
