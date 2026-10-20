# Data bundle

Every derived input the paper's results need ships here (~398 MB). The raw sources
(Sentinel-2 imagery, EEA Waterbase, the EU-Hydro geodatabase, CORINE) are not included.
You only need them to rebuild these files from scratch (last section).

## Compressed files

Files over GitHub's 100 MiB limit ship gzipped. Scripts read them through
`scripts/_bootstrap.open_text`, so commands name the plain `.csv` and nothing needs
unpacking.

| file | shipped as |
|------|------------|
| `de/turb_series_national.csv` | `.csv.gz` (109 MB -> 33 MB) |
| `ro/turb_perdate_ro_full.csv` | `.csv.gz` (178 MB -> 57 MB) |
| `de/pollution_field_national.csv` | `.csv.gz` (80 MB -> 10 MB) |
| `de/turb_perdate_national_full.csv` | four gzip parts `.csv.gz.part00` to `.part03` (613 MB -> 4 x ~49 MB) |

The German parts are line-aligned gzip members. `open_text` reads them in order as one
stream, so no script needs them joined.

Check (md5 of the decompressed text, not of the .gz):

```bash
cat data/de/turb_perdate_national_full.csv.gz.part* | zcat | md5sum   # cb7baab82db8cd82a2b8a12199126f17
cat data/de/turb_perdate_national_full.csv.gz.part* | zcat | wc -l    # 14605120 (header + 14.6M rows)
```

A joined `.csv.gz` or unpacked `.csv` is gitignored.

## Manifest

| file | content | used by |
|------|---------|---------|
| `de/segment_graph_de.json` | 66,261-reach EU-Hydro flow DAG (lat/lon, Strahler, river name, next-down) | all German experiments |
| `de/turb_perdate_national_full.csv.gz.part*` | per-pass turbidity + NDCI, 66,160 reaches, 2020-2023, 14.6M rows (Table I) | `exp_provenance`, `run_resolution_sweep`, `build_optical_series` |
| `de/turb_series_national.csv.gz` | per-(reach, month) mean turbidity + NDCI, 3.07M rows | co-fluctuation, anomaly suite, basins, conditional |
| `de/turbidity_field_national.csv` | per-reach mean turbidity + NDCI | propagation (static levels) |
| `de/turb_perdate_national.csv` | per-pass turbidity + NDCI on the 209 in-situ reaches | `run_proxy` |
| `de/pass_obs_national.csv` | per-pass Sentinel-2 indices on 714 reaches (covering 202 of the 209 in-situ reaches) | `run_events` |
| `de/insitu_readings_national_de.csv` | daily surface-water in-situ readings, 209 reaches, 2020-2023 | autocorrelation, gap-filling, conditional, proxy |
| `de/strong_labels_national_de.csv` | per-(reach, month) exceedance labels | attribution, GNN, Fig. 1 |
| `de/national_sites_all_snapped.csv` | all German Waterbase stations snapped to the graph | label provenance |
| `de/features_national_de.csv` | detection panel, 7,739 rows: optical `o_` (NDWI, MNDWI, NDVI, NDCI, NDTI, TURBIDITY, BSI), upstream prior `p_` (propagates turbidity), topology `t_`, facility pressure `f_` | detection, model comparison, ensemble |
| `de/graph_snapshots_national.npz` | 48-month node features and edges, 2,597-node labelled subgraph | `run_gnn` |
| `de/osm_industrial_de.json` | 113,619 industrial / wastewater facilities | attribution, `f_` features |
| `de/pollution_field_national.csv.gz` | per-(reach, month) mean risk score, 2020-2023 | `exp_attribution` (Sec. IV) |
| `de/ag_fraction_de_full.csv` | per-reach cropland fraction (CORINE) | conditional |
| `ro/segment_graph_ro.json` | 18,692-reach EU-Hydro flow DAG for Romania | all Romanian experiments |
| `ro/turb_perdate_ro_full.csv.gz` | per-pass turbidity + NDCI, 2024-07 to 2026-05, 4,141,108 rows over 34,654 reaches (Table I's 4.1M). The archive covers a wider reach set than the 18,692-reach graph: 2,179,358 rows fall on 18,459 graph reaches, and scripts drop the rest | `exp_provenance`, `build_optical_series` |
| `ro/turb_series_ro.csv` | per-(reach, month) mean turbidity + NDCI, 18,459 reaches x 23 months (411,820 rows, not every reach-month has a clear pass) | co-fluctuation, anomaly suite, basins |
| `ro/turbidity_field_ro.csv` | per-reach mean turbidity + NDCI, 18,459 reaches | propagation, attribution field |
| `ro/osm_industrial_ro.json` | 21,924 industrial / wastewater facilities | attribution |
| `ro/features_ro_national.csv` | per-reach panel with the same feature families as Germany | `run_romania` |
| `ro/segment_labels_ro_national.csv` | WFD-2022 ecological status per snapped reach (2,268 water bodies; `label` = worse than good) | `run_romania`, Romanian attribution targets |

Caveats:
- `de/osm_industrial_de.json` was fetched over a bounding box (lon 8.18-15.03,
  lat 47.28-53.88) smaller than the graph, so 22% of reaches have `f_total == 0` from
  missing coverage, not true absence. This affects the `f_` features and the German
  attribution runs.
- The two per-reach fields were averaged differently: Germany over all passes,
  Romania over monthly means. `build_optical_series.py` reproduces each to within
  1e-4 (rounding at the 4th decimal).
- `de/ag_fraction_de_full.csv` lists reach `RL29000634` twice, as in the raw export.
  It is kept so `CONDITIONAL.txt` reproduces.
- Monthly series and per-reach fields are rounded to 4 decimals (the per-pass archives
  are not). This reproduces every reported statistic to its printed precision.
- `de/pollution_field_national.csv.gz` is the monthly mean of the per-pass risk score
  (`experiments.ingest.indices.risk_score`) per reach. Only the German run of
  `exp_attribution.py` (Sec. IV) uses it.

## Commands for every log

The main README lists the scripts. These are the exact commands, from the repository
root, including the ones that need extra flags.

```bash
RO="--graph data/ro/segment_graph_ro.json --series data/ro/turb_series_ro.csv"
DE_UP="--graph data/de/segment_graph_de.json --series data/de/turb_series_national.csv --osm data/de/osm_industrial_de.json"
L=results/national

# Germany, no extra flags
python scripts/train.py                 > $L/detection.txt
python scripts/run_model_comparison.py  > $L/model_comparison.txt
python scripts/run_gnn.py               > $L/gnn.txt
python scripts/run_proxy.py             > $L/proxy.txt
python scripts/run_events.py            > $L/events.txt
python scripts/exp_autocorr.py          > $L/AUTOCORRELATION.txt
python scripts/exp_gapfill.py           > $L/GAPFILL.txt
python scripts/exp_propagation.py       > $L/PROPAGATION.txt
python scripts/exp_temporal.py          > $L/COFLUCTUATION.txt
python scripts/exp_temporal.py --orient-match > $L/COFLUCTUATION_orient.txt
python scripts/exp_anomaly.py --exp all > $L/anomaly.txt
python scripts/exp_basins.py --random-ref 3 > $L/BASINS.txt
python scripts/exp_conditional.py       > $L/CONDITIONAL.txt
python scripts/run_ndci_strata.py       > $L/ndci_strahler.txt
python scripts/run_romania.py           > $L/romania.txt
python scripts/exp_attribution.py       > $L/ATTRIBUTION.txt
python scripts/exp_provenance.py --region de > $L/provenance_de.txt
python scripts/exp_inherited.py --region de  > $L/inherited_de.txt
python scripts/facility_shares.py       > $L/facility_shares.txt
python scripts/run_ensemble.py          > $L/ensemble.txt
python scripts/run_resolution_sweep.py --out $L/resolution_sweep.json
python scripts/exp_multiplicity.py      > $L/multiplicity.txt
(python scripts/exp_attribution_upstream.py --pct 99 $DE_UP
 python scripts/exp_attribution_upstream.py --pct 95 $DE_UP) > $L/de_attribution_upstream.txt

# Romania
python scripts/exp_propagation.py --graph data/ro/segment_graph_ro.json \
    --field data/ro/turbidity_field_ro.csv           > $L/ro_propagation.txt
python scripts/exp_temporal.py $RO                   > $L/ro_cofluctuation.txt
python scripts/exp_temporal.py $RO --orient-match    > $L/ro_cofluctuation_orient.txt
python scripts/exp_anomaly.py $RO --exp all          > $L/ro_anomaly.txt
python scripts/exp_basins.py $RO --k-grid 1,2,3,5,10,20,50,100 \
    --sample 1500 --n-boot 300                        > $L/ro_basins.txt
python scripts/exp_provenance.py --region ro         > $L/provenance_ro.txt
python scripts/exp_inherited.py --region ro          > $L/inherited_ro.txt
(python scripts/exp_attribution_upstream.py --pct 99
 python scripts/exp_attribution_upstream.py --pct 95) > $L/ro_attribution_upstream.txt
for v in wfd sat; do   # ro_attribution_wfd.txt and ro_attribution_sat.txt
  python scripts/prep_ro_attribution.py --variant $v --out build/ro_attr_$v
  python scripts/exp_attribution.py --graph data/ro/segment_graph_ro.json \
      --strong build/ro_attr_$v/targets.csv --osm data/ro/osm_industrial_ro.json \
      --field build/ro_attr_$v/field.csv > $L/ro_attribution_$v.txt
done

# ag_ablation.txt is a frozen log of a withdrawn result (see PAPER_NOTES.md). It has no
# script in this release.

# Extras (not in the paper)
python scripts/exp_basins.py --sample 3000 --n-boot 2000 > $L/BASINS_headline.txt
python scripts/exp_basins.py $RO --k-grid 1,2,3,5,10,20,50,100 --sample 3000 --n-boot 2000 \
                                                     > $L/ro_basins_headline.txt
python scripts/exp_anomaly.py --exp tracing          > $L/tracing_reversed.txt
python scripts/exp_anomaly.py --exp residual,break --euc-decay-km 30 > $L/break_euc_decay.txt
(python scripts/exp_anomaly.py --exp break --brief
 python scripts/exp_anomaly.py --exp break --brief --inject-decay-km 15
 python scripts/exp_anomaly.py --exp tracing --brief
 python scripts/exp_anomaly.py --exp tracing --brief --hop-decay 0.6) > $L/robustness.txt
```

## Rebuilding from raw sources

- Graph: `scripts/ingest/build_graph.py` (EU-Hydro geodatabase).
- In-situ: `scripts/ingest/snap_sites.py`, then `extract_insitu.py` (EEA Waterbase).
- Facilities: `scripts/ingest/fetch_osm.py` (Overpass).
- Sentinel-2 indices: `src/experiments/ingest/gee.py` (cloud mask, index image, reach
  means on Earth Engine) and `indices.py` (the index formulas). `scripts/ingest/verify_gee.py` spot-checks shipped
  values against Earth Engine. See `scripts/ingest/README.md` for credentials.
- Monthly series and per-reach fields: `scripts/build_optical_series.py` rebuilds them
  from the shipped per-pass archives.
- Labels: `scripts/build_strong_labels.py`.

The feature panels, GNN snapshots and `pass_obs_national.csv` were joined from our
per-pass Sentinel-2 database with scripts that are not part of this package. They ship
ready to use.

## Sources and licences

Retain these credits when redistributing:

- **River network**: EU-Hydro, Copernicus Land Monitoring Service / EEA (free and open).
- **Satellite imagery**: modified Copernicus Sentinel-2 data, via Google Earth Engine.
- **In-situ chemistry and WFD status**: EEA Waterbase (Water Quality) and the WISE WFD
  database, European Environment Agency.
- **Facilities**: (c) OpenStreetMap contributors, Open Database License (ODbL). The
  `osm_industrial_*.json` files are ODbL-derived: keep the attribution and share-alike terms.
- **Land cover**: CORINE Land Cover, Copernicus Land Monitoring Service.
