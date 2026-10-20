# River flow vs. proximity

Code and data for the IEEE ICCP 2026 paper *"As the Crow Flies, or as the River Flows?
Graph Topology vs. Proximity in Satellite Water-Quality Monitoring"*.

We wanted to know whether modelling how water flows between river reaches tells you more
about Sentinel-2 water-quality signals than simply looking at what is nearby. So every
flow-graph result is compared with a proximity baseline that has the same number of
neighbours. We ran this on two whole countries: Germany (66,261 reaches, with in-situ
chemistry labels) and Romania (18,692 reaches, no labels).

Short answer: the flow graph doesn't help predict *how polluted* a reach is, but it does
help track *how signals change over time*. Connected reaches rise and fall together more
than equally close unconnected ones, in both countries.

Everything needed to reproduce the paper is in this repository, including about 400 MB of
derived data.

## Getting started

Tested with Python 3.12 (3.10 or later should work).

```bash
python -m venv .venv && . .venv/bin/activate
python -m pip install -r requirements-lock.txt   # the exact versions behind the logs
python -m pip install -e ".[models,data,dev]"
pytest                                   # ~200 tests, under a minute
python scripts/exp_propagation.py        # one Table III row, 30 seconds
python scripts/exp_temporal.py           # the main result, about 15 minutes
```

The first line pins the library versions we used. You can skip it, but other versions can
shift a few XGBoost numbers in the third decimal. The GAT experiment also needs the `gnn`
extra (torch, a large install), see `requirements-lock.txt`.

Scripts run from the repository root and print their results. Long ones print progress to
stderr, sometimes minutes apart. Most need under 1 GB of RAM; the German per-pass runs
need more (Fig. 4 about 3 GB, the resolution sweep about 6 GB). Each script explains its
options with `--help`. They default to the German data, except
`exp_attribution_upstream.py`, `exp_provenance.py`, `exp_inherited.py` and
`build_optical_series.py`, which default to Romania (`--region de` switches).

## Reproducing each result

Run `python scripts/<script>`. The output is what we saved in `results/national/`.

| Paper | Script | Log | Time |
|-------|--------|-----|------|
| Sec. V-A, detection and feature groups | `train.py` | `detection.txt` | 1 min |
| Table II, classifiers, top-decile lift | `run_model_comparison.py` | `model_comparison.txt` | 2 min |
| Table II, GAT vs MLP | `run_gnn.py` (needs the `gnn` extra) | `gnn.txt` | 25-30 min |
| Sec. V-A, satellite vs lab readings | `run_proxy.py`, `run_events.py` | `proxy.txt`, `events.txt` | seconds |
| Sec. V-B, Moran's I and gap-filling | `exp_autocorr.py`, `exp_gapfill.py` | `AUTOCORRELATION.txt`, `GAPFILL.txt` | seconds |
| Table III, static levels | `exp_propagation.py` | `PROPAGATION.txt` | 30 s |
| Table III and Fig. 3, co-fluctuation | `exp_temporal.py` | `COFLUCTUATION.txt` | 15 min |
| Sec. V-B, bearing-matched control | `exp_temporal.py --orient-match` | `COFLUCTUATION_orient.txt` | 10 min |
| Table III, synthetic anomalies | `exp_anomaly.py --exp all` | `anomaly.txt` | 2 min |
| Table III, scaling across basins | `exp_basins.py --random-ref 3` | `BASINS.txt` | 40 min |
| Sec. V-B, results by river size | `exp_conditional.py`, `run_ndci_strata.py` | `CONDITIONAL.txt`, `ndci_strahler.txt` | 30 + 20 min |
| Sec. V-B, Romanian detector | `run_romania.py` | `romania.txt` | 2 min |
| Sec. IV, reversed-flow placebo | `exp_attribution.py` | `ATTRIBUTION.txt` | 30 s |
| Sec. IV, Romanian placebo | `prep_ro_attribution.py`, then `exp_attribution.py` | `ro_attribution_wfd.txt`, `ro_attribution_sat.txt` | 1 min |
| Sec. V-B, upstream source ranking | `exp_attribution_upstream.py --pct 99`, then `--pct 95` (German inputs for the DE log, see `data/README.md`) | `ro_attribution_upstream.txt`, `de_attribution_upstream.txt` | 30 s / 3 min |
| Sec. V-B, inherited vs local pollution | `exp_inherited.py --region de` | `inherited_de.txt` | 10 min |
| Sec. V-B, facility density | `facility_shares.py` | `facility_shares.txt` | 2 s |
| Fig. 4, how exceedances arise | `exp_provenance.py --region de` | `provenance_de.txt` | 3 min |
| Fig. 5, time resolution | `run_resolution_sweep.py --out results/national/resolution_sweep.json` | `resolution_sweep.json` | 30-45 min, 6 GB RAM |
| Sec. IV, multiple-testing correction | `exp_multiplicity.py` | `multiplicity.txt` | 1 s |

Most Romanian runs use the same script with the Romanian files and write to the
matching Romanian log (`ro_*.txt`, `provenance_ro.txt`, `inherited_ro.txt`):

```bash
python scripts/exp_temporal.py --graph data/ro/segment_graph_ro.json --series data/ro/turb_series_ro.csv
python scripts/exp_propagation.py --graph data/ro/segment_graph_ro.json --field data/ro/turbidity_field_ro.csv
python scripts/exp_provenance.py --region ro
```

A few need specific flags (the basin curve, the German upstream ranking, the Romanian
source ranking). Those are spelled out in each script's `--help` and in
[`data/README.md`](data/README.md).

`results/results_data.py` collects the numbers the paper quotes, and the tests check
them against the logs.

### Where the code and the paper differ

Two numbers in the paper don't quite match what the shipped data gives: Fig. 4b for
Romania (+2.0 pp, the paper says +1.8) and Table I's Romanian observation count. Neither
changes a conclusion. [`PAPER_NOTES.md`](PAPER_NOTES.md) explains both, plus a few
wording details.

## Beyond the paper

Some analyses didn't make it into the paper but might be useful:

- `run_ensemble.py` calibrates the detector with a bagged ensemble (`ensemble.txt`).
- `exp_conditional.py` also reports partial Moran's I, variograms and the extra value of
  graph features over standard ones.
- `exp_anomaly.py --exp tracing` tests tracing a synthetic spill back to its source.
  `robustness.txt` repeats some anomaly tests with other decay settings.
- `BASINS_headline.txt` reruns the basin curve at full precision.

One more log, `ag_ablation.txt`, belongs to an agricultural-fusion result from an earlier
study that we withdrew (the gain came from each reach's own cropland, not from upstream).
It has no script and isn't in the paper. It is kept only so the multiple-testing count
matches the paper.

## What's where

```
src/experiments/   the library: river graph, discharge and upstream prior, metrics,
                   cross-validation, the experiments themselves, and data parsers
scripts/           one script per experiment; scripts/ingest/ rebuilds data from raw sources
results/           saved logs, results_data.py, make_figures.py
data/              the derived data, documented in data/README.md
tests/             tests on small synthetic graphs, plus checks of the data and logs
```

`python results/make_figures.py` (with the `figures` extra) redraws Figs. 1, 3, 4 and 5
as PNGs in `figures/`. The paper draws them in TikZ from the same numbers. Fig. 2 is a
schematic with no data.

## Terms

- **reach**: one EU-Hydro river segment (*segment* in code and logs); a node of the graph.
- **proximity control** (*euclidean* in logs): neighbours chosen by straight-line distance.
- **NDCI**: a Sentinel-2 chlorophyll index. **Turbidity**: red-band reflectance.
- **Strahler order**: river size, 1 at the headwaters.
- **common-mode**: each month's network-wide average removed before comparing reaches.
- **WFD**: the EU Water Framework Directive status classes used as Romanian labels.
- **AP / PR-AUC, MRR**: ranking scores for detection and source ranking.

## License

The code is Apache-2.0 (`LICENSE`). The data keeps the terms of its sources, listed in
`data/README.md`. Note that the OpenStreetMap layers are ODbL, which asks you to keep
the attribution and share derived databases under the same terms.
