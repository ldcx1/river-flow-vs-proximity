"""Canonical national-scale results behind the paper's tables and figures.

Most dicts mirror a captured run log under results/national/ (ATTRIBUTION ->
ATTRIBUTION.txt, GAPFILL -> GAPFILL.txt, and so on). The resolution sweep (Fig. 5)
is read directly from results/national/resolution_sweep.json and has no dict here.
After editing a number, regenerate the figures (make_figures.py) and keep the paper
in sync. tests/test_results_consistency.py pins many of these values.

CI = 95% cluster bootstrap (resampling segments, targets or pairs), 2000 resamples
unless noted.
"""

DATA = {
    "graph_segments": 66261,
    "optical_segments": 66160, "optical_months": 48,
    # Sentinel-2 per-pass observations (Table I, abstract): the row count of the
    # shipped archive data/de/turb_perdate_national_full.csv.gz.part* (14,605,119).
    "s2_pass_obs_M": 14.6,
    "facilities": 113619,
    # Raw daily surface-water readings (Table I). The event pipeline collapses
    # co-located same-day duplicates to 45,873 (see EVENTS).
    "insitu_segments": 209, "insitu_readings": 46154, "insitu_span": "2020-2023",
    # Station selection (paper Sec. Data): 1,506 candidate Waterbase sites within 1 km
    # of a reach (the rows of national_sites_all_snapped.csv), 38% of records
    # groundwater (discarded). The 38% is documented only: the raw
    # Waterbase station table is not shipped.
    "insitu_sites_candidate": 1506, "groundwater_record_frac": 0.38,
    # 8,100 labelled reach-months, 994 positive (12.3%). The detection set is the
    # 7,739-row subset with complete optical coverage (DETECTION, 12.6% positive).
    # eb_targets = attribution target count (ATTRIBUTION["n_targets"]).
    "strong_cells": 8100, "strong_pos_frac": 0.123, "eb_targets": 858,
}

# ---- Spatial autocorrelation of in-situ determinands (AUTOCORRELATION.txt) ----
# Moran's I on the flow graph vs a degree-matched Euclidean graph.
# diff > 0 means flow organises the values beyond geography.
AUTOCORRELATION = {  # name: (n, I_flow, I_euc, diff, p_diff, I_rewired)
    "Dissolved oxygen": (56, 0.168, 0.107, 0.061, 0.658, -0.110),
    "Nitrate":          (68, 0.333, 0.261, 0.072, 0.574, -0.011),
    "Exceedance flag":  (70, 0.273, 0.265, 0.009, 0.966, -0.003),
}

# ---- Gap-filling (GAPFILL.txt) ----
# RMSE(control) - RMSE(graph), so positive = graph better. Each control is (mean, lo, hi).
GAPFILL = {  # name: (n, vs_global, vs_euclidean, vs_permuted)
    "Dissolved oxygen": (191, (-0.046, -0.103, 0.007), (0.053, -0.059, 0.174), (0.010, -0.065, 0.084)),
    "Nitrate":          (208, (0.032, -0.266, 0.336), (0.207, -0.505, 0.884), (1.236, 0.709, 1.827)),
    "Exceedance flag":  (209, (-0.016, -0.048, 0.018), (0.067, 0.012, 0.122), (0.059, 0.015, 0.101)),
}

# ---- Source attribution, Germany, unrestricted candidates (ATTRIBUTION.txt) ----
# 858 in-situ exceedance targets with a facility within 1.5 km. Each method ranks
# candidate source reaches on the 66k-reach optical field, and hits are scored
# against OSM facilities. The paper quotes "vs reversed" (+0.056) and
# "vs euclidean" (-0.041) as the unrestricted variant. Field: the monthly mean risk
# score, data/de/pollution_field_national.csv.gz.
ATTRIBUTION = {
    "n_targets": 858,
    "mrr":   {"graph": 0.585, "euclidean": 0.627, "random": 0.723, "permuted": 0.589, "reversed": 0.530},
    "hit1":  {"graph": 0.512, "euclidean": 0.570, "random": 0.545, "permuted": 0.515, "reversed": 0.437},
    "nonempty": {"graph": 1.00, "euclidean": 1.00, "random": 1.00, "permuted": 0.97, "reversed": 0.91},
    "delta": {  # MRR graph - control (mean, lo, hi)
        "vs reversed":  (0.056, 0.025, 0.087),
        "vs reversed (paired)": (0.046, 0.013, 0.078),
        "vs permuted":  (-0.004, -0.024, 0.016),
        "vs euclidean": (-0.041, -0.069, -0.016),
        "vs random":    (-0.137, -0.173, -0.102),
    },
}

# ---- Source attribution, Romania, unrestricted candidates ----
# Same experiment as ATTRIBUTION on the RO network and RO OSM facilities. Only 18%
# of RO reaches lie within 1.5 km of a facility (Germany 49%), yet flow-upstream
# tracing still loses to proximity and to random. OSM facilities sit in downstream
# urban reaches, while upstream tracing heads to rural headwaters.
# Targets do not use the satellite field: "wfd" = reaches in a WFD water body below
# good status (segment_labels_ro_national.csv). "sat" = top-1% turbidity reach-months
# (robustness). Field = turbidity.
# Run scripts/prep_ro_attribution.py, then scripts/exp_attribution.py.
# Logs: results/national/ro_attribution_{wfd,sat}.txt.
RO_ATTRIBUTION = {
    "frac_reach_within_1500m_of_facility": 0.180,   # Germany: 48.8%
    "n_facilities": 21924,                           # Germany: 113,619
    "wfd": {   # WFD-degraded targets, static mean-turbidity field
        "n_targets": 5192,
        "mrr":  {"graph": 0.264, "euclidean": 0.340, "random": 0.378, "permuted": 0.269, "reversed": 0.385},
        "hit1": {"graph": 0.176, "euclidean": 0.228, "random": 0.181, "permuted": 0.182, "reversed": 0.198},
        "delta": {  # MRR graph - control (mean, lo, hi)
            "vs euclidean": (-0.076, -0.087, -0.065),
            "vs permuted":  (-0.005, -0.013, 0.003),
            "vs reversed":  (-0.120, -0.134, -0.106),
            "vs random":    (-0.114, -0.127, -0.101),
        },
    },
    "sat": {   # top-1% turbidity reach-month targets, per-month field
        "n_targets": 2453,
        "mrr":  {"graph": 0.213, "euclidean": 0.311, "random": 0.369, "permuted": 0.220, "reversed": 0.385},
        "hit1": {"graph": 0.146, "euclidean": 0.199, "random": 0.173, "permuted": 0.161, "reversed": 0.247},
        "delta": {
            "vs euclidean": (-0.098, -0.112, -0.083),
            "vs permuted":  (-0.007, -0.016, 0.002),
            "vs reversed":  (-0.172, -0.191, -0.153),
            "vs random":    (-0.156, -0.175, -0.137),
        },
    },
}

# ---- Source attribution restricted to upstream candidates, Romania ----
# scripts/exp_attribution_upstream.py, log ro_attribution_upstream.txt.
# Targets are point-source events (top local-excess reach-months). Both methods rank
# the same upstream candidate pool: the graph by flow-weighted contribution, the
# control by distance. So proximity can no longer pick downstream urban facilities.
# hit = ranked reach within R m of an OSM facility. base_rate = share of the target's
# upstream pool near a facility. lift = hit@1 / base_rate.
# delta_* = graph - proximity (positive favours the graph).
# Result: unlike the unrestricted test, the graph does not lose. It ties at most radii, beats proximity on MRR at
# 1.5 km and 3 km, and is marginally worse at hit@1. Split by base-rate tertile, the
# MRR edge is significant only in the middle third. The low third has no facility
# near any upstream reach, so it carries no information.
RO_ATTRIBUTION_UPSTREAM = {
    "top1pct": {   # strict point-source targets, low power
        "n_targets": 128,
        "delta_mrr": {  # radius_m: (mean, lo, hi), all include 0
            500: (-0.004, -0.045, 0.037), 1000: (0.005, -0.047, 0.056),
            1500: (0.009, -0.046, 0.066), 3000: (0.006, -0.059, 0.073),
        },
    },
    "top5pct": {   # main target set
        "n_targets": 1239,
        "base_rate":  {500: 0.032, 1000: 0.086, 1500: 0.145, 3000: 0.301},
        "graph_hit1": {500: 0.038, 1000: 0.113, 1500: 0.179, 3000: 0.367},
        "euclid_hit1":{500: 0.055, 1000: 0.127, 1500: 0.193, 3000: 0.393},
        "delta_mrr": {  # radius_m: (mean, lo, hi), CI excludes 0 at 1500 and 3000
            500: (-0.005, -0.018, 0.008), 1000: (0.009, -0.008, 0.026),
            1500: (0.021, 0.001, 0.040), 3000: (0.021, 0.001, 0.042),
        },
        "delta_lift_hit1": {  # radius_m: (mean, lo, hi) on hit@1 / base_rate
            500: (-0.54, -1.03, -0.07), 1000: (-0.16, -0.41, 0.09),
            1500: (-0.09, -0.26, 0.08), 3000: (-0.09, -0.18, 0.01),
        },
        # delta_mrr at 1500 m by base-rate tertile: (mean, lo, hi, n)
        "strat_mrr_1500m": {
            "low":  (0.000, 0.000, 0.000, 413),   # base rate 0
            "med":  (0.042, 0.005, 0.078, 413),
            "high": (0.020, -0.027, 0.068, 413),
        },
    },
}

# ---- Same restricted test on Germany (log de_attribution_upstream.txt) ----
# Germany is saturated with facilities: the upstream base rate is 47% at 1.5 km and
# 70% at 3 km (Romania 15% and 30%), so lift is about 1 and there is little room to
# discriminate. The graph ties proximity at every radius. Layout as in
# RO_ATTRIBUTION_UPSTREAM.
DE_ATTRIBUTION_UPSTREAM = {
    "top1pct": {
        "n_targets": 908,
        "delta_mrr": {  # all include 0
            500: (-0.009, -0.030, 0.011), 1000: (0.010, -0.011, 0.030),
            1500: (0.015, -0.005, 0.034), 3000: (0.002, -0.011, 0.014),
        },
    },
    "top5pct": {
        "n_targets": 1121,
        "base_rate":  {500: 0.131, 1000: 0.311, 1500: 0.467, 3000: 0.702},
        "graph_hit1": {500: 0.144, 1000: 0.380, 1500: 0.518, 3000: 0.709},
        "euclid_hit1":{500: 0.161, 1000: 0.386, 1500: 0.540, 3000: 0.723},
        "delta_mrr": {  # all include 0
            500: (-0.011, -0.031, 0.010), 1000: (0.014, -0.007, 0.035),
            1500: (0.003, -0.015, 0.021), 3000: (0.002, -0.008, 0.012),
        },
        "delta_lift_hit1": {
            500: (-0.14, -0.34, 0.06), 1000: (-0.02, -0.11, 0.08),
            1500: (-0.05, -0.10, 0.01), 3000: (-0.02, -0.04, 0.00),
        },
        "strat_mrr_1500m": {  # (mean, lo, hi, n), all include 0
            "low":  (-0.020, -0.042, 0.003, 374),
            "med":  (0.026, -0.016, 0.067, 374),
            "high": (0.004, -0.020, 0.028, 373),
        },
    },
}

# ---- Static optical field: predict a reach from its neighbour (PROPAGATION.txt) ----
# Spearman r between a reach and its flow-downstream partner, its nearest Euclidean
# partner, a random partner, and its flow-upstream partner (reversed).
# diff = r_flow - r_euclidean (Table III, static levels). The Euclidean partner is
# the globally nearest reach, not distance-matched, so a proximity win here is
# uninformative.
PROPAGATION = {  # signal: (r_flow, r_euclidean, r_random, diff_lo, diff_hi, r_reversed)
    "Turbidity": (0.443, 0.541, -0.022, -0.127, -0.069, 0.381),
    "NDCI":      (0.473, 0.454, -0.006, -0.012, 0.049, 0.395),
}

# ---- Temporal co-fluctuation, Germany (COFLUCTUATION.txt, Table III, Fig. 3) ----
# Mean pairwise time-correlation of flow-connected pairs vs distance-matched
# unconnected pairs. perm_p uses a 2000-sample permutation null (floor 0.0005).
COFLUCTUATION = {  # (signal, deseason): (c_flow, c_euc, diff, lo, hi, perm_p, n_pairs)
    ("Turbidity", "common-mode"): (0.608, 0.579, 0.029, 0.019, 0.038, 0.0005, 1908),
    ("Turbidity", "seasonal"):    (0.717, 0.698, 0.019, 0.011, 0.027, 0.004, 1908),
    ("NDCI", "common-mode"):      (0.584, 0.493, 0.091, 0.076, 0.107, 0.0005, 1908),
    ("NDCI", "seasonal"):         (0.666, 0.590, 0.076, 0.064, 0.089, 0.0005, 1908),
}

# ---- Co-fluctuation under a bearing-matched control (COFLUCTUATION_orient.txt) ----
# exp_temporal.py --orient-match. Each control pair matches its flow
# pair's compass bearing as well as its distance, so a field that is merely smoother
# along the river axis affects both equally. All four CIs still exclude 0. n drops
# from 1908 to 1098 because fewer sources have a bearing-matched control.
# On a synthetic anisotropic field this control forms no pairs at all
# (tests/test_temporal.py), so the evidence is the real-data effect surviving, not a
# synthetic null.
COFLUCTUATION_ORIENT = {  # (signal, deseason): (c_flow, c_euc, diff, lo, hi, perm_p, n_pairs)
    ("Turbidity", "common-mode"): (0.599, 0.572, 0.026, 0.015, 0.038, 0.007, 1098),
    ("Turbidity", "seasonal"):    (0.713, 0.693, 0.021, 0.012, 0.030, 0.013, 1098),
    ("NDCI", "common-mode"):      (0.570, 0.469, 0.102, 0.083, 0.120, 0.0005, 1098),
    ("NDCI", "seasonal"):         (0.659, 0.573, 0.086, 0.070, 0.103, 0.0005, 1098),
}

# ---- Synthetic-injection anomaly suite, Germany (anomaly.txt, Table III) ----
# ci_kind says what each interval resamples:
#   "cell-bootstrap": segments/cells on one injected field (residual, break, tracing).
#   "injection-run":  the ~20 random-injection runs on one fixed field. This is
#                     Monte-Carlo noise, not a river-population resample, so these
#                     effects are excluded from the FDR family (exp_multiplicity.py).
# Injections follow the flow by construction, so these are sanity checks.
# Tracing's euclidean and reversed controls are 0 by construction (a flow-injected
# source cannot be reached by a Euclidean or downstream walk, see tracing_reversed.txt),
# so its real contrast is spatio-temporal vs static graph tracing.
ANOMALY = {
    "residual": {"label": "Graph-expectation residual", "ci_kind": "cell-bootstrap",
           "vs_node": (0.029, 0.020, 0.038),
           "vs_euclidean": (-0.002, -0.006, 0.001), "frac_dropped": 0.34},
    "break": {"label": "Coherence-violation", "ci_kind": "cell-bootstrap",
           "vs_node": (0.114, 0.099, 0.129),
           "vs_euclidean": (0.007, 0.002, 0.012)},
    "recovery": {"label": "Connected-event F1 (injection-run interval)",
           "ci_kind": "injection-run", "flow": 0.059, "euc": 0.040,
           "delta": (0.019, 0.013, 0.025)},
    "denoise": {"label": "Network denoiser precision (injection-run interval)",
           "ci_kind": "injection-run", "flow": 0.025, "euc": 0.018,
           "nrej_flow": 0.257, "nrej_euc": 0.019, "delta": (0.007, 0.005, 0.009)},
    "tracing": {"label": "Spatio-temporal attribution", "ci_kind": "cell-bootstrap",
           "static_top1": 0.317, "st_top1": 0.367,
           "euc_top1": 0.000, "mrr_static": 0.449, "mrr_euc": 0.112, "mrr_st": 0.534,
           "mrr_reversed": 0.000,
           "delta": (0.050, 0.017, 0.092)},
}

# ---- Conditional and stratified analyses (CONDITIONAL.txt) ----
# Only cofluct_strahler (turbidity co-fluctuation by source Strahler bin) is quoted in
# the paper. Magnitudes vary slightly across software versions, the signs do not.
CONDITIONAL = {
    # Moran's I after residualising on reach covariates. Dissolved oxygen and nitrate
    # organise along flow (diff > 0), turbidity along geography (diff < 0).
    "partial_moran": {  # name: (I_flow, I_euc, diff)
        "Dissolved oxygen": (0.145, 0.058, 0.087),
        "Nitrate": (0.311, 0.201, 0.109),
        "Turbidity": (0.216, 0.445, -0.230),
    },
    # stream / Euclidean semivariance ratio (< 1 = flow pairs more similar)
    "variogram_ratio": {"Dissolved oxygen": 0.717, "Nitrate": 0.911, "Turbidity": 0.899},
    # turbidity co-fluctuation diff by source Strahler bin: (diff, lo, hi)
    "cofluct_strahler": {
        "Headwater (1-2)": (0.020, 0.009, 0.031),
        "Mid (3-4)": (0.014, -0.004, 0.032),
        "Large (5+)": (0.008, -0.005, 0.022),
    },
    "cofluct_rho": -1.0,   # rank trend of cofluct_strahler across bins
    # ROC-AUC with and without graph features on top of node + Euclidean features
    "incremental_auc": {"base": 0.968, "full": 0.976, "gain": (0.008, 0.004, 0.012)},
}

# ---- Basin accretion, Germany (BASINS.txt, Table III scaling row) ----
# Co-fluctuation diff (turbidity, common-mode) as basins are added largest first
# (729 basins). Paper run: 1,200 sampled sources and 200 resamples per point
# (exp_basins.py defaults). At ~800 pairs per point the wiggles are mostly sampling
# noise. The k=3 and k=729 rows agree only after rounding (0.0352 vs 0.0349).
BASINS = {
    "curve": [  # (k, n_segments, diff, lo, hi)
        (1, 15979, 0.012, -0.002, 0.029), (2, 30033, 0.023, 0.008, 0.036),
        (3, 39020, 0.035, 0.020, 0.050), (5, 48845, 0.017, 0.003, 0.031),
        (10, 53250, 0.017, 0.004, 0.031), (20, 57050, 0.026, 0.014, 0.040),
        (50, 61365, 0.019, 0.006, 0.032), (100, 63359, 0.026, 0.012, 0.043),
        (200, 64823, 0.023, 0.007, 0.040), (400, 65767, 0.020, 0.003, 0.033),
        (729, 66160, 0.035, 0.020, 0.050),
    ],
    "full_set_value": 0.035,
}

# Same curve at the headline co-fluctuation settings (3,000 sources, 2,000
# resamples): BASINS_headline.txt / ro_basins_headline.txt. Not in the paper. The
# full-set point equals the COFLUCTUATION turbidity common-mode cell, and every CI
# excludes 0 (including Germany's k=1).
BASINS_HEADLINE = {
    "de": [
        (1, 15979, 0.018, 0.008, 0.028), (2, 30033, 0.024, 0.015, 0.033),
        (3, 39020, 0.025, 0.015, 0.035), (5, 48845, 0.020, 0.011, 0.030),
        (10, 53250, 0.025, 0.016, 0.035), (20, 57050, 0.024, 0.013, 0.034),
        (50, 61365, 0.023, 0.013, 0.032), (100, 63359, 0.027, 0.017, 0.037),
        (200, 64823, 0.027, 0.017, 0.036), (400, 65767, 0.025, 0.015, 0.034),
        (729, 66160, 0.029, 0.019, 0.038),
    ],
    "ro": [
        (1, 4874, 0.066, 0.050, 0.084), (2, 6565, 0.063, 0.047, 0.080),
        (3, 8236, 0.066, 0.049, 0.083), (5, 10761, 0.057, 0.040, 0.074),
        (10, 13060, 0.047, 0.029, 0.064), (20, 14790, 0.027, 0.010, 0.043),
        (50, 15996, 0.054, 0.037, 0.071), (100, 16406, 0.056, 0.039, 0.074),
        (1344, 18459, 0.046, 0.029, 0.062),
    ],
}

# ---- Decay robustness (robustness.txt, break_euc_decay.txt) ----
# break_*: neighbour-break graph-vs-Euclidean delta with the injection decay matched
# to the scoring decay (30 km) or decoupled from it (15 km). tracing_hop*: tracing
# spatio-temporal minus static delta at hop decay 0.3 and 0.6.
# break_euc_decay: break delta against a decay-weighted Euclidean control (the main
# result uses an unweighted control). Every CI excludes 0.
ROBUST = {
    "break_matched": (0.0074, 0.0024, 0.0120), "break_decoupled": (0.0079, 0.0042, 0.0117),
    "tracing_hop0.3": (0.050, 0.017, 0.092), "tracing_hop0.6": (0.108, 0.058, 0.167),
    "break_euc_decay": (0.006, 0.001, 0.011),
}

# ============================================================================
# Supervised detection and related analyses. Repeated 5x5 segment-grouped CV,
# tie-aware metrics, segment cluster-bootstrap CIs.
# Logs: results/national/{detection,ag_ablation,proxy,romania}.txt.
# ============================================================================

# ---- Detection: predict in-situ exceedance from satellite features (detection.txt) ----
# Optics = 7 public indices (NDWI, MNDWI, NDVI, NDCI, NDTI, TURBIDITY, BSI). The
# flow prior (p_) propagates TURBIDITY.
DETECTION = {
    "n_segments": 209, "n_rows": 7739, "prevalence": 0.126,
    # config: (ROC mean, ROC sd, PR mean, PR sd)
    "optical":            (0.605, 0.002, 0.183, 0.002),
    "optical+physics":    (0.603, 0.004, 0.183, 0.006),
    "optical+topology":   (0.615, 0.011, 0.187, 0.015),
    # NOTE: "all" is the optical+physics+topology stack behind the +0.003 ablation
    # delta. Table II's "All features" XGBoost cell (0.599/0.178) is detection.txt's
    # "all + pressure" config instead. Do not sync the table to this entry.
    "all":                (0.598, 0.010, 0.181, 0.015),
    "context_no_optical": (0.572, 0.012, 0.162, 0.009),
    # PR-AUC deltas over optical (point, lo, hi). Both include 0.
    "d_physics_vs_optical": (0.002, -0.019, 0.021),
    "d_all_vs_optical":     (0.003, -0.027, 0.035),
}

# ---- Optical + agricultural land-cover fusion (ag_ablation.txt) ----
# Withdrawn: not in the paper and not a finding. Kept only because the
# multiplicity family in exp_multiplicity.py counts these two effects. The log is
# archived; its generating script is not part of this release.
AG = {  # PR-AUC mean per config
    "optical": 0.183, "opt+local": 0.210, "opt+upstream": 0.241, "opt+all": 0.228,
    "delta": {  # PR-AUC delta (point, lo, hi)
        "upstream vs optical": (0.065, 0.008, 0.127),
        "all-ag vs optical":   (0.053, 0.004, 0.105),
        "upstream vs local":   (0.033, -0.053, 0.114),   # graph-specific contrast, includes 0
    },
}

# ---- Proxy correspondence: satellite index vs in-situ determinand (proxy.txt) ----
PROXY = {  # name: (Spearman r, lo, hi, n_pairs)
    "Ammonium~turbidity":         (0.204, 0.157, 0.248, 6083),
    "Dissolved oxygen~turbidity": (0.094, 0.059, 0.128, 7461),
    "BOD5~turbidity":             (0.065, 0.008, 0.118, 4908),
    "Nitrate~turbidity":          (0.067, 0.012, 0.122, 6549),
    "Chlorophyll a~NDCI":         (0.003, -0.349, 0.340, 84),   # only 84 pairs
}

# ---- Uncertainty-aware ensemble (scripts/run_ensemble.py, ensemble.txt) ----
# Runs from the shipped 7,739-row detection panel. Not quoted in the paper.
# Values come from xgboost 3.3.0 and scikit-learn 1.9.0. The run is deterministic
# for fixed library versions, but other versions shift the third decimal.
ENSEMBLE = {  # (ROC, PR, Brier, ECE)
    "single":            (0.601, 0.167, 0.170, 0.186),
    "ensemble":          (0.604, 0.169, 0.152, 0.153),
    "ensemble+isotonic": (0.576, 0.150, 0.112, 0.031),
    "selective_roc_50pct": 0.588,
}

# ---- Romania replication of the label-free suite (Table III, Romania column) ----
# Same code and controls on the independent Romanian network, with no in-situ labels.
# 18,692-reach EU-Hydro graph. 18,459 reaches carry a >=12-month Sentinel-2 series
# over 23 months (2024-07 to 2026-05). 1,344 outlet basins.
# Inputs: turb_series_ro.csv, turbidity_field_ro.csv. Scripts: exp_temporal.py
# (+ --orient-match), exp_propagation.py, exp_anomaly.py, exp_basins.py.
# Logs: ro_cofluctuation.txt, ro_cofluctuation_orient.txt, ro_propagation.txt,
# ro_anomaly.txt, ro_basins.txt.
ROMANIA_REPL = {
    "n_reaches_graph": 18692, "n_reaches_series": 18459, "n_months": 23, "n_basins": 1344,
    # Sentinel-2 per-pass observations (Table I): row count of
    # data/ro/turb_perdate_ro_full.csv.gz (4,141,108).
    "s2_pass_obs_M": 4.1,

    # (signal, deseason): (c_flow, c_euc, diff, lo, hi, perm_p, n_pairs), as COFLUCTUATION
    "cofluctuation": {
        ("Turbidity", "common-mode"): (0.667, 0.621, 0.046, 0.029, 0.062, 0.003, 1413),
        ("Turbidity", "seasonal"):    (0.676, 0.647, 0.029, 0.011, 0.047, 0.029, 1413),
        ("NDCI", "common-mode"):      (0.622, 0.476, 0.146, 0.123, 0.169, 0.0005, 1413),
        ("NDCI", "seasonal"):         (0.613, 0.512, 0.101, 0.080, 0.121, 0.0005, 1413),
    },
    # bearing-matched (+/-45 deg) control, as COFLUCTUATION_ORIENT
    "cofluctuation_orient": {
        ("Turbidity", "common-mode"): (0.656, 0.585, 0.071, 0.048, 0.093, 0.001, 753),
        ("Turbidity", "seasonal"):    (0.670, 0.631, 0.039, 0.015, 0.062, 0.036, 753),
        ("NDCI", "common-mode"):      (0.582, 0.427, 0.155, 0.123, 0.186, 0.0005, 753),
        ("NDCI", "seasonal"):         (0.592, 0.494, 0.098, 0.070, 0.124, 0.0005, 753),
    },
    # Static field. Note the tuple order differs from PROPAGATION:
    # (r_flow, r_euc, r_reversed, diff, lo, hi)
    "propagation": {
        "Turbidity": (0.615, 0.625, 0.647, -0.010, -0.034, 0.015),
        "NDCI":      (0.732, 0.698, 0.705,  0.034,  0.011, 0.056),
    },
    # Synthetic injections, flow vs Euclidean. CI kinds as in ANOMALY.
    "anomaly": {
        "residual": {"flow": 0.997, "euc": 0.998, "vs_euclidean": (-0.001, -0.004, 0.002)},
        "break":    {"flow": 0.976, "euc": 0.974, "vs_euclidean": (0.002, -0.002, 0.006)},
        "recovery": {"flow": 0.168, "euc": 0.130, "delta": (0.038, 0.025, 0.052)},
        "denoise":  {"flow": 0.078, "euc": 0.060, "nrej_flow": 0.359, "nrej_euc": 0.127,
                     "delta": (0.018, 0.013, 0.023)},
        "tracing":  {"st_top1": 0.583, "static_top1": 0.567, "euc_top1": 0.083,
                     "reversed_top1": 0.000, "mrr_st": 0.714, "mrr_euc": 0.260},
    },
    # Basin accretion, largest first: (k, n_segments, diff, lo, hi), ro_basins.txt.
    # Paper run: 1,500 sources and 300 resamples per point. Every CI excludes 0.
    "basins": [
        (1, 4874, 0.060, 0.038, 0.080), (2, 6565, 0.029, 0.007, 0.048),
        (3, 8236, 0.068, 0.044, 0.090), (5, 10761, 0.055, 0.031, 0.077),
        (10, 13060, 0.048, 0.025, 0.068), (20, 14790, 0.041, 0.016, 0.068),
        (50, 15996, 0.041, 0.017, 0.064), (100, 16406, 0.057, 0.034, 0.082),
        (1344, 18459, 0.054, 0.030, 0.079),
    ],
}

# ---- Graph neural network vs no-edge MLP (gnn.txt, Table II) ----
# 2,597-node labelled 5-hop subgraph, two GAT layers. Means over 5 seeds with the
# folds fixed (weight init and shuffling vary). Only the GAT-MLP gap is meaningful.
GNN = {  # (ROC, PR) means
    "mlp_no_edges": (0.586, 0.161),
    "gat_directed": (0.576, 0.151),
    "mlp_roc_sd": 0.010, "gat_roc_sd": 0.021,
    "roc_delta_mean": -0.010, "roc_delta_sd": 0.019,   # GAT - MLP over 5 seeds
    "n_seeds": 5, "n_nodes": 2597, "n_cells": 8100, "prevalence": 0.123,
}

# ---- Per-pass event correspondence (scripts/run_events.py, events.txt) ----
# 164k passes over 714 segments, matched to 39,725 of the 45,873 deduplicated
# in-situ readings.
EVENTS = {
    "match_rate": {1: 0.40, 3: 0.67, 7: 0.87},        # by +/- day window
    "spearman_pm3d": {"Ammonium~NDTI": 0.34, "Dissolved oxygen~NDTI": 0.25,
                      "Nitrate~NDTI": 0.22, "Ammonium~turbidity": 0.21},
    # ROC-AUC of the per-pass index risk score (ingest.indices.risk_score) for
    # exceedance, by day window. Near chance.
    "riskscore_auc": {1: 0.521, 3: 0.516, 7: 0.511},
}

# ---- Time-aware source attribution (Sec. V-B, Fig. 4) ----------------------------
# scripts/exp_provenance.py -> provenance_{de,ro}.txt. Per-pass turbidity exceedances
# (p95), three-hop upstream neighbourhood, 30-day window, passes within +/-6 days
# count as simultaneous. Shares are over all classified exceedances (the Fig. 4
# caption says facility-adjacent; those shares differ by at most 2.4 points).
PROVENANCE = {
    # label -> (Germany %, Romania %)
    "classes": [
        ("concurrent with\nelevated upstream", 82, 78),
        ("fresh local spike", 8, 9),
        ("lagged upstream\narrival", 7, 7),
        ("gradual local\nbuild-up", 3, 6),
    ],
    # Flow-upstream vs distance-matched unconnected control, co-elevated at an
    # exceedance: (flow %, control %, delta pp). Values as printed in the paper.
    "co_elevated": {"Germany": (83.3, 85.6, -2.2), "Romania": (79.8, 78.0, +1.8)},
    # What the shipped data regenerates. Germany matches the paper. Romania is
    # 0.1-0.3 points off: the paper's run included January 2023, later dropped from the
    # Romanian data (see PAPER_NOTES.md). Reach-cluster bootstrap
    # CIs on the delta: Germany [-2.4, -2.1], Romania [+1.3, +2.6].
    "co_elevated_regenerated": {"Germany": (83.3, 85.6, -2.2), "Romania": (79.7, 77.7, +2.0)},
    # Share of facility-adjacent per-pass exceedances whose excess is mostly inherited
    # from upstream (local fraction < 0.5). scripts/exp_inherited.py, inherited_{de,ro}.txt.
    "inherited_share": {"Germany": 0.90, "Romania": 0.85},
    # Median own passes in the 30 days before an exceedance.
    "passes_per_window": {"Germany": 5, "Romania": 4},
}

# Facility proximity (scripts/facility_shares.py -> facility_shares.txt): share of
# reach centroids within 1.5 km of an OSM facility, and the share of German
# exceedance months on such reaches.
FACILITY_PROXIMITY = {
    "frac_reaches_within_1500m": {"Germany": 0.488, "Romania": 0.180},
    "frac_exceedance_months_within_1500m_de": 0.560,
}

# ---- NDCI co-fluctuation by source Strahler bin (run_ndci_strata.py, ndci_strahler.txt) ----
# Same protocol as CONDITIONAL["cofluct_strahler"] (common-mode, sample=2500, n_boot=2000).
# The NDCI advantage grows with river size and is largest on wide (5+) reaches,
# where mixed land/water pixels are rare. So it is not a narrow-reach mixed-pixel
# artefact (unlike turbidity, which is headwater-heavy).
NDCI_STRAHLER = {  # bin: (diff, lo, hi)
    "Headwater (1-2)": (0.040, 0.025, 0.056),
    "Mid (3-4)": (0.072, 0.051, 0.094),
    "Large (5+)": (0.128, 0.100, 0.156),
}
