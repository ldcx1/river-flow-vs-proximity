"""Positive and negative controls for the experiment runners: Moran's I
autocorrelation, partial Moran's I and variograms, the synthetic anomaly runners
(residual, break, recovery, denoise, tracing) and source attribution."""

import numpy as np

from experiments.graph import RiverGraph


# --- toy graphs ------------------------------------------------------------
def two_parallel_chains(L=10):
    """Chain A (lon 0) and chain B (~1.5 km east), flow-connected within each
    chain but not across. A_k and B_k are adjacent, so geography and flow disagree."""
    nodes = {}
    for tag, lon in (("a", 0.0), ("b", 0.015)):
        for k in range(L):
            nodes[f"{tag}{k}"] = {
                "up": [f"{tag}{k-1}"] if k > 0 else [],
                "next_down": f"{tag}{k+1}" if k < L - 1 else None,
                "strahler": 1, "length_m": 2000.0,
                "lat": k * 0.02, "lon": lon, "river_id": tag, "river_name": tag,
            }
    return RiverGraph(nodes), L


# --- Moran's I autocorrelation ---------------------------------------------
def test_morans_autocorr_detects_and_rejects():
    from experiments.eval import morans_autocorr
    g, L = two_parallel_chains(12)
    # opposite gradients per chain: flow neighbours similar, cross-chain neighbours not
    vals = {f"{t}{k}": (k if t == "a" else (L - k)) * 1.0 for t in "ab" for k in range(L)}
    pos = morans_autocorr(g, vals, max_dist_km=5.0, n_perm=500)
    assert pos.I_flow > 0.3 and pos.p_flow < 0.05      # flow autocorrelation detected
    rng = np.random.default_rng(0)
    noise = {s: float(rng.standard_normal()) for s in g.nodes}
    neg = morans_autocorr(g, noise, max_dist_km=5.0, n_perm=500)
    assert neg.p_flow > 0.05                            # noise -> not significant


# --- partial Moran's I and variograms --------------------------------------
def test_partial_morans_beyond_covariates():
    from experiments.eval import partial_morans
    g, L = two_parallel_chains(12)
    vals = {f"{t}{k}": (k if t == "a" else (L - k)) * 1.0 for t in "ab" for k in range(L)}
    # covariates are ~constant here, so the flow gradient survives
    # residualisation and flow should beat Euclidean
    res = partial_morans(g, vals, None, max_dist_km=5.0, n_perm=500)
    assert res.diff > 0 and res.p_diff < 0.05
    rng = np.random.default_rng(1)
    noise = {s: float(rng.standard_normal()) for s in g.nodes}
    neg = partial_morans(g, noise, None, max_dist_km=5.0, n_perm=500)
    assert neg.p_diff > 0.05


def test_variograms_stream_below_euclidean_when_flow_structured():
    from experiments.eval import variograms
    g, L = two_parallel_chains(12)
    vals = {f"{t}{k}": (k if t == "a" else (L - k)) * 1.0 for t in "ab" for k in range(L)}
    centers, gf, ge = variograms(g, vals, max_dist_km=30.0, n_bins=4, sample=100)
    # short-range stream pairs are more similar (lower semivariance) than short-range
    # geographic pairs (which cross the two anti-correlated chains)
    assert np.nanmean(gf[:2]) < np.nanmean(ge[:2])


# --- recovery, denoise, tracing and attribution runners --------------------
def _field_graph(L=40):
    """Two parallel chains, enough segments for the anomaly injection harness
    (the top-q% threshold needs a non-trivial population)."""
    return two_parallel_chains(L)


# The flow-vs-Euclidean win is a property of the real data and does not reproduce on
# two straight chains. The component mechanism is tested in test_anomaly.py. These
# tests check that the runners detect planted structure and return valid CIs.
def test_run_a3_detects_planted_events():
    from experiments.eval import run_recovery
    g, L = _field_graph()
    segs = sorted(g.nodes)
    rng = np.random.default_rng(0)
    X = np.abs(rng.standard_normal((len(segs), 24))) * 0.1     # quiet background
    res = run_recovery(g, segs, X, n_runs=8, n_plumes=3, n_geo=3, plume_len=5, q=85.0, seed=0)
    assert res.n_runs > 0
    assert res.flow["rec"] > 0 and 0 <= res.flow["f1"] <= 1     # finds injected plumes
    assert res.f1_delta["lo"] <= res.f1_delta["mean"] <= res.f1_delta["hi"]  # valid CI


def test_run_a4_filters_and_detects():
    from experiments.eval import run_denoise
    g, L = _field_graph()
    segs = sorted(g.nodes)
    rng = np.random.default_rng(1)
    X = np.abs(rng.standard_normal((len(segs), 24))) * 0.1
    res = run_denoise(g, segs, X, n_runs=8, n_plumes=3, n_geo=3, plume_len=5, q=85.0, seed=0)
    assert res.n_runs > 0
    assert res.raw["rec"] > 0 and 0 <= res.flow["prec"] <= 1
    assert res.prec_delta["lo"] <= res.prec_delta["mean"] <= res.prec_delta["hi"]


def test_run_a5_recovers_source_above_chance():
    from experiments.eval import run_tracing
    g, L = two_parallel_chains(20)
    segs = sorted(g.nodes)
    rng = np.random.default_rng(2)
    X = np.abs(rng.standard_normal((len(segs), 24))) * 0.1
    res = run_tracing(g, segs, X, n_events=20, plume_len=6, seed=0)
    assert res.n_events > 0
    assert res.mrr["static"] > 0.1                             # graph trace >> chance (~1/30)
    assert res.mrr["spatio_temporal"] >= res.mrr["static"] - 1e-9


# --- residual and break runners ---------------------------------------------
# The contrast each runner reports is a mechanism, not a data-dependent win, so it
# reproduces on the toy.
def test_run_a1_separates_local_from_inherited():
    from experiments.eval import run_residual
    g, L = _field_graph()
    segs = sorted(g.nodes)
    rng = np.random.default_rng(3)
    X = np.abs(rng.standard_normal((len(segs), 24))) * 0.1
    res, contrast = run_residual(g, segs, X, n_local=40, n_inherited=40, n_bg=200, mag=5.0, seed=0)
    assert res.n_pos > 0 and res.n_neg > 0
    assert set(contrast) == {"graph_expectation", "node_temporal", "spatial_euclidean"}
    # the graph-expectation residual flags a local spike, while an inherited-high
    # reach is explained by its upstream expectation
    assert contrast["graph_expectation"] >= 0.6
    # and it must beat the graph-blind node-temporal baseline on that contrast
    assert contrast["graph_expectation"] > contrast["node_temporal"]


def test_run_a2_runs_and_decoupled_decay_path_works():
    from experiments.eval import run_break
    g, L = _field_graph()
    segs = sorted(g.nodes)
    rng = np.random.default_rng(4)
    X = np.abs(rng.standard_normal((len(segs), 24))) * 0.1
    res, contrast = run_break(g, segs, X, n_viol=40, n_sources=20, n_bg=200, mag=5.0, seed=0)
    assert res.n_pos > 0 and res.n_neg > 0
    assert set(contrast) == {"graph_neighbors", "node_temporal", "spatial_euclidean"}
    assert np.isfinite(contrast["graph_neighbors"])
    # the directed-neighbour residual must beat the graph-blind node-temporal baseline
    assert contrast["graph_neighbors"] > contrast["node_temporal"]
    # decoupled injection decay (robustness run) must also work
    res2, contrast2 = run_break(g, segs, X, n_viol=40, n_sources=20, n_bg=200, mag=5.0,
                             inject_decay_km=15.0, seed=0)
    assert res2.n_pos > 0 and np.isfinite(contrast2["graph_neighbors"])
    assert contrast2["graph_neighbors"] > contrast2["node_temporal"]


def test_run_attribution_upstream_beats_reversed():
    from experiments.eval import run_attribution
    from experiments.osm import FacilityIndex
    g, L = _field_graph()
    segs = sorted(g.nodes)
    idx = {s: i for i, s in enumerate(segs)}
    # one month: a high source upstream in chain A that propagates downstream
    field = {"m": {}}
    for s in segs:
        field["m"][s] = 0.0
    src = "a3"
    field["m"][src] = 6.0
    for d, rec in enumerate(g.walk_downstream(src, 6)):
        field["m"][rec["object_id"]] = 6.0 * np.exp(-(d + 1) * 0.3)
    detect = g.walk_downstream(src, 6)[-1]["object_id"]
    targets = {"m": [detect]}
    # facility at the true source's location
    fac = FacilityIndex([{"lat": g[src]["lat"], "lon": g[src]["lon"], "kind": "x"}])
    res = run_attribution(g, field, targets, fac, radius_m=3000.0, top_k=20)
    assert res.n_targets > 0
    # graph upstream tracing should align with the (upstream) facility better than
    # the reversed (downstream) control
    assert res.mrr["graph_upstream"] >= res.mrr["reversed"]
    # the reversed control must return downstream candidates (it ranks by
    # concentration, since injected_load is empty downstream of the source)
    assert res.n_nonempty["reversed"] > 0


# --- negative controls: no planted structure -> no manufactured win ----------
def test_run_a3_no_detection_without_injection():
    from experiments.eval import run_recovery
    g, L = _field_graph()
    segs = sorted(g.nodes)
    rng = np.random.default_rng(0)
    X = np.abs(rng.standard_normal((len(segs), 24))) * 0.1
    res = run_recovery(g, segs, X, n_runs=8, n_plumes=0, n_geo=3, plume_len=5, q=85.0, seed=0)
    assert res.n_runs == 0          # no injected events -> nothing to recover, no spurious F1 win


def test_run_a4_no_detection_without_injection():
    from experiments.eval import run_denoise
    g, L = _field_graph()
    segs = sorted(g.nodes)
    rng = np.random.default_rng(1)
    X = np.abs(rng.standard_normal((len(segs), 24))) * 0.1
    res = run_denoise(g, segs, X, n_runs=8, n_plumes=0, n_geo=3, plume_len=5, q=85.0, seed=0)
    assert res.n_runs == 0


def test_run_attribution_no_win_without_aligned_pressure():
    from experiments.eval import run_attribution
    from experiments.osm import FacilityIndex
    g, L = _field_graph()
    segs = sorted(g.nodes)
    field = {"m": {s: 0.0 for s in segs}}
    src = "a3"
    field["m"][src] = 6.0
    for d, rec in enumerate(g.walk_downstream(src, 6)):
        field["m"][rec["object_id"]] = 6.0 * np.exp(-(d + 1) * 0.3)
    detect = g.walk_downstream(src, 6)[-1]["object_id"]
    targets = {"m": [detect]}
    # facility far from every reach: the graph must not beat reversed
    fac = FacilityIndex([{"lat": 0.0, "lon": 40.0, "kind": "x"}])
    res = run_attribution(g, field, targets, fac, radius_m=500.0, top_k=20)
    assert res.boot_delta["reversed"]["lo"] <= 0


def test_run_a5_no_event_when_chain_too_short():
    """plume_len=1 gives a downstream chain under 3 reaches, so no event is placed
    and MRR is nan rather than a spurious value."""
    from experiments.eval import run_tracing
    g, L = two_parallel_chains(20)
    segs = sorted(g.nodes)
    rng = np.random.default_rng(7)
    X = np.abs(rng.standard_normal((len(segs), 24))) * 0.1
    res = run_tracing(g, segs, X, n_events=20, plume_len=1, seed=0)
    assert res.n_events == 0
    assert not np.isfinite(res.mrr["static"])      # nan, not a spurious >0


def test_run_a2_no_win_without_signal():
    """mag=0: positives are labelled but carry no spike, so the directed-neighbour
    residual must not clearly beat the node-temporal baseline. The residual runner
    has no such control because its label is topological by construction."""
    from experiments.eval import run_break
    g, L = _field_graph()
    segs = sorted(g.nodes)
    rng = np.random.default_rng(9)
    X = np.abs(rng.standard_normal((len(segs), 24))) * 0.1
    res, contrast = run_break(g, segs, X, n_viol=30, n_sources=30, n_bg=200, mag=0.0, seed=0)
    # null gap stays under +0.02 across seeds, so +0.05 allows for noise
    assert contrast["graph_neighbors"] <= contrast["node_temporal"] + 0.05


def test_flow_neighbors_excludes_self():
    """A reach must never appear in its own flow-neighbour set. Self-inclusion
    would leak a reach's own value into its neighbour statistics."""
    from experiments.eval.graph_structure import flow_neighbors
    g, L = two_parallel_chains(6)
    segs = sorted(g.nodes)
    nb = flow_neighbors(g, segs, max_dist_km=50.0)
    assert all(s not in nb[s] for s in segs)
