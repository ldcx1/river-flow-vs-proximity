"""Controls for the temporal co-fluctuation experiment. Two parallel chains share
a seasonal cycle, and a pulse travels downstream along each. After removing the
common mode the experiment must find flow co-fluctuation above Euclidean and a
downstream lag. With independent noise it must find nothing."""

import numpy as np

from experiments.eval.temporal_transport import (
    common_mode_anomaly,
    seasonal_anomaly,
    temporal_transport,
)
from experiments.graph import RiverGraph


def _two_chains(L=12):
    """Chain A (lon 0) and chain B (~1.1 km east), each flow-connected internally
    but not to each other. A_k and B_k are geographically near."""
    nodes = {}
    for tag, lon in (("a", 0.0), ("b", 0.01)):
        for k in range(L):
            nodes[f"{tag}{k}"] = {
                "up": [f"{tag}{k-1}"] if k > 0 else [],
                "next_down": f"{tag}{k+1}" if k < L - 1 else None,
                "strahler": 1, "length_m": 2000.0,
                "lat": k * 0.02, "lon": lon, "river_id": tag, "river_name": tag,
            }
    return RiverGraph(nodes), L


def _ar1(T, phi, rng):
    r = np.zeros(T)
    for t in range(1, T):
        r[t] = phi * r[t - 1] + rng.standard_normal()
    return r


def _series(graph, L, advect, seed=0, T=40):
    """X[node, month] = shared season + (advected chain pulse or independent
    per-node AR(1)) + small noise."""
    rng = np.random.default_rng(seed)
    segs = sorted(graph.nodes)
    X = np.zeros((len(segs), T))
    season = 3.0 * np.sin(2 * np.pi * np.arange(T) / 12.0)
    rA, rB = _ar1(T, 0.85, rng), _ar1(T, 0.85, rng)
    for i, s in enumerate(segs):
        tag, k = s[0], int(s[1:])
        if advect:
            pulse = rA if tag == "a" else rB
            shifted = np.array([pulse[t - k] if t - k >= 0 else 0.0 for t in range(T)])
            X[i] = season + 2.0 * shifted + 0.2 * rng.standard_normal(T)
        else:
            X[i] = season + _ar1(T, 0.85, rng) + 0.2 * rng.standard_normal(T)
    return segs, X


def test_h1_h2_detect_advected_transport():
    g, L = _two_chains()
    segs, X = _series(g, L, advect=True, seed=1)
    anom = common_mode_anomaly(X)
    res = temporal_transport(g, segs, anom, name="syn", deseason="common-mode",
                             n_min=12, max_dist_km=5.0, sample=100,
                             long_km=1.0, max_lag=2, n_boot=500)
    assert res.diff > 0 and res.boot_lo > 0          # flow co-fluctuation > Euclidean
    assert res.perm_p < 0.05                          # beats the permutation null
    assert res.asym_long > 0                          # downstream lags upstream


def test_h1_null_on_independent_series():
    g, L = _two_chains()
    segs, X = _series(g, L, advect=False, seed=2)
    anom = common_mode_anomaly(X)
    res = temporal_transport(g, segs, anom, name="syn", deseason="common-mode",
                             n_min=12, max_dist_km=5.0, sample=100,
                             long_km=1.0, max_lag=2, n_boot=500)
    assert res.boot_lo <= 0                            # no real flow structure
    assert res.perm_p > 0.05                           # and the permutation null agrees


def _aniso_grid(N=14, T=40, rho=0.85, flow="EW", seed=0):
    """N x N reaches with a field that is AR(1)-smooth along the east-west axis,
    independent across rows, and has no flow coupling. ``flow='EW'`` runs flow
    along the smooth axis, ``'NS'`` across it."""
    rng = np.random.default_rng(seed)
    f = np.zeros((N, N, T))
    for r in range(N):
        prev = rng.standard_normal(T)
        f[r, 0] = prev
        for c in range(1, N):
            prev = rho * prev + np.sqrt(1 - rho ** 2) * rng.standard_normal(T)
            f[r, c] = prev
    nodes = {}
    for r in range(N):
        for c in range(N):
            if flow == "EW":
                nd = f"{r}_{c+1}" if c + 1 < N else None
                up = [f"{r}_{c-1}"] if c > 0 else []
            else:
                nd = f"{r+1}_{c}" if r + 1 < N else None
                up = [f"{r-1}_{c}"] if r > 0 else []
            nodes[f"{r}_{c}"] = {"up": up, "next_down": nd, "strahler": 1,
                                 "length_m": 5500.0, "lat": r * 0.05, "lon": c * 0.05,
                                 "river_id": "R", "river_name": "R"}
    segs = [f"{r}_{c}" for r in range(N) for c in range(N)]
    X = np.array([f[int(s.split("_")[0]), int(s.split("_")[1])] for s in segs])
    return RiverGraph(nodes), segs, X


def test_anisotropy_confound_documented():
    """Documents a known limit. The distance-matched control ignores orientation,
    so it cannot tell flow coupling from a field that is smoother along the river
    axis. With no flow coupling it reports a significant flow advantage when flow
    runs along the smooth axis, and the sign flips when flow runs across it."""
    g, segs, X = _aniso_grid(flow="EW", seed=0)
    leg = temporal_transport(g, segs, X, "aniso", "raw", n_min=12, n_boot=400, n_perm=400)
    assert leg.diff > 0.1 and leg.boot_lo > 0 and leg.perm_p < 0.05   # spurious positive
    g2, segs2, X2 = _aniso_grid(flow="NS", seed=0)
    ns = temporal_transport(g2, segs2, X2, "aniso", "raw", n_min=12, n_boot=400, n_perm=400)
    assert ns.diff < 0                                                # sign tracks geometry
    # The bearing-matched control drops controls with a different bearing. On this
    # axis-aligned grid no unconnected control shares the flow bearing, so it forms
    # no pairs and reports nothing. It is a partial mitigation only: the evidence on
    # real data is that the effect survives this control.
    fix = temporal_transport(g, segs, X, "aniso", "raw", n_min=12, n_boot=400,
                             n_perm=400, orient_match=True)
    assert fix.n_pairs < leg.n_pairs               # fewer pairs (0 here)
    assert not fix.significant                     # no spurious flow advantage


def test_seasonal_anomaly_removes_per_segment_climatology():
    # Each segment has its own month-of-year cycle, which common-mode removal would
    # miss. seasonal_anomaly must strip each segment's monthly climatology.
    months = [f"20{y:02d}-{m:02d}" for y in (20, 21) for m in range(1, 13)]
    mo = np.array([int(m[5:7]) for m in months])
    X = np.vstack([mo.astype(float),            # seg0 climatology = month number
                   -2.0 * mo.astype(float)])    # seg1 = a different climatology
    A = seasonal_anomaly(X, months)
    assert np.allclose(A, 0.0, atol=1e-9)        # pure climatology fully removed
    # with noise, each (segment, calendar-month) group is still de-meaned to ~0
    rng = np.random.default_rng(0)
    An = seasonal_anomaly(X + 0.1 * rng.standard_normal(X.shape), months)
    for s in range(2):
        for m in range(1, 13):
            cols = np.where(mo == m)[0]
            assert abs(An[s, cols].mean()) < 1e-9
