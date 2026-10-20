"""Primitives of the conditional analyses: covariate residualisation, Strahler
binning, monotonic trend, and incremental AUC (a real graph signal is recovered,
a noise column is not). Partial Moran's I and variograms are tested in
test_experiment_runners.py."""

import numpy as np
import pytest

from experiments.eval.conditional import (
    BIN_ORDER,
    incremental_auc,
    monotonic_trend,
    residualize,
    strahler_bin,
)


def test_residualize_removes_covariate_variance():
    rng = np.random.default_rng(0)
    c = rng.standard_normal(200)
    y = 3.0 * c + 0.01 * rng.standard_normal(200)
    r = residualize(y, c[:, None])
    assert abs(np.corrcoef(r, c)[0, 1]) < 0.1       # covariate explained away


def test_strahler_bins():
    assert strahler_bin(1) == BIN_ORDER[0]
    assert strahler_bin(4) == BIN_ORDER[1]
    assert strahler_bin(7) == BIN_ORDER[2]


def test_monotonic_trend_sign():
    up = monotonic_trend(BIN_ORDER, [0.1, 0.2, 0.3])
    dn = monotonic_trend(BIN_ORDER, [0.3, 0.2, 0.1])
    assert up["rho"] > 0.9 and up["slope"] > 0
    assert dn["rho"] < -0.9 and dn["slope"] < 0


def test_incremental_auc_recovers_graph_signal():
    pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(1)
    n, g = 600, 60
    groups = np.repeat(np.arange(g), n // g)
    noise = rng.standard_normal(n)
    graph = rng.standard_normal(n)
    # label depends on the graph feature, not the baseline noise
    p = 1 / (1 + np.exp(-(1.5 * graph)))
    y = (rng.random(n) < p).astype(float)
    res = incremental_auc(y, [noise], graph, groups, n_boot=300)
    assert res["gain"] > 0.05 and res["lo"] > 0      # graph adds real detection value


def test_incremental_auc_no_gain_from_noise():
    pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(2)
    n, g = 600, 60
    groups = np.repeat(np.arange(g), n // g)
    base = rng.standard_normal(n)
    p = 1 / (1 + np.exp(-(1.5 * base)))
    y = (rng.random(n) < p).astype(float)
    noise_graph = rng.standard_normal(n)             # graph col is pure noise
    res = incremental_auc(y, [base], noise_graph, groups, n_boot=300)
    assert res["lo"] <= 0                            # no real incremental value
