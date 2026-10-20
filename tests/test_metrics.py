"""Metric implementations used for every reported number, pinned to hand-computed
values and cross-checked against scikit-learn and scipy."""

import numpy as np
import pytest

from experiments import metrics


# -- hand-computed anchors (no sklearn needed) ------------------------------
def test_roc_auc_perfect_and_inverted():
    y = np.array([0, 0, 1, 1])
    assert metrics.roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert metrics.roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_roc_auc_all_ties_is_half():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    assert metrics.roc_auc(y, p) == pytest.approx(0.5)


def test_roc_auc_single_class_is_nan():
    assert np.isnan(metrics.roc_auc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])))


def test_average_precision_hand_value():
    # descending: (y1,.9)(y0,.8)(y1,.7)(y0,.6) -> AP = (1/1 + 2/3) / 2
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.7, 0.6])
    assert metrics.average_precision(y, p) == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)


def test_average_precision_no_positives_is_zero():
    assert metrics.average_precision(np.zeros(5), np.random.rand(5)) == 0.0


def test_brier_and_ece_perfect():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    assert metrics.brier_score(y, y) == 0.0
    assert metrics.expected_calibration_error(y, y, n_bins=10) == 0.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        metrics.roc_auc(np.array([0, 1]), np.array([0.1, 0.2, 0.3]))


# -- cross-check against scikit-learn on random data ------------------------
def test_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = rng.integers(30, 200)
        y = (rng.random(n) < rng.uniform(0.1, 0.6)).astype(float)
        if y.sum() == 0 or y.sum() == n:
            continue
        p = rng.random(n)  # tie-free
        assert metrics.roc_auc(y, p) == pytest.approx(sk.roc_auc_score(y, p), abs=1e-9)
        assert metrics.average_precision(y, p) == pytest.approx(
            sk.average_precision_score(y, p), abs=1e-9)
        assert metrics.brier_score(y, p) == pytest.approx(
            sk.brier_score_loss(y, p), abs=1e-9)


def test_average_precision_tie_correct_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(3)
    # heavily tied scores: AP must still match sklearn
    y = (rng.random(300) < 0.3).astype(float)
    p = rng.integers(0, 4, size=300).astype(float)
    assert metrics.average_precision(y, p) == pytest.approx(
        sk.average_precision_score(y, p), abs=1e-9)


def test_roc_auc_tie_aware_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    # coarse scores give many ties, and tie-aware ranking must still match sklearn
    y = (rng.random(300) < 0.3).astype(float)
    p = rng.integers(0, 5, size=300).astype(float)
    assert metrics.roc_auc(y, p) == pytest.approx(sk.roc_auc_score(y, p), abs=1e-9)


# -- spearman_r ---------------------------------------------------------------
def test_spearman_r_perfect_and_inverted():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert metrics.spearman_r(a, a) == pytest.approx(1.0)
    assert metrics.spearman_r(a, a[::-1]) == pytest.approx(-1.0)


def test_spearman_r_tie_correct_hand_value():
    # b ties on the first two: average ranks [1.5, 1.5, 3, 4], rho = 4.5 / sqrt(22.5)
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([10.0, 10.0, 30.0, 40.0])
    assert metrics.spearman_r(a, b) == pytest.approx(4.5 / np.sqrt(22.5), abs=1e-9)


def test_spearman_r_edge_cases():
    assert np.isnan(metrics.spearman_r(np.array([1.0, 2.0]), np.array([1.0, 2.0])))       # <3 points
    assert np.isnan(metrics.spearman_r(np.array([1.0, 1.0, 1.0]),
                                       np.array([1.0, 2.0, 3.0])))                          # zero variance


def test_spearman_r_independent_near_zero():
    # negative control: two independent random vectors -> rank correlation near 0
    rng = np.random.default_rng(123)
    a = rng.standard_normal(2000)
    b = rng.standard_normal(2000)
    assert abs(metrics.spearman_r(a, b)) < 0.1


def test_spearman_r_tie_correct_matches_scipy():
    sp = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(7)
    a = rng.integers(0, 4, size=200).astype(float)   # heavily tied
    b = rng.integers(0, 5, size=200).astype(float)
    assert metrics.spearman_r(a, b) == pytest.approx(sp.spearmanr(a, b)[0], abs=1e-9)
