"""Classification metrics shared by every experiment.

Each metric has one implementation, tested against scikit-learn in
``tests/test_metrics.py``. Functions take ``y`` (0/1 ground truth) and ``p``
(scores, higher means more positive) as 1-D array-likes and return a float.
Only numpy is required.
"""

from __future__ import annotations

import numpy as np


def _as_arrays(y, p):
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError(f"y and p length mismatch: {y.shape} vs {p.shape}")
    return y, p


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties given their average rank, like
    ``scipy.stats.rankdata(method='average')``."""
    n = x.shape[0]
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ordinal = np.arange(1, n + 1, dtype=float)
    is_start = np.empty(n, dtype=bool)
    is_start[0] = True
    np.not_equal(xs[1:], xs[:-1], out=is_start[1:])
    group = np.cumsum(is_start) - 1
    sums = np.bincount(group, weights=ordinal)
    cnts = np.bincount(group).astype(float)
    avg = (sums / cnts)[group]
    ranks = np.empty(n, dtype=float)
    ranks[order] = avg
    return ranks


def roc_auc(y, p) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (tie-aware).
    Returns NaN if either class is absent."""
    y, p = _as_arrays(y, p)
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = _average_ranks(p)
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def spearman_r(a, b) -> float:
    """Spearman rank correlation (tie-aware) between two paired 1-D arrays.
    Returns NaN for fewer than 3 points or zero variance."""
    a, b = _as_arrays(a, b)
    if len(a) < 3:
        return float("nan")
    ra, rb = _average_ranks(a), _average_ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra @ rb) / denom) if denom > 0 else float("nan")


def average_precision(y, p) -> float:
    """Average precision = sum over thresholds of (recall_n - recall_{n-1}) *
    precision_n. Tied scores are collapsed to one threshold, so the result
    matches ``sklearn.average_precision_score`` even with ties (e.g. discrete
    scores or imputed anomaly residuals)."""
    y, p = _as_arrays(y, p)
    npos = y.sum()
    if npos == 0:
        return 0.0
    order = np.argsort(-p, kind="mergesort")
    ys, ps = y[order], p[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1.0 - ys)
    # last index of each distinct-score group (collapse ties to one threshold)
    boundary = np.r_[np.nonzero(np.diff(ps))[0], len(ys) - 1]
    prec = tp[boundary] / (tp[boundary] + fp[boundary])
    rec = tp[boundary] / npos
    rec_prev = np.r_[0.0, rec[:-1]]
    return float(np.sum((rec - rec_prev) * prec))


def brier_score(y, p) -> float:
    """Mean squared error between predicted probability and outcome."""
    y, p = _as_arrays(y, p)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """Expected Calibration Error: weighted mean gap between confidence and
    accuracy over equal-width probability bins. 0 = perfectly calibrated."""
    y, p = _as_arrays(y, p)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize puts p==1.0 in an extra bin; clamp to the last bin.
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    n = len(p)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf = p[m].mean()
        acc = y[m].mean()
        ece += (m.sum() / n) * abs(conf - acc)
    return float(ece)


def prevalence(y) -> float:
    """Positive rate, the PR-AUC of a constant predictor."""
    y = np.asarray(y, dtype=float).ravel()
    return float(y.mean()) if len(y) else float("nan")
