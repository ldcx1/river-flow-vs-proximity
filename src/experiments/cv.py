"""Segment-grouped cross-validation.

A segment is observed in many correlated months. Training on its January row
and scoring its July row would measure memorisation, so folds are built over
whole segments (``object_id``).

``grouped_folds`` is a greedy GroupKFold. Groups are sorted by descending
positive rate and each goes to the fold with the fewest rows so far, which keeps
fold sizes and class balance even. The seed only breaks ties between groups with
equal positive rate, so different seeds give different splits.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def grouped_folds(groups, y, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Assign each row a fold index in ``[0, n_splits)`` such that all rows
    sharing a ``group`` value land in the same fold. Balanced greedily on group
    size and positive rate.

    Parameters
    ----------
    groups : array-like of group labels (e.g. segment object_id), one per row.
    y      : 0/1 labels, one per row (used only to balance positive rate).

    With fewer groups than ``n_splits``, or very skewed group sizes, some folds
    can be empty. ``cross_val_oof`` and ``nested_oof_predict`` skip empty folds,
    which lowers the effective fold count.
    """
    groups = np.asarray(groups)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)

    uniq = np.array(sorted(set(groups.tolist())))
    rng.shuffle(uniq)                       # only matters for positive-rate ties
    gpos = {g: y[groups == g].mean() for g in uniq}
    gsize = {g: int((groups == g).sum()) for g in uniq}
    order = sorted(uniq.tolist(), key=lambda g: -gpos[g])  # stable: keeps shuffle within ties

    assign: dict = {}
    load = [0] * n_splits
    for g in order:
        f = int(np.argmin(load))
        assign[g] = f
        load[f] += gsize[g]
    return np.array([assign[g] for g in groups.tolist()])


def cross_val_oof(
    X: np.ndarray,
    y: np.ndarray,
    fold: np.ndarray,
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    n_splits: int = 5,
) -> np.ndarray:
    """Out-of-fold predictions, one per input row. For each fold ``k``,
    ``fit_predict(X_train, y_train, X_val)`` trains on the other folds and
    returns positive-class probabilities for the held-out rows.
    """
    y = np.asarray(y, dtype=float)
    oof = np.zeros(len(y))
    for k in range(n_splits):
        val = np.where(fold == k)[0]
        tr = np.where(fold != k)[0]
        if len(val) == 0:
            continue
        oof[val] = fit_predict(X[tr], y[tr], X[val])
    return oof


def repeated_grouped_oof(
    X: np.ndarray,
    y: np.ndarray,
    groups,
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Repeated segment-grouped out-of-fold predictions. Repeat ``r`` uses folds
    built with seed ``seed + r``. Returns shape ``(n_repeats, n_rows)``, one
    out-of-fold prediction vector per repeat."""
    y = np.asarray(y, dtype=float)
    out = np.empty((n_repeats, len(y)))
    for r in range(n_repeats):
        fold = grouped_folds(groups, y, n_splits, seed=seed + r)
        out[r] = cross_val_oof(X, y, fold, fit_predict, n_splits)
    return out


def standardize_in_fold(X_tr: np.ndarray, X_va: np.ndarray):
    """Standardize with train-fold statistics only, so nothing leaks from the
    validation rows. NaN cells get the train-column mean, then columns are
    centred and scaled by the train mean and std (std=1 for zero-variance
    columns). Returns ``(X_tr_std, X_va_std)``.
    """
    X_tr = np.asarray(X_tr, dtype=float)
    X_va = np.asarray(X_va, dtype=float)
    mu = np.nanmean(X_tr, axis=0) if X_tr.size else np.zeros(X_tr.shape[1])
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.nanstd(X_tr, axis=0) if X_tr.size else np.ones(X_tr.shape[1])
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)

    def _apply(X):
        Xf = np.where(np.isfinite(X), X, mu)
        return (Xf - mu) / sd

    return _apply(X_tr), _apply(X_va)
