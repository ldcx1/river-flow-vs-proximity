"""Segment-grouped CV: no segment may appear in both train and test. Also checks
fold determinism, which the repeated-CV results rely on."""

import numpy as np

from experiments.cv import (
    cross_val_oof,
    grouped_folds,
    repeated_grouped_oof,
    standardize_in_fold,
)


def _make(n_groups=40, rows_per_group=6, seed=0):
    rng = np.random.default_rng(seed)
    groups, y = [], []
    for g in range(n_groups):
        pos_rate = rng.uniform(0, 1)
        for _ in range(rows_per_group):
            groups.append(f"seg{g}")
            y.append(float(rng.random() < pos_rate))
    return np.array(groups), np.array(y, dtype=float)


def test_no_group_straddles_folds():
    groups, y = _make()
    fold = grouped_folds(groups, y, n_splits=5, seed=0)
    for g in set(groups.tolist()):
        folds_for_g = set(fold[groups == g].tolist())
        assert len(folds_for_g) == 1, f"group {g} split across folds {folds_for_g}"


def test_all_folds_used_and_in_range():
    groups, y = _make()
    fold = grouped_folds(groups, y, n_splits=5, seed=0)
    assert set(fold.tolist()) == {0, 1, 2, 3, 4}
    assert fold.min() >= 0 and fold.max() < 5


def test_deterministic_for_same_seed():
    groups, y = _make()
    a = grouped_folds(groups, y, 5, seed=3)
    b = grouped_folds(groups, y, 5, seed=3)
    assert np.array_equal(a, b)


def test_folds_roughly_balanced_in_size():
    groups, y = _make(n_groups=50, rows_per_group=6)
    fold = grouped_folds(groups, y, 5, seed=0)
    sizes = np.array([(fold == k).sum() for k in range(5)])
    # greedy size balancing should keep folds within a small spread
    assert sizes.max() - sizes.min() <= 2 * 6


def test_cross_val_oof_predicts_every_row_once_from_unseen_data():
    groups, y = _make()
    X = np.arange(len(y), dtype=float).reshape(-1, 1)   # column 0 = row id
    fold = grouped_folds(groups, y, 5, seed=0)
    calls = np.zeros(len(y), dtype=int)

    def fit_predict(X_tr, y_tr, X_va):
        tr, va = X_tr[:, 0].astype(int), X_va[:, 0].astype(int)
        assert not set(tr) & set(va)                    # never trains on a validated row
        assert not set(groups[tr]) & set(groups[va])    # nor on its group
        calls[va] += 1
        return X_va[:, 0]                               # echo the row id

    oof = cross_val_oof(X, y, fold, fit_predict, n_splits=5)
    assert (calls == 1).all()                           # each row predicted exactly once
    assert np.array_equal(oof, X[:, 0])                 # and written back to its own slot


def test_repeated_grouped_oof_shape_and_distinct_repeats():
    groups, y = _make()
    X = np.random.default_rng(0).random((len(y), 3))

    def fit_predict(X_tr, y_tr, X_va):
        # depend on the train fold so that different splits give different
        # predictions
        return np.full(len(X_va), float(y_tr.mean()))

    out = repeated_grouped_oof(X, y, groups, fit_predict, n_splits=5, n_repeats=4, seed=0)
    assert out.shape == (4, len(y))
    # repeats use different fold assignments, so predictions differ
    assert not np.allclose(out[0], out[1])


def test_repeated_grouped_oof_no_group_straddles_any_repeat():
    # each repeat's fold assignment (seed + r) must keep every group whole
    groups, y = _make()
    g = np.asarray(groups)
    for r in range(5):
        fold = grouped_folds(groups, y, n_splits=5, seed=0 + r)
        for gid in set(groups):
            assert len(set(fold[g == gid].tolist())) == 1


def test_repeated_grouped_oof_each_repeat_covers_every_row():
    groups, y = _make()
    X = np.random.default_rng(1).random((len(y), 2))

    def fit_predict(X_tr, y_tr, X_va):
        return np.full(len(X_va), 0.3)

    out = repeated_grouped_oof(X, y, groups, fit_predict, n_splits=5, n_repeats=3, seed=7)
    # every cell is filled by an out-of-fold prediction, none left at 0
    assert np.allclose(out, 0.3)


def test_standardize_in_fold_uses_train_stats_only():
    # a validation-only outlier must not move the fitted mean or scale
    X_tr = np.array([[1.0], [1.0], [1.0], [1.0]])
    X_va = np.array([[1.0], [101.0]])           # 101 is a validation-only outlier
    Xtr_s, Xva_s = standardize_in_fold(X_tr, X_va)
    # train column is constant -> std clamped to 1, mean 1 -> train standardizes to 0
    assert np.allclose(Xtr_s, 0.0)
    # the validation outlier is centred by the train mean (1.0)
    assert np.isclose(Xva_s[0, 0], 0.0)
    assert np.isclose(Xva_s[1, 0], 100.0)


def test_standardize_in_fold_imputes_nan_with_train_mean():
    X_tr = np.array([[0.0], [2.0], [4.0]])      # train mean 2.0
    X_va = np.array([[np.nan]])
    _, Xva_s = standardize_in_fold(X_tr, X_va)
    # NaN imputed with train mean -> standardizes to 0
    assert np.isclose(Xva_s[0, 0], 0.0)
