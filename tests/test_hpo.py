"""Nested-CV hyperparameter search and repeated CV, using the numpy logistic
model. Checks plumbing, determinism, and that tuning only sees outer-train rows."""

import numpy as np

from experiments.eval import repeated_cv, repeated_paired
from experiments.io.panels import Panel
from experiments.models import nested_oof_predict


def _panel(n_groups=30, rows_per_group=4, seed=0):
    rng = np.random.default_rng(seed)
    X, y, groups = [], [], []
    for g in range(n_groups):
        sig = rng.random()
        for _ in range(rows_per_group):
            X.append([rng.random(), sig + 0.05 * rng.standard_normal()])
            y.append(float(rng.random() < 0.2 + 0.6 * sig))
            groups.append(f"seg{g}")
    return Panel(np.array(X), np.array(y, float), np.array(groups),
                 ["o_noise", "f_ag_sig"], [])


LOGREG_GRID = [{"l2": 1e-3}, {"l2": 1e-1}, {"l2": 1.0}]


def test_nested_oof_shape_and_chosen_params():
    p = _panel()
    oof, chosen = nested_oof_predict(p.X, p.y, p.groups, param_grid=LOGREG_GRID,
                                     outer_splits=5, inner_splits=3, seed=0,
                                     model="logreg")
    assert oof.shape == (len(p.y),)
    assert len(chosen) == 5                      # one selection per outer fold
    assert all(c in LOGREG_GRID for c in chosen)


def test_nested_oof_deterministic():
    p = _panel()
    a, _ = nested_oof_predict(p.X, p.y, p.groups, param_grid=LOGREG_GRID,
                              seed=1, model="logreg")
    b, _ = nested_oof_predict(p.X, p.y, p.groups, param_grid=LOGREG_GRID,
                              seed=1, model="logreg")
    assert np.allclose(a, b)


def test_repeated_cv_runs_all_repeats():
    p = _panel()
    idx = p.select_prefix(["o_", "f_ag_"])
    res = repeated_cv(p, idx, name="all", n_repeats=4, model="logreg")
    assert len(res.pr) == 4 and len(res.roc) == 4


def test_repeated_paired_is_paired_and_complete():
    p = _panel()
    a = p.select_prefix(["o_"])
    b = p.select_prefix(["o_", "f_ag_"])
    res = repeated_paired(p, a, b, "opt", "opt+ag", n_repeats=5, model="logreg")
    assert len(res.deltas) == 5
    assert 0 <= res.wins <= 5
    # deltas equal b_pr - a_pr elementwise (true pairing)
    assert np.allclose(np.array(res.deltas), np.array(res.b_pr) - np.array(res.a_pr))


def test_logreg_standardizes_on_training_rows_only():
    """Imputation, standardisation and fitted weights must depend only on the
    training rows. Appending an extreme validation row must leave the predictions
    on the other validation rows unchanged."""
    from experiments.models.train import logreg_fit_predict
    rng = np.random.default_rng(0)
    X_tr = rng.standard_normal((40, 3))
    y_tr = (rng.random(40) < 0.5).astype(float)
    X_va = rng.standard_normal((6, 3))
    fp = logreg_fit_predict(spw=1.0, seed=0)
    p_small = fp(X_tr, y_tr, X_va)
    p_big = fp(X_tr, y_tr, np.vstack([X_va, X_va[:1] * 1e6]))   # extreme extra VAL row
    assert np.allclose(p_small, p_big[: len(X_va)], atol=1e-9)
