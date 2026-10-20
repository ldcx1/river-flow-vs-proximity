"""Class weighting is computed per training fold (no train/validation coupling),
and the permutation null has enough draws not to floor the p-values."""

import inspect

import numpy as np
import pytest

from experiments.eval.temporal_transport import temporal_transport
from experiments.models.train import logreg_fit_predict, make_fit_predict


def test_temporal_transport_default_n_perm_is_2000():
    # p-values are floored at 1/(n_perm + 1), so 2000 draws give a 0.0005 floor
    assert inspect.signature(temporal_transport).parameters["n_perm"].default == 2000


def test_logreg_handles_spw_none_per_fold():
    rng = np.random.default_rng(0)
    X_tr = rng.random((40, 3))
    y_tr = (rng.random(40) < 0.3).astype(float)
    X_va = rng.random((10, 3))
    fp = logreg_fit_predict(None, seed=0)        # spw=None -> weight from train fold
    p = fp(X_tr, y_tr, X_va)
    assert p.shape == (10,)
    assert np.all((p >= 0) & (p <= 1))


def test_spw_none_weight_differs_by_fold_balance():
    # different training-fold class balance must give a different class weight
    rng = np.random.default_rng(1)
    X = rng.random((50, 2))
    y_bal = np.array([0, 1] * 25, dtype=float)        # 50% positive
    y_rare = np.array([0] * 45 + [1] * 5, dtype=float)  # 10% positive
    fp = logreg_fit_predict(None, seed=0)
    p_bal = fp(X, y_bal, X[:5])
    p_rare = fp(X, y_rare, X[:5])
    # the two fits use different class weights -> predictions are not identical
    assert not np.allclose(p_bal, p_rare)


def test_xgboost_spw_none_runs_and_varies_by_fold_balance():
    # Same check for the XGBoost model used in the paper (make_fit_predict).
    pytest.importorskip("xgboost")
    rng = np.random.default_rng(1)
    X = rng.random((60, 3))
    y_bal = np.array([0, 1] * 30, dtype=float)          # 50% positive
    y_rare = np.array([0] * 54 + [1] * 6, dtype=float)  # 10% positive
    fp = make_fit_predict(None, seed=0)
    p_bal = fp(X, y_bal, X[:5])
    p_rare = fp(X, y_rare, X[:5])
    assert p_bal.shape == (5,) and np.all((p_bal >= 0) & (p_bal <= 1))
    assert not np.allclose(p_bal, p_rare)
