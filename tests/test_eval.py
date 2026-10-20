"""Evaluation helpers on a tiny synthetic panel: paired seed test, segment
bootstrap, BH adjustment. Uses the numpy logistic model, so xgboost is not
needed. Checks plumbing and invariants, not a scientific result."""

import numpy as np

from experiments.eval import (
    bh_adjust,
    bootstrap_delta_ci,
    cluster_bootstrap_ci,
    paired_seed_test,
)
from experiments.io.panels import Panel
from experiments.metrics import roc_auc


def _synthetic_panel(n_groups=40, rows_per_group=4, seed=0):
    rng = np.random.default_rng(seed)
    X, y, groups = [], [], []
    for g in range(n_groups):
        signal = rng.random()                 # per-segment informative feature
        for _ in range(rows_per_group):
            noise = rng.random()              # uninformative "optical" feature
            label = float(rng.random() < 0.2 + 0.6 * signal)
            X.append([noise, signal + 0.05 * rng.standard_normal()])
            y.append(label)
            groups.append(f"seg{g}")
    return Panel(X=np.array(X), y=np.array(y, dtype=float),
                 groups=np.array(groups), feature_cols=["o_noise", "f_ag_signal"],
                 rows=[])


def test_paired_seed_test_runs_and_is_consistent():
    panel = _synthetic_panel()
    a = panel.select_prefix(["o_"])
    b = panel.select_prefix(["o_", "f_ag_"])
    res = paired_seed_test(panel, a, b, "opt", "opt+ag", n_seeds=4, model="logreg")
    assert 0 <= res.pr_wins <= 4
    assert res.delta_pr_ci_lo <= res.delta_pr_mean <= res.delta_pr_ci_hi
    # the informative f_ag_ feature should not make B worse on average here
    assert res.delta_pr_mean > -0.1


def test_bootstrap_delta_ci_invariants():
    panel = _synthetic_panel()
    a = panel.select_prefix(["o_"])
    b = panel.select_prefix(["o_", "f_ag_"])
    boot = bootstrap_delta_ci(panel, a, b, "opt", "opt+ag", seeds=(0, 1),
                              n_boot=200, model="logreg")
    assert boot.ci_lo <= boot.boot_mean <= boot.ci_hi
    assert 0.0 <= boot.prob_positive <= 1.0
    assert boot.n_boot > 0


def test_bootstrap_delta_ci_noise_feature_includes_zero():
    """A pure-noise feature must not give a significant gain. Resampling rows
    instead of segments would shrink the CI and could make noise look significant."""
    rng = np.random.default_rng(3)
    X, y, groups = [], [], []
    for g in range(40):
        sig = rng.random()
        for _ in range(4):
            X.append([sig + 0.05 * rng.standard_normal(), rng.standard_normal()])  # col1 pure noise
            y.append(float(rng.random() < 0.2 + 0.6 * sig))
            groups.append(f"seg{g}")
    panel = Panel(X=np.array(X), y=np.array(y, dtype=float), groups=np.array(groups),
                  feature_cols=["o_real", "f_ag_noise"], rows=[])
    a = panel.select_prefix(["o_"])
    b = panel.select_prefix(["o_", "f_ag_"])
    boot = bootstrap_delta_ci(panel, a, b, "opt", "opt+noise", seeds=(0, 1),
                              n_boot=300, model="logreg")
    assert boot.ci_lo <= 0.0    # noise does not manufacture a significant gain


def _scored(n_groups=60, rows_per_group=4, signal=True, seed=0):
    """Build (y, scores, groups) with optional real signal in the scores."""
    rng = np.random.default_rng(seed)
    y, s, g = [], [], []
    for gi in range(n_groups):
        base = rng.random()
        for _ in range(rows_per_group):
            label = float(rng.random() < 0.2 + 0.6 * base)
            score = (base if signal else rng.random()) + 0.05 * rng.standard_normal()
            y.append(label); s.append(score); g.append(f"seg{gi}")
    return np.array(y, float), np.array(s, float), np.array(g)


def test_bh_adjust_known_vector():
    p = np.array([0.001, 0.01, 0.02, 0.5])
    adj, reject = bh_adjust(p, alpha=0.05)
    # BH adjusted = p * n / rank, made monotone: [0.004, 0.02, ~0.0267, 0.5]
    assert np.isclose(adj[0], 0.004)
    assert np.isclose(adj[1], 0.02)
    assert np.isclose(adj[2], 0.02 * 4 / 3)
    assert adj[3] <= 1.0 and adj[3] >= adj[2]
    assert reject.tolist() == [True, True, True, False]


def test_bh_adjust_empty():
    adj, reject = bh_adjust([])
    assert adj.size == 0 and reject.size == 0


def test_cluster_bootstrap_ci_signal_vs_noise():
    y, s, g = _scored(signal=True, seed=1)
    ci = cluster_bootstrap_ci(y, s, g, metric=roc_auc, n_boot=300, seed=0)
    assert ci.ci_lo > 0.5           # a real signal: AUC interval above chance
    yn, sn, gn = _scored(signal=False, seed=2)
    cin = cluster_bootstrap_ci(yn, sn, gn, metric=roc_auc, n_boot=300, seed=0)
    assert cin.ci_lo <= 0.5 <= cin.ci_hi   # noise: AUC interval straddles chance
