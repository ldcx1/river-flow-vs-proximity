"""Calibration and abstention on the detection panel (extra analysis, not in the paper).

Bagged XGBoost ensemble with isotonic calibration and selective prediction:
ensemble disagreement gives an uncertainty, and the most uncertain cells can be
skipped. Within each segment-grouped test fold, the training segments are split
into a fit set and a calibration set, so calibration never sees test data.
Needs the ``models`` and ``dev`` extras. The figure is drawn only with --out
and then also needs matplotlib (the ``figures`` extra).

    python scripts/run_ensemble.py > results/national/ensemble.txt
    python scripts/run_ensemble.py --out fig_uncertainty.pdf
"""
import argparse

import numpy as np
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

import _bootstrap
from experiments.cv import grouped_folds  # noqa: E402
from experiments.io import load_panel  # noqa: E402
from experiments.metrics import (  # noqa: E402
    average_precision,
    brier_score,
    expected_calibration_error,
    roc_auc,
)

N_ENS, SEED, FOLDS = 12, 0, 5


def brier(y, p):
    return brier_score(y, p)


def ece(y, p, bins=10):
    return expected_calibration_error(y, p, n_bins=bins)


def fit_ens(X, y, n, seed0):
    rng = np.random.default_rng(seed0)
    spw = (y == 0).sum() / max(1, (y == 1).sum())
    models = []
    for j in range(n):
        idx = rng.integers(0, len(y), len(y))            # bootstrap sample
        clf = xgb.XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                                scale_pos_weight=spw, random_state=j, n_jobs=4)
        clf.fit(X[idx], y[idx]); models.append(clf)
    return models


def pred_ens(models, X):
    P = np.stack([m.predict_proba(X)[:, 1] for m in models])
    return P.mean(0), P.std(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=_bootstrap.data_de("features_national_de.csv"),
                    help="feature-panel CSV")
    ap.add_argument("--out", default=None,
                    help="write the reliability and selective-prediction figure here")
    args = ap.parse_args()

    panel = load_panel(args.panel)
    X, y, groups = panel.X, panel.y, panel.groups
    prev = y.mean()
    fold = grouped_folds(groups, y, FOLDS, SEED)
    # grouped_folds(SEED) consumes one shuffle of the sorted groups. Advance a
    # local rng(SEED) the same way before drawing calibration groups so the
    # published numbers reproduce.
    rng = np.random.default_rng(SEED)
    rng.shuffle(np.array(sorted(set(groups.tolist()))))
    n = len(y)
    p_single = np.zeros(n); p_ens = np.zeros(n); p_cal = np.zeros(n); u = np.zeros(n)

    for k in range(FOLDS):
        te = fold == k; tr = np.where(~te)[0]
        gtr = np.array(sorted(set(groups[tr]))); rng.shuffle(gtr)
        cut = max(1, int(0.2 * len(gtr)))
        cal_g = set(gtr[:cut])
        fit_idx = np.array([i for i in tr if groups[i] not in cal_g])
        cal_idx = np.array([i for i in tr if groups[i] in cal_g])
        models = fit_ens(X[fit_idx], y[fit_idx], N_ENS, SEED + k)
        single = xgb.XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                                   scale_pos_weight=(y[fit_idx] == 0).sum() /
                                   max(1, (y[fit_idx] == 1).sum()),
                                   random_state=SEED, n_jobs=4).fit(X[fit_idx], y[fit_idx])
        p_single[te] = single.predict_proba(X[te])[:, 1]
        mcal, _ = pred_ens(models, X[cal_idx])
        iso = IsotonicRegression(out_of_bounds="clip").fit(mcal, y[cal_idx])
        mte, ste = pred_ens(models, X[te])
        p_ens[te] = mte; u[te] = ste; p_cal[te] = iso.transform(mte)

    print(f"panel: {n} cells, prevalence {prev:.3f} | ensemble of {N_ENS} bagged XGBoost, "
          f"segment-grouped CV\n")
    print(f"{'model':<26} {'ROC-AUC':>8} {'PR-AUC':>8} {'Brier':>7} {'ECE':>6}")
    print("-" * 58)
    for name, p in [("single XGBoost", p_single), ("ensemble (mean)", p_ens),
                    ("ensemble + isotonic cal.", p_cal)]:
        print(f"{name:<26} {roc_auc(y,p):>8.3f} {average_precision(y,p):>8.3f} "
              f"{brier(y,p):>7.3f} {ece(y,p):>6.3f}")

    print("\nSelective prediction (abstain on high ensemble-uncertainty cells):")
    print(f"  {'coverage':>9} {'kept':>5} {'ret.prev':>8} {'ROC-AUC':>8} {'PR-AUC':>8}")
    order = np.argsort(u)
    covs, rocs = [], []
    for cov in (1.00, 0.90, 0.75, 0.50):
        keep = order[:int(cov * n)]
        yk, pk = y[keep], p_ens[keep]
        roc = roc_auc(yk, pk)
        covs.append(cov); rocs.append(roc)
        print(f"  {cov:>8.0%} {len(keep):>5} {yk.mean():>8.3f} {roc:>8.3f} "
              f"{average_precision(yk,pk):>8.3f}")
    if args.out:
        _fig(y, p_single, p_cal, covs, rocs, args.out)


def _fig(y, p_single, p_cal, covs, rocs, path):
    # Imported here so the table run needs no plotting stack.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for name, p, c in [("uncalibrated", p_single, "#d62728"), ("calibrated", p_cal, "#2ca02c")]:
        edges = np.linspace(0, 1, 11); xs, ys = [], []
        for i in range(10):
            m = (p >= edges[i]) & (p < edges[i + 1] if i < 9 else p <= edges[i + 1])
            if m.sum() >= 10:
                xs.append(p[m].mean()); ys.append(y[m].mean())
        a1.plot(xs, ys, "o-", c=c, ms=4, lw=1.2, label=name)
    a1.plot([0, 1], [0, 1], "k--", lw=0.7)
    a1.set_xlabel("predicted probability"); a1.set_ylabel("observed frequency")
    a1.set_title("(a) Reliability"); a1.legend(fontsize=7)
    a2.plot([c * 100 for c in covs], rocs, "o-", c="#1f77b4", lw=1.3)
    a2.set_xlabel("coverage (%)"); a2.set_ylabel("ROC-AUC on retained")
    a2.set_title("(b) Selective prediction"); a2.invert_xaxis()
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
