"""Classifier comparison on the detection panel (paper Table II).

Same panel and repeated segment-grouped CV as train.py. Compares logistic
regression, random forest and XGBoost on optical-only and all features. Also
prints a cluster-bootstrap delta for logistic regression (all vs optical) and the
top-10% screening lift of optical XGBoost. The GAT/MLP comparison is in
run_gnn.py.

    python scripts/run_model_comparison.py > results/national/model_comparison.txt
"""

import argparse

import _bootstrap
import numpy as np

from experiments.cv import cross_val_oof, grouped_folds
from experiments.eval import cluster_bootstrap_delta
from experiments.io import load_panel
from experiments.metrics import average_precision, roc_auc
from experiments.models import build_fit_predict


def rf_fit_predict(seed: int):
    """Random forest with class weighting and train-fold mean imputation."""
    from sklearn.ensemble import RandomForestClassifier

    def fit_predict(X_tr, y_tr, X_va):
        mu = np.nanmean(X_tr, axis=0)
        mu = np.where(np.isnan(mu), 0.0, mu)
        Xtr = np.where(np.isnan(X_tr), mu, X_tr)
        Xva = np.where(np.isnan(X_va), mu, X_va)
        clf = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, n_jobs=4,
            class_weight="balanced_subsample", random_state=seed)
        clf.fit(Xtr, y_tr)
        return clf.predict_proba(Xva)[:, 1]

    return fit_predict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=_bootstrap.data_de("features_national_de.csv"),
                    help="feature-panel CSV")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    panel = load_panel(args.panel)
    prev = panel.y.mean()
    print(f"panel: {len(panel)} cells, {len(set(panel.groups))} segments, "
          f"prevalence {prev:.3f} | {args.seeds} re-splits, "
          f"{args.folds}-fold segment-grouped CV")
    print(f"baseline PR-AUC (prevalence) = {prev:.3f}\n")

    featsets = {"optical only": panel.select(keep={"optical"}),
                "all features": panel.select()}
    oofs = {}  # (feature-set, model) -> seed-mean OOF scores
    for fname, idx in featsets.items():
        print(f"--- {fname} ({len(idx)} features) ---")
        for mname in ("logreg", "rf", "xgb"):
            rocs, aps, seed_oofs = [], [], []
            for s in range(args.seeds):
                fold = grouped_folds(panel.groups, panel.y, args.folds, s)
                if mname == "rf":
                    fp = rf_fit_predict(s)
                else:
                    fp = build_fit_predict(mname, None, s)
                oof = cross_val_oof(panel.X[:, idx], panel.y, fold, fp, args.folds)
                seed_oofs.append(oof)
                rocs.append(roc_auc(panel.y, oof))
                aps.append(average_precision(panel.y, oof))
            oofs[(fname, mname)] = np.mean(seed_oofs, axis=0)
            print(f"{mname:<8} ROC {np.mean(rocs):.3f}+/-{np.std(rocs):.3f}  "
                  f"PR {np.mean(aps):.3f}+/-{np.std(aps):.3f}")
        print()

    # Logistic regression gains ROC from the full feature set. Test that delta
    # with the same segment cluster bootstrap as the XGBoost ablation.
    pa = oofs[("optical only", "logreg")]
    pb = oofs[("all features", "logreg")]
    for metric, mname in ((roc_auc, "ROC-AUC"), (average_precision, "AP")):
        r = cluster_bootstrap_delta(panel.y, pa, pb, panel.groups, metric=metric,
                                    n_boot=2000)
        print(f"logreg all-vs-optical {mname} delta {r.point:+.3f} "
              f"[{r.ci_lo:+.3f},{r.ci_hi:+.3f}]  P(>0)={r.prob_positive:.3f}")

    # Exceedance rate in the top decile of ranked reach-months vs the base rate.
    scores = oofs[("optical only", "xgb")]
    k = max(1, int(0.10 * len(scores)))
    top = np.argsort(-scores)[:k]
    lift_rate = float(panel.y[top].mean())
    print(f"\nxgb optical top-10% screening: exceedance rate {lift_rate:.3f} "
          f"vs base rate {prev:.3f} ({lift_rate / prev:.2f}x, n={k})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
