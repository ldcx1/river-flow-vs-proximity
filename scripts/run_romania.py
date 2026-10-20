"""Germany-to-Romania transfer, evaluated per WFD water body (paper: Romanian detector).

The target is WFD-2022 ecological status worse than "good", one label per
water body. Per-reach scores are averaged per body, and CIs come from a body
cluster bootstrap. Three regimes, each with optical and optical+flow-prior
features:
  zero-shot            train on Germany, predict Romania
  DE->RO fine-tuned    continue boosting the German model on a Romanian fold
  RO-only calibration  train on the Romanian fold only
Body-grouped 5-fold CV. Also prints the paired flow-prior effect. Needs the
``models`` extra.

    python scripts/run_romania.py > results/national/romania.txt
"""
import argparse
import csv

import numpy as np
import xgboost as xgb

import _bootstrap
from experiments.cv import grouped_folds  # noqa: E402
from experiments.eval import cluster_bootstrap_ci, cluster_bootstrap_delta  # noqa: E402
from experiments.metrics import average_precision, prevalence, roc_auc  # noqa: E402

XGB = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
           subsample=0.8, colsample_bytree=0.8, min_child_weight=3, n_jobs=4)
FT_TREES, FOLDS, SEED = 100, 5, 0


def spw(y):
    return float((y == 0).sum()) / max(1, int((y == 1).sum()))


def _read(path, cols):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    X = np.full((len(rows), len(cols)), np.nan)
    y = np.array([int(float(r["label"])) for r in rows], float)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            v = r.get(c, "")
            if v not in ("", None):
                try:
                    X[i, j] = float(v)
                except ValueError:
                    pass
    return X, y


def by_body(scores, y, wb):
    """Average per-reach scores to one score per WFD water body; the label is shared
    within a body. Returns body-level (labels, scores, body-ids)."""
    bodies = sorted(set(wb))
    bs = np.array([scores[wb == b].mean() for b in bodies])
    yl = np.array([int(round(y[wb == b].mean())) for b in bodies], float)
    return yl, bs, np.array(bodies)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--de", default=_bootstrap.data_de("features_national_de.csv"),
                    help="German training panel CSV")
    ap.add_argument("--ro", default=_bootstrap.data_ro("features_ro_national.csv"),
                    help="Romanian test panel CSV")
    ap.add_argument("--seglab", default=_bootstrap.data_ro("segment_labels_ro_national.csv"),
                    help="Romanian per-reach WFD labels CSV (object_id, wfd_code)")
    ap.add_argument("--region", default="all Romania",
                    help="region name printed in the header")
    args = ap.parse_args()
    DE, RO, SEGLAB, REGION = args.de, args.ro, args.seglab, args.region

    de_cols = csv.DictReader(open(DE, encoding="utf-8")).fieldnames
    ro_cols = set(csv.DictReader(open(RO, encoding="utf-8")).fieldnames)
    fam = lambda p: sorted(c for c in de_cols if c.startswith(p) and c in ro_cols)
    opt, prior = fam("o_"), fam("p_")
    configs = {"optical": opt, "optical+flow-prior": opt + prior}

    body = {r["object_id"]: r["wfd_code"]
            for r in csv.DictReader(open(SEGLAB, encoding="utf-8"))}
    ro_rows = list(csv.DictReader(open(RO, encoding="utf-8")))
    wb = np.array([body.get(r["object_id"], r["object_id"]) for r in ro_rows])
    y_reach = np.array([int(float(r["label"])) for r in ro_rows], float)
    fold = grouped_folds(wb, y_reach, FOLDS, SEED)            # body-grouped, over reaches
    n_bodies = len(set(wb))
    yl0, _, _ = by_body(np.zeros(len(wb)), y_reach, wb)

    lines = ["GERMANY -> ROMANIA transfer with local calibration, per WFD water body.",
             f"Target = WFD-2022 ecological status worse than good ({REGION}).",
             f"train pool: German national panel; test: {n_bodies} WFD water bodies "
             f"({len(ro_rows)} reaches), body prevalence {prevalence(yl0):.1%} "
             f"(chance ROC 0.5).\n",
             f"  {'features / regime':<42}{'ROC [95% CI]':<26}{'PR-AUC [95% CI]':<24}"]
    body_sc = {}   # (config, regime) -> (body labels, body scores, body ids)
    for name, cols in configs.items():
        Xde, yde = _read(DE, cols)
        Xro, _ = _read(RO, cols)
        base = xgb.XGBClassifier(**XGB, scale_pos_weight=spw(yde),
                                 random_state=SEED).fit(Xde, yde)
        zs = base.predict_proba(Xro)[:, 1]                   # zero-shot (DE only)
        ft = np.zeros(len(Xro)); ro = np.zeros(len(Xro))     # OOF, body-grouped CV
        for k in range(FOLDS):
            te = fold == k; tr = ~te
            sw = spw(y_reach[tr])
            m_ft = xgb.XGBClassifier(n_estimators=FT_TREES, max_depth=4, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                                     n_jobs=4, scale_pos_weight=sw, random_state=SEED)
            m_ft.fit(Xro[tr], y_reach[tr], xgb_model=base.get_booster())   # fine-tune
            ft[te] = m_ft.predict_proba(Xro[te])[:, 1]
            m_ro = xgb.XGBClassifier(**XGB, scale_pos_weight=sw,
                                     random_state=SEED).fit(Xro[tr], y_reach[tr])
            ro[te] = m_ro.predict_proba(Xro[te])[:, 1]
        for reg, sc in [("zero-shot", zs), ("DE->RO fine-tuned", ft),
                        ("RO-only calibration", ro)]:
            yl, bs, bid = by_body(sc, y_reach, wb)
            body_sc[(name, reg)] = (yl, bs, bid)
            rc = cluster_bootstrap_ci(yl, bs, bid, metric=roc_auc, n_boot=2000)
            pc = cluster_bootstrap_ci(yl, bs, bid, metric=average_precision, n_boot=2000)
            lines.append(f"  {name+' / '+reg:<42}{rc.point:.3f} [{rc.ci_lo:.3f},{rc.ci_hi:.3f}]    "
                         f"{pc.point:.3f} [{pc.ci_lo:.3f},{pc.ci_hi:.3f}]")
        lines.append("")

    # Paired flow-prior effect over the same bodies. CI excluding 0 => the prior helps.
    lines.append("Flow-prior effect = (optical+flow-prior) minus optical, paired body bootstrap:")
    for reg in ("zero-shot", "DE->RO fine-tuned", "RO-only calibration"):
        yl, bs_o, bid = body_sc[("optical", reg)]
        _, bs_p, _ = body_sc[("optical+flow-prior", reg)]
        dr = cluster_bootstrap_delta(yl, bs_o, bs_p, bid, metric=roc_auc, n_boot=2000)
        dp = cluster_bootstrap_delta(yl, bs_o, bs_p, bid, metric=average_precision, n_boot=2000)
        star = "*" if dr.ci_lo > 0 else " "
        lines.append(f"  {reg:<26} dROC {dr.point:+.3f} [{dr.ci_lo:+.3f},{dr.ci_hi:+.3f}]{star}  "
                     f"dPR {dp.point:+.3f} [{dp.ci_lo:+.3f},{dp.ci_hi:+.3f}]")
    lines.append("  (* ROC delta CI excludes 0)")
    lines.append("")
    lines.append("Region-specific optical retrieval: zero-shot transfer is weak; local WFD")
    lines.append("calibration (fine-tune or RO-only) is what makes the model usable in Romania.")
    lines.append(f"Weak-label stratum (N={n_bodies} bodies): not comparable to German strong labels.")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
