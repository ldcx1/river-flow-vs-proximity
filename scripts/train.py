"""Detection and feature-group ablation (paper Sec. V-A). Produces detection.txt.

XGBoost under 5 re-splits of 5-fold reach-grouped CV, on growing feature sets:
optical, + upstream prior, + topology, + facility pressure, and a context-only
reference. Then the segment cluster-bootstrap 95% CIs on the PR-AUC gain of the
upstream prior and of the full o+p+t stack over optics alone.

    python scripts/train.py > results/national/detection.txt     # ~1 min
"""

import argparse

import _bootstrap

from experiments.eval import bootstrap_delta_ci, repeated_cv
from experiments.io import load_panel
from experiments.models import ablation_experiments, evaluate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=_bootstrap.data_de("features_national_de.csv"),
                    help="feature-panel CSV")
    ap.add_argument("--seeds", type=int, default=5,
                    help="number of CV re-splits (repeats of the whole experiment)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--model", choices=["auto", "xgb", "logreg"], default="auto")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--tune", action="store_true",
                    help="select hyperparameters per repeat by nested CV (no CIs)")
    ap.add_argument("--inner-folds", type=int, default=3, help="inner folds for --tune")
    args = ap.parse_args()

    panel = load_panel(args.panel)
    seeds = tuple(range(args.seeds))
    prev = panel.y.mean()
    print(f"detection: {len(set(panel.groups))} segments, {len(panel)} rows, "
          f"{int(panel.y.sum())} positives = {prev:.1%}; {args.seeds}x{args.folds}-fold "
          f"segment-grouped CV; PR-AUC baseline {prev:.3f}"
          f"{' + nested HPO' if args.tune else ''}\n")
    exps = ablation_experiments(panel)
    print(f"  {'config':<30} {'ROC-AUC (mean+/-sd)':>20} {'PR-AUC (mean+/-sd)':>20}")
    for name, idx in exps.items():
        if args.tune:
            r = repeated_cv(panel, idx, name=name, n_repeats=args.seeds,
                            n_splits=args.folds, model=args.model, tune=True,
                            inner_splits=args.inner_folds)
            print(r.line())
            continue
        r = evaluate(panel, idx, name=name, seeds=seeds, n_splits=args.folds,
                     model=args.model)
        print(f"  {name:<30} {r.roc_mean:>11.3f} +/-{r.roc_std:.3f} "
              f"{r.pr_mean:>11.3f} +/-{r.pr_std:.3f}")
    if args.tune:
        return 0

    print(f"\n  ablation deltas (PR-AUC, segment cluster-bootstrap 95% CI, "
          f"{args.n_boot} resamples):")
    base = exps["optical only"]
    for name in ("optical + physics", "all (o+p+t)"):
        b = bootstrap_delta_ci(panel, base, exps[name], seeds=seeds, n_boot=args.n_boot,
                               n_splits=args.folds, model=args.model)
        verdict = "excl. 0" if b.ci_lo > 0 or b.ci_hi < 0 else "incl. 0"
        print(f"    {name + ' - optical only':<40} {b.point_delta:+.3f}  "
              f"[{b.ci_lo:+.3f}, {b.ci_hi:+.3f}]  P(>0)={b.prob_positive:.0%}  ({verdict})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
