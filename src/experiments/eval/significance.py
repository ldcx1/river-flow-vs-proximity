"""Significance tests for comparing two feature sets or two score vectors.

The paired seed test scores both feature sets on the same segment-grouped folds
for each seed, so the per-seed PR-AUC delta is paired. The cluster bootstraps
resample whole segments and give population-level intervals. A Benjamini-Hochberg
helper adjusts p-values across a family of effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..io.panels import Panel
from ..metrics import average_precision, roc_auc
from ..models.train import oof_predict


@dataclass
class PairedResult:
    name_a: str
    name_b: str
    n_seeds: int
    a_pr_mean: float
    a_pr_std: float
    b_pr_mean: float
    b_pr_std: float
    delta_pr_mean: float
    delta_pr_ci_lo: float
    delta_pr_ci_hi: float
    pr_wins: int
    delta_roc_mean: float
    roc_wins: int

    @property
    def seed_stable(self) -> bool:
        """The per-seed CI excludes 0.

        This measures seed/split stability on a fixed dataset, not significance
        over rivers. The half-width shrinks as 1/sqrt(n_seeds) however small the
        true effect. For significance use :func:`bootstrap_delta_ci`."""
        return self.delta_pr_ci_lo > 0 or self.delta_pr_ci_hi < 0

    def summary(self) -> str:
        verdict = ("stable across seeds (per-seed CI excludes 0)" if self.seed_stable
                   else "not stable at 95% across seeds")
        return (
            f"paired comparison over {self.n_seeds} RNG seeds (segment-grouped CV)\n"
            f"  [measures seed/split stability on a fixed dataset, not population\n"
            f"   significance over rivers - use bootstrap_delta_ci for that]\n\n"
            f"  {self.name_a:<16} PR-AUC: {self.a_pr_mean:.3f} +/- {self.a_pr_std:.3f}\n"
            f"  {self.name_b:<16} PR-AUC: {self.b_pr_mean:.3f} +/- {self.b_pr_std:.3f}\n\n"
            f"  PR-AUC delta (B - A): mean {self.delta_pr_mean:+.3f}  "
            f"per-seed 95% CI [{self.delta_pr_ci_lo:+.3f}, {self.delta_pr_ci_hi:+.3f}]\n"
            f"  B wins on PR-AUC: {self.pr_wins}/{self.n_seeds} seeds\n"
            f"  ROC-AUC delta: mean {self.delta_roc_mean:+.3f}, "
            f"wins {self.roc_wins}/{self.n_seeds}\n\n"
            f"  seed-stability: {verdict}"
        )


@dataclass
class BootstrapResult:
    name_a: str
    name_b: str
    point_delta: float
    boot_mean: float
    ci_lo: float
    ci_hi: float
    prob_positive: float
    n_boot: int

    @property
    def significant(self) -> bool:
        """Population-level: the segment-bootstrap 95% CI excludes 0."""
        return self.ci_lo > 0 or self.ci_hi < 0

    def summary(self) -> str:
        verdict = ("significant over rivers (CI excludes 0)" if self.significant
                   else "not significant over rivers (CI includes 0)")
        return (
            f"segment cluster-bootstrap ({self.n_boot} resamples of the segments)\n\n"
            f"  point delta ({self.name_b} - {self.name_a}): {self.point_delta:+.3f}\n"
            f"  bootstrap mean {self.boot_mean:+.3f}  "
            f"95% CI [{self.ci_lo:+.3f}, {self.ci_hi:+.3f}]\n"
            f"  P(delta > 0) = {self.prob_positive:.1%}\n\n"
            f"  verdict: {verdict}"
        )


def paired_seed_test(panel: Panel, idx_a: Sequence[int], idx_b: Sequence[int],
                     name_a: str = "A", name_b: str = "B",
                     n_seeds: int = 20, n_splits: int = 5,
                     model: str = "auto") -> PairedResult:
    """Compare feature set B (``idx_b``) against A (``idx_a``) on the same
    segment-grouped folds for each of ``n_seeds`` seeds. The hypothesis is that B
    has higher PR-AUC."""
    y = panel.y
    a_pr, b_pr, d_pr, d_roc = [], [], [], []
    for s in range(n_seeds):
        pa = oof_predict(panel.X[:, list(idx_a)], y, panel.groups, s, n_splits, model)
        pb = oof_predict(panel.X[:, list(idx_b)], y, panel.groups, s, n_splits, model)
        apa, apb = average_precision(y, pa), average_precision(y, pb)
        ra, rb = roc_auc(y, pa), roc_auc(y, pb)
        a_pr.append(apa)
        b_pr.append(apb)
        d_pr.append(apb - apa)
        d_roc.append(rb - ra)
    d = np.array(d_pr)
    # t-quantile with n_seeds-1 df, since n is small. Still only a seed-stability
    # interval (see PairedResult.seed_stable).
    if n_seeds > 1:
        try:
            from scipy.stats import t as _t
            crit = float(_t.ppf(0.975, n_seeds - 1))
        except Exception:
            crit = 2.093 if n_seeds <= 25 else 1.96
        ci = crit * d.std(ddof=1) / math.sqrt(n_seeds)
    else:
        ci = 0.0
    return PairedResult(
        name_a=name_a, name_b=name_b, n_seeds=n_seeds,
        a_pr_mean=float(np.mean(a_pr)), a_pr_std=float(np.std(a_pr)),
        b_pr_mean=float(np.mean(b_pr)), b_pr_std=float(np.std(b_pr)),
        delta_pr_mean=float(d.mean()),
        delta_pr_ci_lo=float(d.mean() - ci), delta_pr_ci_hi=float(d.mean() + ci),
        pr_wins=int((d > 0).sum()),
        delta_roc_mean=float(np.mean(d_roc)), roc_wins=int((np.array(d_roc) > 0).sum()),
    )


def bootstrap_delta_ci(panel: Panel, idx_a: Sequence[int], idx_b: Sequence[int],
                       name_a: str = "A", name_b: str = "B",
                       seeds: Sequence[int] = (0, 1, 2, 3, 4),
                       n_boot: int = 2000, n_splits: int = 5,
                       boot_seed: int = 0, model: str = "auto") -> BootstrapResult:
    """Population-level 95% CI on the PR-AUC delta (B minus A).

    Out-of-fold predictions are averaged over ``seeds``. Segments, the inference
    unit, are then resampled with replacement and the delta recomputed on each
    resample. Resamples with a single class are skipped.
    """
    y = panel.y
    groups = panel.groups
    pa = np.mean([oof_predict(panel.X[:, list(idx_a)], y, groups, s, n_splits, model)
                  for s in seeds], axis=0)
    pb = np.mean([oof_predict(panel.X[:, list(idx_b)], y, groups, s, n_splits, model)
                  for s in seeds], axis=0)
    point = average_precision(y, pb) - average_precision(y, pa)

    uniq = np.array(sorted(set(groups.tolist())))
    rows_by = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(boot_seed)
    deltas = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        ri = np.concatenate([rows_by[g] for g in samp])
        yy = y[ri]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        deltas.append(average_precision(yy, pb[ri]) - average_precision(yy, pa[ri]))
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return BootstrapResult(
        name_a=name_a, name_b=name_b, point_delta=float(point),
        boot_mean=float(deltas.mean()), ci_lo=float(lo), ci_hi=float(hi),
        prob_positive=float((deltas > 0).mean()), n_boot=len(deltas))


# --------------------------------------------------------------------------
# Generic, model-free inference helpers (operate on precomputed scores)
# --------------------------------------------------------------------------
@dataclass
class CIResult:
    """Bootstrap interval on a metric, or a metric delta, from precomputed scores."""
    point: float
    ci_lo: float
    ci_hi: float
    prob_positive: float   # P(delta > 0), NaN for a single-metric CI
    n_boot: int

    @property
    def significant(self) -> bool:
        """The 95% CI excludes 0 (meaningful for a delta interval)."""
        return self.ci_lo > 0 or self.ci_hi < 0


def _cluster_resample(groups, seed: int, n_boot: int):
    """Yield row-index arrays, each a with-replacement resample of whole groups."""
    groups = np.asarray(groups)
    uniq = np.array(sorted(set(groups.tolist())))
    rows_by = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        yield np.concatenate([rows_by[g] for g in samp])


def cluster_bootstrap_delta(y, pa, pb, groups, metric=average_precision,
                            n_boot: int = 2000, seed: int = 0) -> CIResult:
    """Segment cluster-bootstrap CI on ``metric(y, pb) - metric(y, pa)``.

    Works on precomputed score vectors (e.g. existing out-of-fold predictions),
    so no model is refit."""
    y = np.asarray(y, dtype=float)
    pa = np.asarray(pa, dtype=float)
    pb = np.asarray(pb, dtype=float)
    point = metric(y, pb) - metric(y, pa)
    deltas = []
    for ri in _cluster_resample(groups, seed, n_boot):
        yy = y[ri]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        d = metric(yy, pb[ri]) - metric(yy, pa[ri])
        if np.isfinite(d):
            deltas.append(d)
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5]) if len(deltas) else (float("nan"),) * 2
    return CIResult(float(point), float(lo), float(hi),
                    float((deltas > 0).mean()) if len(deltas) else float("nan"),
                    len(deltas))


def cluster_bootstrap_ci(y, scores, groups, metric=roc_auc,
                         n_boot: int = 2000, seed: int = 0) -> CIResult:
    """Segment cluster-bootstrap CI on a single ``metric(y, scores)``.
    ``prob_positive`` is NaN because this is not a delta."""
    y = np.asarray(y, dtype=float)
    scores = np.asarray(scores, dtype=float)
    point = metric(y, scores)
    vals = []
    for ri in _cluster_resample(groups, seed, n_boot):
        yy = y[ri]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        v = metric(yy, scores[ri])
        if np.isfinite(v):
            vals.append(v)
    vals = np.array(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5]) if len(vals) else (float("nan"),) * 2
    return CIResult(float(point), float(lo), float(hi), float("nan"), len(vals))


def bh_adjust(pvalues, alpha: float = 0.05):
    """Benjamini-Hochberg step-up FDR adjustment.

    Returns ``(adjusted, reject)``: adjusted p-values in input order (monotone,
    clipped to 1) and a mask of hypotheses rejected at FDR ``alpha``."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([]), np.array([], dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1.0)
    adj = np.minimum.accumulate(adj[::-1])[::-1]   # enforce monotonicity
    adj = np.clip(adj, 0.0, 1.0)
    adjusted = np.empty(n)
    adjusted[order] = adj
    return adjusted, adjusted <= alpha
