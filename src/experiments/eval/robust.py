"""Repeated cross-validation, with optional leakage-free hyperparameter tuning.

With few labelled segments, one fold partition can be lucky. This module
re-splits the data ``n_repeats`` times (segment-grouped) and reports the spread.
With ``tune=True``, hyperparameters are chosen by nested CV inside each repeat,
so the outer test folds never inform tuning.

Repeated CV measures stability under re-splitting. The bootstrap in
:func:`experiments.eval.bootstrap_delta_ci` measures uncertainty over segments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from ..io.panels import Panel
from ..metrics import average_precision, roc_auc
from ..models.train import nested_oof_predict, oof_predict


def _score(panel: Panel, idx, seed, n_splits, model, tune, grid, inner_splits):
    if tune:
        oof, _ = nested_oof_predict(panel.X[:, list(idx)], panel.y, panel.groups,
                                    param_grid=grid, outer_splits=n_splits,
                                    inner_splits=inner_splits, seed=seed, model=model)
    else:
        oof = oof_predict(panel.X[:, list(idx)], panel.y, panel.groups,
                          seed=seed, n_splits=n_splits, model=model)
    return roc_auc(panel.y, oof), average_precision(panel.y, oof)


@dataclass
class RepeatedResult:
    name: str
    n_repeats: int
    tuned: bool
    roc: List[float] = field(default_factory=list)
    pr: List[float] = field(default_factory=list)

    @property
    def roc_mean(self):
        return float(np.mean(self.roc))

    @property
    def pr_mean(self):
        return float(np.mean(self.pr))

    def line(self) -> str:
        return (f"{self.name:<26} ROC {np.mean(self.roc):.3f}+/-{np.std(self.roc):.3f} "
                f"[{min(self.roc):.3f},{max(self.roc):.3f}]  "
                f"PR {np.mean(self.pr):.3f}+/-{np.std(self.pr):.3f} "
                f"[{min(self.pr):.3f},{max(self.pr):.3f}]")


def repeated_cv(panel: Panel, idx: Sequence[int], name: str = "",
                n_repeats: int = 10, n_splits: int = 5, model: str = "auto",
                tune: bool = False, grid: Optional[List[dict]] = None,
                inner_splits: int = 3) -> RepeatedResult:
    """Run ``n_repeats`` independent segment-grouped CV re-splits for one feature
    set; collect per-repeat ROC/PR-AUC."""
    res = RepeatedResult(name=name or f"{len(idx)} feats", n_repeats=n_repeats, tuned=tune)
    for r in range(n_repeats):
        roc, pr = _score(panel, idx, r, n_splits, model, tune, grid, inner_splits)
        res.roc.append(roc)
        res.pr.append(pr)
    return res


@dataclass
class RepeatedPairedResult:
    name_a: str
    name_b: str
    n_repeats: int
    tuned: bool
    a_pr: List[float]
    b_pr: List[float]
    deltas: List[float]

    @property
    def delta_mean(self):
        return float(np.mean(self.deltas))

    @property
    def wins(self):
        return int(sum(1 for d in self.deltas if d > 0))

    def summary(self) -> str:
        d = np.array(self.deltas)
        tag = "TUNED (nested CV)" if self.tuned else "fixed params"
        lo, hi = (np.percentile(d, [2.5, 97.5]) if len(d) >= 4 else (d.min(), d.max()))
        return (
            f"repeated CV over {self.n_repeats} re-splits ({tag})\n\n"
            f"  {self.name_a:<16} PR-AUC: {np.mean(self.a_pr):.3f} +/- {np.std(self.a_pr):.3f}\n"
            f"  {self.name_b:<16} PR-AUC: {np.mean(self.b_pr):.3f} +/- {np.std(self.b_pr):.3f}\n\n"
            f"  PR-AUC delta (B - A): mean {d.mean():+.3f} +/- {d.std():.3f}  "
            f"range [{d.min():+.3f}, {d.max():+.3f}]\n"
            f"  empirical 2.5-97.5 pct: [{lo:+.3f}, {hi:+.3f}]\n"
            f"  B wins: {self.wins}/{self.n_repeats} re-splits"
        )


def repeated_paired(panel: Panel, idx_a: Sequence[int], idx_b: Sequence[int],
                    name_a: str = "A", name_b: str = "B", n_repeats: int = 10,
                    n_splits: int = 5, model: str = "auto", tune: bool = False,
                    grid: Optional[List[dict]] = None,
                    inner_splits: int = 3) -> RepeatedPairedResult:
    """Paired A-vs-B over ``n_repeats`` re-splits. Each repeat uses the same split
    for both feature sets, so the per-repeat delta is paired. With ``tune=True``
    each set gets its own nested-CV hyperparameters."""
    a_pr, b_pr, deltas = [], [], []
    for r in range(n_repeats):
        _, pa = _score(panel, idx_a, r, n_splits, model, tune, grid, inner_splits)
        _, pb = _score(panel, idx_b, r, n_splits, model, tune, grid, inner_splits)
        a_pr.append(pa)
        b_pr.append(pb)
        deltas.append(pb - pa)
    return RepeatedPairedResult(name_a, name_b, n_repeats, tune, a_pr, b_pr, deltas)
