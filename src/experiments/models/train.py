"""The classifier, its cross-validation driver, and the ablation runner.

The main model is XGBoost. A pure-numpy L2 logistic regression is the fallback,
so the package and its tests run without the optional ``models`` extra. Both are
``fit_predict(X_train, y_train, X_val) -> probs`` callables for
:func:`experiments.cv.cross_val_oof`.

The XGBoost hyperparameters are the ones behind the paper's numbers. Changing
them means re-running every table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..cv import cross_val_oof, grouped_folds
from ..io.panels import Panel
from ..metrics import average_precision, roc_auc

# Hyperparameters behind every reported result.
XGB_PARAMS = dict(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3, n_jobs=4,
)

# Search space for nested-CV tuning. Each entry overrides XGB_PARAMS. The
# published setting (depth 4, lr 0.05, 300 trees) is included, so tuning cannot
# do worse than it on the inner folds.
DEFAULT_XGB_GRID = [
    {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 300},
    {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 300},   # published default
    {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 300},
    {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 300},
    {"max_depth": 4, "learning_rate": 0.03, "n_estimators": 500},
    {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 500},
    {"max_depth": 4, "learning_rate": 0.10, "n_estimators": 200},
    {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 300, "min_child_weight": 1},
    {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 300, "min_child_weight": 5},
]

FitPredict = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def scale_pos_weight(y: np.ndarray) -> float:
    """``n_neg / n_pos``, the class-imbalance weight for the positive class.

    This uses whatever labels it is given. ``oof_predict`` instead passes
    ``spw=None`` so each fold computes the weight from its own training rows.
    """
    y = np.asarray(y, dtype=float)
    return float((y == 0).sum() / max(1, (y == 1).sum()))


def make_fit_predict(spw: float, seed: int,
                     params: Optional[Dict] = None) -> FitPredict:
    """An XGBoost ``fit_predict``. ``params`` overrides :data:`XGB_PARAMS`.
    ``spw=None`` computes the class weight per training fold. Needs the
    ``models`` extra (xgboost)."""
    import xgboost as xgb

    cfg = {**XGB_PARAMS, **(params or {})}

    def fit_predict(X_tr, y_tr, X_va):
        spw_eff = spw if spw is not None else float(
            (y_tr == 0).sum() / max(1, (y_tr == 1).sum()))   # per training fold
        clf = xgb.XGBClassifier(**cfg, scale_pos_weight=spw_eff,
                                random_state=seed, eval_metric="aucpr")
        clf.fit(X_tr, y_tr)
        return clf.predict_proba(X_va)[:, 1]

    return fit_predict


def logreg_fit_predict(spw: float, seed: int,
                       params: Optional[Dict] = None) -> FitPredict:
    """Pure-numpy class-weighted L2 logistic regression. Mean-imputes NaNs and
    standardises with training-fold statistics only.
    ``params`` may override ``iters``/``lr``/``l2``."""
    p = {"iters": 3000, "lr": 0.2, "l2": 1e-3, **(params or {})}
    iters, lr, l2 = int(p["iters"]), float(p["lr"]), float(p["l2"])

    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit_predict(X_tr, y_tr, X_va):
        mu = np.nanmean(X_tr, axis=0)
        mu = np.where(np.isnan(mu), 0.0, mu)
        Xtr = np.where(np.isnan(X_tr), mu, X_tr)
        Xva = np.where(np.isnan(X_va), mu, X_va)
        sd = Xtr.std(axis=0)
        sd[sd == 0] = 1.0
        Xtr = (Xtr - mu) / sd
        Xva = (Xva - mu) / sd
        n, d = Xtr.shape
        Xb = np.hstack([np.ones((n, 1)), Xtr])
        w = np.zeros(d + 1)
        spw_eff = spw if spw is not None else float(
            (y_tr == 0).sum() / max(1, (y_tr == 1).sum()))   # per training fold
        sw = np.where(y_tr == 1, spw_eff, 1.0)
        for _ in range(iters):
            p = _sigmoid(Xb @ w)
            grad = Xb.T @ ((p - y_tr) * sw) / n
            grad[1:] += l2 * w[1:]
            w -= lr * grad
        return _sigmoid(np.hstack([np.ones((Xva.shape[0], 1)), Xva]) @ w)

    return fit_predict


def _have_xgb() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except Exception:
        return False


def _resolve_model(model: str) -> str:
    if model == "auto":
        return "xgb" if _have_xgb() else "logreg"
    return model


def build_fit_predict(model: str, spw: float, seed: int,
                      params: Optional[Dict] = None) -> FitPredict:
    """Model-agnostic ``fit_predict`` factory. ``model`` resolves xgb/logreg;
    ``params`` overrides that model's hyperparameters."""
    if _resolve_model(model) == "xgb":
        return make_fit_predict(spw, seed, params)
    return logreg_fit_predict(spw, seed, params)


def oof_predict(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                seed: int = 0, n_splits: int = 5,
                model: str = "auto", params: Optional[Dict] = None) -> np.ndarray:
    """Out-of-fold positive-class probabilities under segment-grouped CV.

    ``model`` is ``"xgb"``, ``"logreg"``, or ``"auto"`` (xgb if available).
    ``params`` optionally overrides the model hyperparameters (fixed, no tuning).
    """
    y = np.asarray(y, dtype=float)
    fold = grouped_folds(groups, y, n_splits, seed)
    # spw=None: each fold computes scale_pos_weight from its own training rows.
    fp = build_fit_predict(model, None, seed, params)
    return cross_val_oof(X, y, fold, fp, n_splits)


def nested_oof_predict(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                       param_grid: Optional[List[Dict]] = None,
                       outer_splits: int = 5, inner_splits: int = 3,
                       seed: int = 0, model: str = "auto"):
    """Hyperparameter search with nested segment-grouped CV.

    For each outer fold, an inner grouped CV on the outer-train rows picks the
    ``param_grid`` entry with the best PR-AUC. That setting is refit on the whole
    outer-train set and predicts the outer-test fold, so test rows are never used
    for tuning. ``scale_pos_weight`` comes from the outer-train rows. The inner
    search reuses that value across inner folds, which can affect which setting
    wins but not the outer-test predictions.

    Returns ``(oof_predictions, chosen_params_per_outer_fold)``.
    """
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    if _resolve_model(model) == "xgb":
        grid = param_grid if param_grid is not None else DEFAULT_XGB_GRID
    else:
        grid = param_grid if param_grid is not None else [{}]
    outer = grouped_folds(groups, y, outer_splits, seed)
    oof = np.zeros(len(y))
    chosen: List[Dict] = []
    for k in range(outer_splits):
        te = np.where(outer == k)[0]
        tr = np.where(outer != k)[0]
        if len(te) == 0:
            continue
        X_tr, y_tr, g_tr = X[tr], y[tr], groups[tr]
        spw_tr = scale_pos_weight(y_tr)
        inner = grouped_folds(g_tr, y_tr, inner_splits, seed)
        best_params, best_score = grid[0], -np.inf
        for params in grid:
            fp = build_fit_predict(model, spw_tr, seed, params)
            inner_oof = cross_val_oof(X_tr, y_tr, inner, fp, inner_splits)
            score = average_precision(y_tr, inner_oof)
            if score > best_score:
                best_score, best_params = score, params
        # Refit the best setting on the full outer-train set.
        fp = build_fit_predict(model, spw_tr, seed, best_params)
        oof[te] = fp(X_tr, y_tr, X[te])
        chosen.append(best_params)
    return oof, chosen


@dataclass
class EvalResult:
    name: str
    roc_mean: float
    roc_std: float
    pr_mean: float
    pr_std: float
    n_seeds: int

    def __str__(self) -> str:
        return (f"{self.name:<28} ROC {self.roc_mean:.3f}+/-{self.roc_std:.3f}  "
                f"PR {self.pr_mean:.3f}+/-{self.pr_std:.3f}")


def evaluate(panel: Panel, idx: Sequence[int], name: str = "",
             seeds: Sequence[int] = (0, 1, 2, 3, 4), n_splits: int = 5,
             model: str = "auto") -> EvalResult:
    """Mean+/-std ROC-AUC and PR-AUC over ``seeds`` for the feature columns
    ``idx``, under segment-grouped CV."""
    rocs, aps = [], []
    for s in seeds:
        oof = oof_predict(panel.X[:, idx], panel.y, panel.groups, s, n_splits, model)
        rocs.append(roc_auc(panel.y, oof))
        aps.append(average_precision(panel.y, oof))
    return EvalResult(name or f"{len(idx)} features",
                      float(np.mean(rocs)), float(np.std(rocs)),
                      float(np.mean(aps)), float(np.std(aps)), len(seeds))


def ablation_experiments(panel: Panel) -> Dict[str, List[int]]:
    """The paper's feature-group ablation as ``name -> column indices``, in the
    row order of ``results/national/detection.txt`` and Table I: optical only, then
    adding the flow-physics, topology and facility-pressure groups, plus a
    context-only (no optical) reference. The pressure row appears only if
    pressure features are present.

    Other XGBoost versions can shift the absolute AUCs by up to about 0.005. The
    ablation conclusions do not depend on that.
    """
    exps: Dict[str, List[int]] = {
        "optical only": panel.select(keep={"optical"}),
        "optical + physics": panel.select(keep={"optical", "physics"}),
        "optical + topology": panel.select(keep={"optical", "topo"}),
        "all (o+p+t)": panel.select(keep={"optical", "physics", "topo"}),
    }
    if "pressure" in panel.present_groups():
        exps["all + pressure"] = panel.select()
    exps["context only (no optical)"] = panel.select(drop={"optical"})
    return {k: v for k, v in exps.items() if v}
