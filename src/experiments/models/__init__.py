"""Gradient-boosted classifier, its cross-validation driver, and the feature
ablation runner behind the paper's tables."""

from .train import (
    DEFAULT_XGB_GRID,
    XGB_PARAMS,
    EvalResult,
    ablation_experiments,
    build_fit_predict,
    evaluate,
    logreg_fit_predict,
    make_fit_predict,
    nested_oof_predict,
    oof_predict,
    scale_pos_weight,
)

__all__ = [
    "XGB_PARAMS", "DEFAULT_XGB_GRID", "EvalResult", "scale_pos_weight",
    "make_fit_predict", "logreg_fit_predict", "build_fit_predict",
    "oof_predict", "nested_oof_predict", "evaluate", "ablation_experiments",
]
