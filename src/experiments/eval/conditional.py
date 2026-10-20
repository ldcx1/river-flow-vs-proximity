"""Stratified and conditional tests shared by the experiments.

Two questions beyond the flow-vs-Euclidean comparisons:

  1. Does the graph advantage rise with stream order (Strahler)? Reported as a
     monotonic trend across bins, not a per-bin search.
  2. Does the graph add anything beyond geography and covariates? Partial
     Moran's I and a stream variogram (Ver Hoef and Peterson) for
     autocorrelation, incremental AUC for detection.

Pure numpy, plus sklearn for the logistic incremental AUC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..graph import RiverGraph
from ..metrics import roc_auc, spearman_r
from ..osm import haversine_m
from .graph_structure import euclidean_weights_matched, flow_weights, morans_I, permutation_test


# --------------------------------------------------------------------------
# Covariates + Strahler binning
# --------------------------------------------------------------------------
def strahler_bin(order: int) -> str:
    if order is None:
        return "?"
    o = int(order)
    return "1-2 (headwater)" if o <= 2 else ("3-4 (mid)" if o <= 4 else "5+ (large)")


BIN_ORDER = ["1-2 (headwater)", "3-4 (mid)", "5+ (large)"]


def covariate_matrix(graph: RiverGraph, segs: Sequence[str],
                     ag: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Standardised per-segment nuisance covariates: Strahler, log length,
    in-degree, and optionally an agricultural feature ``ag``."""
    cols = []
    for s in segs:
        n = graph[s]
        row = [float(n.get("strahler", 0) or 0),
               np.log1p(n.get("length_m", 0.0)),
               float(len(n.get("up", [])))]
        if ag is not None:
            row.append(float(ag.get(s, 0.0)))
        cols.append(row)
    C = np.array(cols, dtype=float)
    mu, sd = C.mean(0), C.std(0)
    sd[sd == 0] = 1.0
    return (C - mu) / sd


def residualize(y: np.ndarray, C: np.ndarray) -> np.ndarray:
    """OLS residuals of y on [1, C] (remove covariate-explained variance)."""
    y = np.asarray(y, float)
    A = np.hstack([np.ones((len(y), 1)), C])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


# --------------------------------------------------------------------------
# Autocorrelation: partial Moran's I and stream variogram
# --------------------------------------------------------------------------
@dataclass
class PartialMoranResult:
    name: str
    n: int
    I_flow_resid: float
    I_euc_resid: float
    diff: float
    p_diff: float

    def line(self) -> str:
        # p_diff is two-sided, so a significant negative diff means Euclidean wins.
        v = ("ns" if self.p_diff >= 0.05 else
             "FLOW>EUC" if self.diff > 0 else "EUC>FLOW" if self.diff < 0 else "ns")
        return (f"{self.name:<16} n={self.n:<4} partial I_flow={self.I_flow_resid:+.3f} "
                f"I_euc={self.I_euc_resid:+.3f} diff={self.diff:+.3f} p={self.p_diff:.3f} {v}")


def partial_morans(graph, seg_values, covariate_segs_vals, name="", max_dist_km=50.0,
                   n_perm=1500, ag=None) -> PartialMoranResult:
    """Moran's I of covariate-residualised values under flow vs Euclidean
    adjacency. Tests whether flow autocorrelation survives once the covariates
    are removed."""
    segs = [s for s, v in seg_values.items() if s in graph and np.isfinite(v)]
    Wf, segs = flow_weights(graph, segs, max_dist_km)
    deg = np.asarray(Wf.getnnz(axis=1)).ravel()
    keep = [s for i, s in enumerate(segs) if deg[i] > 0]
    Wf, keep = flow_weights(graph, keep, max_dist_km)
    x = np.array([seg_values[s] for s in keep], float)
    C = covariate_matrix(graph, keep, ag)
    r = residualize(x, C)
    from .graph_structure import mean_degree
    # Euclidean control matched to the flow graph's realised mean degree.
    We, _ = euclidean_weights_matched(graph, keep, target_degree=mean_degree(Wf))
    If, Ie = morans_I(r, Wf), morans_I(r, We)
    pd = permutation_test(lambda v: morans_I(v, Wf) - morans_I(v, We), r, n=n_perm)
    return PartialMoranResult(name or "value", len(keep), If, Ie, If - Ie, pd["p"])


def variograms(graph, seg_values, max_dist_km=60.0, n_bins=6, sample=4000, seed=0):
    """Empirical semivariogram gamma(h) = 0.5*mean((x_i-x_j)^2) for flow-connected
    pairs by stream distance vs random pairs by Euclidean distance. Returns
    (bin centres in km, flow gamma, Euclidean gamma).

    Caveat: a flow pair at stream distance h is at most h apart in a straight
    line, so a lower flow gamma(h) is partly mechanical. The pair populations
    also differ (all downstream neighbours vs 6 random others). Treat this as a
    weak supporting statistic, not a result on its own."""
    segs = [s for s, v in seg_values.items() if s in graph and np.isfinite(v)]
    rng = np.random.default_rng(seed)
    if len(segs) > sample:
        segs = list(rng.choice(segs, sample, replace=False))
    val = {s: seg_values[s] for s in segs}
    lat = {s: graph[s]["lat"] for s in segs}
    lon = {s: graph[s]["lon"] for s in segs}
    edges = np.linspace(0, max_dist_km, n_bins + 1)
    flow_sv = [[] for _ in range(n_bins)]
    euc_sv = [[] for _ in range(n_bins)]
    sset = set(segs)
    for s in segs:
        # flow pairs, binned by stream distance
        for rec in graph.walk_downstream(s, 200):
            o = rec["object_id"]
            if o in sset:
                h = rec["flow_distance_m"] / 1000.0
                if h <= max_dist_km:
                    b = min(int(h / max_dist_km * n_bins), n_bins - 1)
                    flow_sv[b].append(0.5 * (val[s] - val[o]) ** 2)
        # euclidean pairs: a few random others, binned by geographic distance
        for o in rng.choice(segs, size=6, replace=False):
            if o == s:
                continue
            h = haversine_m(lat[s], lon[s], lat[o], lon[o]) / 1000.0
            if h <= max_dist_km:
                b = min(int(h / max_dist_km * n_bins), n_bins - 1)
                euc_sv[b].append(0.5 * (val[s] - val[o]) ** 2)
    centers = (edges[:-1] + edges[1:]) / 2
    gf = np.array([np.mean(v) if v else np.nan for v in flow_sv])
    ge = np.array([np.mean(v) if v else np.nan for v in euc_sv])
    return centers, gf, ge


# --------------------------------------------------------------------------
# Detection: incremental AUC of the graph score beyond the baselines
# --------------------------------------------------------------------------
def incremental_auc(y, baseline_cols: List[np.ndarray], graph_col: np.ndarray,
                    groups: np.ndarray, n_splits=5, seed=0, n_boot=2000) -> dict:
    """Grouped out-of-fold logistic AUC for the baseline columns vs baseline plus
    the graph score. The gain is the graph's added detection value. The CI is a
    bootstrap over groups."""
    from sklearn.linear_model import LogisticRegression

    from ..cv import cross_val_oof, grouped_folds
    y = np.asarray(y, float)
    Xb = np.column_stack(baseline_cols)
    Xf = np.column_stack(baseline_cols + [graph_col])
    fold = grouped_folds(groups, y, n_splits, seed)

    def fp(Xtr, ytr, Xva):
        # standardise and impute with training-fold stats only (no leakage)
        mu = np.nanmean(Xtr, 0); mu = np.where(np.isfinite(mu), mu, 0.0)
        sd = np.nanstd(Xtr, 0); sd[sd == 0] = 1
        Xtr = (np.where(np.isfinite(Xtr), Xtr, mu) - mu) / sd
        Xva = (np.where(np.isfinite(Xva), Xva, mu) - mu) / sd
        m = LogisticRegression(max_iter=1000, class_weight="balanced")
        m.fit(Xtr, ytr)
        return m.predict_proba(Xva)[:, 1]
    pb = cross_val_oof(Xb, y, fold, fp, n_splits)
    pf = cross_val_oof(Xf, y, fold, fp, n_splits)
    auc_b, auc_f = roc_auc(y, pb), roc_auc(y, pf)
    uniq = np.unique(groups); rows = {u: np.where(groups == u)[0] for u in uniq}
    rng = np.random.default_rng(seed); boot = []
    for _ in range(n_boot):
        ri = np.concatenate([rows[u] for u in rng.choice(uniq, len(uniq), replace=True)])
        yy = y[ri]
        if yy.sum() in (0, len(yy)):
            continue
        boot.append(roc_auc(yy, pf[ri]) - roc_auc(yy, pb[ri]))
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
    return {"auc_base": float(auc_b), "auc_full": float(auc_f),
            "gain": float(auc_f - auc_b), "lo": float(lo), "hi": float(hi)}


# --------------------------------------------------------------------------
# Monotonic trend across Strahler bins
# --------------------------------------------------------------------------
def monotonic_trend(bin_labels: Sequence[str], bin_values: Sequence[float]) -> dict:
    """Spearman correlation and OLS slope between the ordered Strahler-bin index
    and a per-bin graph-advantage metric. Positive means the advantage rises
    with stream order. No p-value is computed."""
    order = {b: i for i, b in enumerate(BIN_ORDER)}
    xs = np.array([order[b] for b in bin_labels if b in order], float)
    ys = np.array([v for b, v in zip(bin_labels, bin_values) if b in order], float)
    m = np.isfinite(ys)
    xs, ys = xs[m], ys[m]
    if len(xs) < 3:
        return {"rho": float("nan"), "n_bins": int(len(xs)), "slope": float("nan")}
    rho = spearman_r(xs, ys)
    slope = float(np.polyfit(xs, ys, 1)[0])
    return {"rho": rho, "n_bins": int(len(xs)), "slope": slope}
