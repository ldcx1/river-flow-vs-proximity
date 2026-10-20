"""Co-fluctuation test: do anomalies travel along the river network?

Main test: after removing the shared seasonal and regional signal, do the monthly
anomalies of flow-connected segment pairs correlate more than those of
distance-matched pairs that are near but not connected? This looks at temporal
fluctuations (e.g. storm pulses), unlike the mean-level spatial autocorrelation
in ``eval/autocorrelation.py``.

Secondary test (limited by the monthly resolution): on long flow pairs, does the
downstream series lag the upstream one (positive lagged cross-correlation
asymmetry)?

Inference uses a bootstrap over source segments and a node-permutation null. Two
deseasonings are supported: network common mode and per-segment climatology.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..graph import RiverGraph
from ..osm import haversine_km_vec, haversine_m


# --------------------------------------------------------------------------
# Series assembly + deseasoning
# --------------------------------------------------------------------------
def load_series(rows: Sequence[dict], value_col: str, graph: RiverGraph
                ) -> Tuple[List[str], List[str], np.ndarray]:
    """Rows with object_id/ym/<value_col> -> (segments, sorted months, X[n_seg, n_mon]).

    Missing values are NaN. Only segments present in the graph are kept."""
    months = sorted({r["ym"] for r in rows})
    mi = {m: j for j, m in enumerate(months)}
    by_seg: Dict[str, Dict[int, float]] = {}
    for r in rows:
        s = r["object_id"]
        if s not in graph:
            continue
        try:
            v = float(r[value_col])
        except (ValueError, KeyError, TypeError):
            continue
        by_seg.setdefault(s, {})[mi[r["ym"]]] = v
    segs = sorted(by_seg)
    X = np.full((len(segs), len(months)), np.nan)
    for i, s in enumerate(segs):
        for j, v in by_seg[s].items():
            X[i, j] = v
    return segs, months, X


def common_mode_anomaly(X: np.ndarray) -> np.ndarray:
    """Subtract the per-month mean over all segments. This removes shared
    seasonality and regional weather."""
    mode = np.nanmean(X, axis=0)
    return X - mode[None, :]


def seasonal_anomaly(X: np.ndarray, months: Sequence[str]) -> np.ndarray:
    """Subtract each segment's own month-of-year climatology."""
    mo = np.array([int(m[5:7]) for m in months])
    A = X.copy()
    for m in range(1, 13):
        cols = np.where(mo == m)[0]
        if len(cols) == 0:
            continue
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN slices -> NaN clim
            clim = np.nanmean(X[:, cols], axis=1)             # per-segment mean for this month
        A[:, cols] = X[:, cols] - clim[:, None]
    return A


# --------------------------------------------------------------------------
# Pair correlation
# --------------------------------------------------------------------------
def pair_corr(a: np.ndarray, b: np.ndarray, n_min: int, lag: int = 0) -> float:
    """Pearson correlation of a(t) and b(t+lag) over months where both exist.
    NaN with fewer than n_min overlapping points or zero variance."""
    if lag > 0:
        a2, b2 = a[:-lag], b[lag:]
    elif lag < 0:
        a2, b2 = a[-lag:], b[:lag]
    else:
        a2, b2 = a, b
    m = np.isfinite(a2) & np.isfinite(b2)
    if m.sum() < n_min:
        return np.nan
    av, bv = a2[m] - a2[m].mean(), b2[m] - b2[m].mean()
    denom = np.sqrt((av ** 2).sum() * (bv ** 2).sum())
    return float((av @ bv) / denom) if denom > 0 else np.nan


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2, degrees in [0, 360)."""
    import math
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(x, y)) % 360.0


def _ang_diff(a: float, b: float) -> float:
    """Smallest absolute angular difference between two bearings (0-180)."""
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


# --------------------------------------------------------------------------
# Pair construction
# --------------------------------------------------------------------------
def _build_pairs(graph, segs, anom, n_min, max_dist_km, sample, seed,
                 source_filter=None, long_km=None,
                 orient_match=False, orient_tol_deg=45.0):
    idx = {s: i for i, s in enumerate(segs)}
    lat = np.array([graph[s]["lat"] for s in segs])
    lon = np.array([graph[s]["lon"] for s in segs])
    cov = np.array([np.isfinite(anom[i]).sum() for i in range(len(segs))])
    eligible = cov >= n_min
    elig_idx = np.where(eligible)[0]
    if source_filter is not None:                 # restrict sources to one stratum
        elig_idx = np.array([i for i in elig_idx if i in source_filter])
        if len(elig_idx) == 0:
            return [], [], [], []
    rng = np.random.default_rng(seed)
    src = list(rng.choice(elig_idx, size=min(sample, len(elig_idx)), replace=False))

    flow_pairs, euc_pairs, sources, long_pairs = [], [], [], []   # parallel lists, one entry per source
    for i in src:
        s = segs[i]
        # Walk the distance-ordered downstream path once. Keep the nearest eligible
        # partner for the main test and the farthest one in [long_km, max_dist_km]
        # for the lag test.
        down = graph.walk_downstream(s, 200)
        d_partner, flow_km = None, None
        far_partner, far_km = None, None
        for rec in down:
            fkm = rec["flow_distance_m"] / 1000.0
            if fkm > max_dist_km:
                break                                # distance-ordered -> nothing further qualifies
            j = idx.get(rec["object_id"])
            if j is None or not eligible[j]:
                continue
            if d_partner is None:
                d_partner, flow_km = j, fkm
            if long_km is not None and fkm >= long_km:
                far_partner, far_km = j, fkm
        if d_partner is None:
            continue
        euc_km = haversine_m(lat[i], lon[i], lat[d_partner], lon[d_partner]) / 1000.0
        # Flow-connected segments, excluded from the control. The default path uses
        # asymmetric hop caps (200 down, 80 up), so a far but connected reach can
        # slip into the control. The orient_match path walks deeper both ways.
        if orient_match:
            connected = {idx.get(r["object_id"]) for r in
                         (graph.walk_downstream(s, 400) + graph.walk_upstream(s, 400))}
        else:
            connected = {idx.get(r["object_id"]) for r in
                         (down + graph.walk_upstream(s, 80))}
        connected.discard(None); connected.add(i)
        # Control partner: eligible, not connected, great-circle distance within
        # 30% of the flow pair's straight-line distance.
        dist = haversine_km_vec(lat[i], lon[i], lat, lon)
        cand = [j for j in elig_idx
                if j not in connected and 0.7 * euc_km <= dist[j] <= 1.3 * euc_km]
        if orient_match:
            # Also match the control pair's bearing to the flow pair's. A field that
            # is just smoother along the river axis than across it then affects both
            # pairs equally, separating flow effects from directional anisotropy.
            # Off by default. orient_match=False reproduces the published numbers.
            fb = _bearing_deg(lat[i], lon[i], lat[d_partner], lon[d_partner])
            cand = [j for j in cand
                    if _ang_diff(_bearing_deg(lat[i], lon[i], lat[j], lon[j]), fb)
                    <= orient_tol_deg]
        if not cand:
            continue
        e_partner = min(cand, key=lambda j: abs(dist[j] - euc_km))
        flow_pairs.append((i, d_partner, flow_km, euc_km))
        euc_pairs.append((i, e_partner))
        sources.append(i)
        if far_partner is not None:
            long_pairs.append((i, far_partner, far_km))
    return flow_pairs, euc_pairs, sources, long_pairs


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------
@dataclass
class TransportResult:
    name: str
    deseason: str
    n_pairs: int
    c_flow: float
    c_euc: float
    diff: float
    boot_lo: float
    boot_hi: float
    perm_p: float
    asym_long: float          # mean lag asymmetry on long pairs
    n_long: int
    asym_lo: float = float("nan")   # bootstrap 95% CI over long pairs
    asym_hi: float = float("nan")

    @property
    def significant(self) -> bool:
        """The bootstrap CI is above 0 and the permutation p-value is <= 0.05.

        The bootstrap CI alone can exclude 0 on a purely spatial field with no
        flow, hence the permutation gate. Neither gate separates flow from
        directional anisotropy. Use orient_match=True for that."""
        gate_perm = (not np.isfinite(self.perm_p)) or self.perm_p <= 0.05
        return self.boot_lo > 0 and gate_perm

    def line(self) -> str:
        v = "FLOW>EUC" if self.boot_lo > 0 else ("EUC>FLOW" if self.boot_hi < 0 else "ns")
        return (f"{self.name:<10}{self.deseason:<14} n={self.n_pairs:<4} "
                f"C_flow={self.c_flow:+.3f} C_euc={self.c_euc:+.3f} "
                f"diff={self.diff:+.3f}[{self.boot_lo:+.3f},{self.boot_hi:+.3f}] "
                f"p={self.perm_p:.3f} {v} | H2 asym(long,n={self.n_long})={self.asym_long:+.3f}"
                f"[{self.asym_lo:+.3f},{self.asym_hi:+.3f}]")


def temporal_transport(graph: RiverGraph, segs, anom: np.ndarray, name: str,
                       deseason: str, n_min: int = 12, max_dist_km: float = 60.0,
                       sample: int = 3000, long_km: float = 80.0, max_lag: int = 2,
                       seed: int = 0, n_boot: int = 2000, source_bins=None,
                       n_perm: int = 2000, orient_match: bool = False,
                       orient_tol_deg: float = 45.0) -> TransportResult:
    source_filter = None
    if source_bins is not None:
        from .conditional import strahler_bin
        source_filter = {i for i, s in enumerate(segs)
                         if strahler_bin(graph[s].get("strahler")) in source_bins}
    flow_pairs, euc_pairs, sources, long_pairs = _build_pairs(
        graph, segs, anom, n_min, max_dist_km, sample, seed, source_filter, long_km=long_km,
        orient_match=orient_match, orient_tol_deg=orient_tol_deg)
    n = len(sources)
    if n == 0:
        return TransportResult(name, deseason, 0, *(np.nan,) * 6, np.nan, 0)
    cf = np.array([pair_corr(anom[i], anom[j], n_min) for i, j, _, _ in flow_pairs])
    ce = np.array([pair_corr(anom[i], anom[j], n_min) for i, j in euc_pairs])
    ok = np.isfinite(cf) & np.isfinite(ce)
    cf, ce = cf[ok], ce[ok]
    src_ok = np.array(sources)[ok]
    c_flow, c_euc = float(np.mean(cf)), float(np.mean(ce))
    diff = c_flow - c_euc

    # bootstrap over sources (each source gives one flow and one control correlation)
    rng = np.random.default_rng(seed)
    m = len(cf)
    boot = np.array([(cf[b].mean() - ce[b].mean())
                     for b in (rng.integers(0, m, m) for _ in range(n_boot))])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    # permutation null: shuffle which series is assigned to which segment
    if n_perm:
        perm_diffs = []
        for p in range(n_perm):
            prng = np.random.default_rng(1000 + p)
            perm = prng.permutation(anom.shape[0])
            ap = anom[perm]
            cfp = np.array([pair_corr(ap[i], ap[j], n_min) for i, j, _, _ in flow_pairs])
            cep = np.array([pair_corr(ap[i], ap[j], n_min) for i, j in euc_pairs])
            okp = np.isfinite(cfp) & np.isfinite(cep)
            if not okp.any():
                continue    # no overlapping pairs: skip it rather than add a NaN
                            # that would inflate the p-value denominator
            perm_diffs.append(np.mean(cfp[okp]) - np.mean(cep[okp]))
        perm_diffs = np.array(perm_diffs)
        perm_p = float((np.sum(np.abs(perm_diffs) >= abs(diff)) + 1) / (len(perm_diffs) + 1))
    else:
        perm_p = float("nan")

    # Lag test on the long flow pairs: downstream should lag upstream. Only pairs
    # with lag-0 overlap are used.
    asyms = []
    for i, j, _fkm in long_pairs:
        if not np.isfinite(pair_corr(anom[i], anom[j], n_min, lag=0)):
            continue                              # no temporal overlap at this resolution
        def _c(lag):                              # a lag with too little overlap counts as 0
            c = pair_corr(anom[i], anom[j], max(6, n_min // 2), lag=lag)
            return c if np.isfinite(c) else 0.0
        pos = sum(_c(l) for l in range(1, max_lag + 1))
        neg = sum(_c(-l) for l in range(1, max_lag + 1))
        asyms.append(pos - neg)
    asym = float(np.mean(asyms)) if asyms else np.nan
    # Bootstrap CI of the mean lag asymmetry over long pairs. A directional claim
    # needs it to exclude 0.
    if len(asyms) >= 2:
        a = np.array(asyms)
        rngh = np.random.default_rng(seed + 1)
        boot_a = np.array([a[rngh.integers(0, len(a), len(a))].mean()
                           for _ in range(n_boot)])
        asym_lo = float(np.percentile(boot_a, 2.5))
        asym_hi = float(np.percentile(boot_a, 97.5))
    else:
        asym_lo = asym_hi = float("nan")

    return TransportResult(name, deseason, m, c_flow, c_euc, diff,
                           float(lo), float(hi), perm_p, asym, len(asyms),
                           asym_lo, asym_hi)
