"""Graph-based anomaly detection suite: scores, synthetic injections and experiments.

The river network gives an expectation for each reach from what flows into it and
what its connected neighbours do. An anomaly is a deviation from that expectation.
Each (segment, month) cell is scored several ways, and we test whether the graph
score separates injected anomalies from distractors better than a per-segment
temporal baseline and a straight-line (Euclidean) neighbour baseline. The
experiments are: residual (local spike vs inherited high value), break (isolated
violation vs propagated plume), recovery (connected event detection), denoise
(corroboration filter) and tracing (source attribution).

Synthetic injection gives exact ground truth and is the main validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..graph import RiverGraph
from ..metrics import average_precision, roc_auc
from ..osm import haversine_km_vec, haversine_m
from ..propagation import compute_field, precompute_flows


# ==========================================================================
# Scores  (higher = more anomalous)
# ==========================================================================
def node_temporal_score(X: np.ndarray) -> np.ndarray:
    """|x - segment's own temporal mean| / std. The per-segment baseline, no graph."""
    mu = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True)
    sd = np.where((sd == 0) | ~np.isfinite(sd), 1.0, sd)
    return np.abs(X - mu) / sd


def graph_expectation_residual(X: np.ndarray, segs: Sequence[str], graph: RiverGraph,
                               discharge=None) -> np.ndarray:
    """|observed - expected from upstream| per cell, using compute_field per month.
    Headwaters have no upstream expectation and stay NaN."""
    R = np.full_like(X, np.nan)
    nseg, nt = X.shape
    flows = precompute_flows(graph, discharge)        # static, so computed once
    for t in range(nt):
        conc = {segs[i]: X[i, t] for i in range(nseg) if np.isfinite(X[i, t])}
        if not conc:
            continue
        field = compute_field(conc, graph, discharge, flows=flows)
        for i in range(nseg):
            if not np.isfinite(X[i, t]):
                continue
            f = field.get(segs[i])
            exp = f.get("expected") if f else None
            if exp is not None:
                R[i, t] = abs(X[i, t] - exp)
    return R


def _flow_neighbor_index(graph: RiverGraph, segs: Sequence[str], targets: Sequence[int],
                         max_dist_km: float) -> Dict[int, List[Tuple[int, float]]]:
    """For each target segment index, its flow (up+down) neighbours within
    max_dist_km, as (neighbor_index, dist_km)."""
    idx = {s: i for i, s in enumerate(segs)}
    out: Dict[int, List[Tuple[int, float]]] = {}
    for i in targets:
        s = segs[i]
        nb = []
        # Deeper downstream (200 hops) than upstream (60): plumes travel far down
        # long rivers, while only nearby tributaries contribute to a detection.
        for rec in graph.walk_downstream(s, 200) + graph.walk_upstream(s, 60):
            j = idx.get(rec["object_id"])
            d = rec["flow_distance_m"] / 1000.0
            if j is not None and d <= max_dist_km:
                nb.append((j, d))
        out[i] = nb
    return out


def _euclidean_neighbor_index(segs, lat, lon, targets, k, with_dist=False):
    """k nearest great-circle neighbours per target.

    ``with_dist=True`` returns (index, dist_km) tuples so the residual can
    decay-weight them like the flow neighbours. The default (bare indices, so an
    unweighted mean) reproduces the published break numbers."""
    out = {}
    for i in targets:
        d = haversine_km_vec(lat[i], lon[i], lat, lon)
        d[i] = np.inf
        kk = min(k, len(d) - 1)   # cap at n-1 so the segment itself is never selected
        sel = list(np.argpartition(d, kk)[:kk])
        out[i] = [(int(j), float(d[j])) for j in sel] if with_dist else sel
    return out


def neighbor_residual_at(X: np.ndarray, neighbors: Dict[int, list], cells: Sequence[Tuple[int, int]],
                         decay_km: Optional[float] = None) -> np.ndarray:
    """Score each (i, t) cell as |x(i,t) - weighted mean of the neighbours' x(., t)|.

    ``neighbors[i]`` holds bare indices (unweighted) or (index, dist_km) tuples,
    weighted by exp(-dist/decay_km) when decay_km is set. NaN if no neighbour has
    data."""
    out = np.full(len(cells), np.nan)
    for c, (i, t) in enumerate(cells):
        nb = neighbors.get(i, [])
        vals, wts = [], []
        for item in nb:
            if isinstance(item, tuple):
                j, d = item
                w = np.exp(-d / decay_km) if decay_km else 1.0
            else:
                j, w = item, 1.0
            if np.isfinite(X[j, t]):
                vals.append(X[j, t]); wts.append(w)
        if vals and np.isfinite(X[i, t]):
            pred = np.average(vals, weights=wts)
            out[c] = abs(X[i, t] - pred)
    return out


# ==========================================================================
# Synthetic injection harness
# ==========================================================================
def _eligible_cells(X, rng, n, exclude):
    fin = np.argwhere(np.isfinite(X))
    rng.shuffle(fin)
    out = []
    for i, t in fin:
        if (i, t) in exclude:
            continue
        out.append((int(i), int(t)))
        if len(out) >= n:
            break
    return out


def inject_local_spikes(X, cells, mag):
    """Add mag * (global std) at each cell. A local spike deviates from both the
    upstream expectation and the neighbours."""
    Xi = X.copy()
    std = np.nanstd(X)
    for i, t in cells:
        Xi[i, t] = Xi[i, t] + mag * std
    return Xi


def inject_inherited_high(X, segs, graph, n, rng):
    """Set reaches to a high value of an upstream neighbour (above the month's 75th
    percentile). The level is high but explained by inheritance, so these cells
    are distractors, not anomalies. Returns (X_injected, cells)."""
    idx = {s: i for i, s in enumerate(segs)}
    Xi = X.copy()
    cells = []
    order = list(range(len(segs)))
    rng.shuffle(order)
    for i in order:
        ups = [idx[u] for u in graph[segs[i]].get("up", []) if u in idx]
        if not ups:
            continue
        for t in rng.permutation(X.shape[1]):
            uv = [X[u, t] for u in ups if np.isfinite(X[u, t])]
            if uv and np.isfinite(X[i, t]):
                hi = max(uv)
                if hi > np.nanpercentile(X[:, t], 75):
                    Xi[i, t] = hi
                    cells.append((i, int(t)))
                    break
        if len(cells) >= n:
            break
    return Xi, cells


def inject_propagated_plume(X, segs, graph, n_sources, mag, decay_km, rng, max_len=6):
    """Spike a source and add an exp(-dist/decay_km) decayed copy to its downstream
    chain, giving a plume consistent with flow. Returns (X_injected, source_cells,
    plume_cells)."""
    idx = {s: i for i, s in enumerate(segs)}
    Xi = X.copy()
    std = np.nanstd(X)
    sources, plume = [], []
    order = list(range(len(segs)))
    rng.shuffle(order)
    for i in order:
        t = int(rng.integers(0, X.shape[1]))
        if not np.isfinite(X[i, t]):
            continue
        Xi[i, t] += mag * std
        sources.append((i, t))
        dist = 0.0
        for rec in graph.walk_downstream(segs[i], max_len):
            j = idx.get(rec["object_id"])
            dist = rec["flow_distance_m"] / 1000.0
            if j is not None and np.isfinite(Xi[j, t]):
                Xi[j, t] += mag * std * np.exp(-dist / decay_km)
                plume.append((j, t))
        if len(sources) >= n_sources:
            break
    return Xi, sources, plume


# ==========================================================================
# Evaluation
# ==========================================================================
@dataclass
class AnomalyResult:
    name: str
    n_pos: int
    n_neg: int
    auc: Dict[str, float] = _field(default_factory=dict)
    ap: Dict[str, float] = _field(default_factory=dict)
    perm_p: Dict[str, float] = _field(default_factory=dict)
    boot_delta: Dict[str, dict] = _field(default_factory=dict)   # graph - baseline AUC
    graph_method: str = "graph"
    # Share of cells dropped because some score was undefined (e.g. the graph
    # residual at a headwater). AUCs only cover cells where every method scores.
    frac_dropped: float = 0.0

    def summary(self) -> str:
        lines = [f"{self.name}  (pos={self.n_pos}, neg={self.n_neg}, "
                 f"dropped={self.frac_dropped:.0%} of cells - scores undefined, e.g. headwaters)\n",
                 f"  {'method':<22}{'ROC-AUC':>9}{'PR-AUC':>9}{'perm p':>9}"]
        for m in self.auc:
            lines.append(f"  {m:<22}{self.auc[m]:>9.3f}{self.ap[m]:>9.3f}"
                         f"{self.perm_p.get(m, float('nan')):>9.3f}")
        lines.append(f"\n  {self.graph_method} vs baselines (ROC-AUC delta, segment bootstrap 95% CI):")
        for b, d in self.boot_delta.items():
            v = "GRAPH WINS" if d["lo"] > 0 else ("baseline wins" if d["hi"] < 0 else "inconclusive")
            lines.append(f"    vs {b:<20} {d['mean']:+.3f}  [{d['lo']:+.3f}, {d['hi']:+.3f}]  {v}")
        return "\n".join(lines)


def _auc_perm(y, s, seed=0, n=1000):
    m = np.isfinite(s)
    y2, s2 = y[m], s[m]
    if y2.sum() == 0 or y2.sum() == len(y2):
        return float("nan"), float("nan"), float("nan")
    obs = roc_auc(y2, s2)
    ap = average_precision(y2, s2)
    rng = np.random.default_rng(seed)
    null = np.array([roc_auc(rng.permutation(y2), s2) for _ in range(n)])
    p = float((np.sum(np.abs(null - 0.5) >= abs(obs - 0.5)) + 1) / (n + 1))
    return float(obs), float(ap), p


def run_residual(graph: RiverGraph, segs, X, n_local=300, n_inherited=300, n_bg=1500,
           mag=4.0, k_euc=None, seed=0) -> Tuple[AnomalyResult, Dict[str, float]]:
    """Residual experiment: graph-expectation residual vs per-segment temporal vs
    Euclidean-neighbour scores.

    Positives are local spikes, distractors are inherited high values plus
    background cells. Returns the full result and the local-vs-inherited ROC-AUC
    per method.
    """
    rng = np.random.default_rng(seed)
    Xi, inh = inject_inherited_high(X, segs, graph, n_inherited, rng)
    local = _eligible_cells(Xi, rng, n_local, exclude=set(inh))
    Xi = inject_local_spikes(Xi, local, mag)
    bg = _eligible_cells(Xi, rng, n_bg, exclude=set(inh) | set(local))

    cells = local + inh + bg
    label = [1] * len(local) + [0] * len(inh) + [0] * len(bg)   # binary truth
    kind = ["local"] * len(local) + ["inherited"] * len(inh) + ["bg"] * len(bg)

    Rg = graph_expectation_residual(Xi, segs, graph)
    Zn = node_temporal_score(Xi)
    lat = np.array([graph[s]["lat"] for s in segs])
    lon = np.array([graph[s]["lon"] for s in segs])
    flow_nb = _flow_neighbor_index(graph, segs, sorted({i for i, _ in cells}), 60.0)
    _deg = [len(v) for v in flow_nb.values() if v]
    k = k_euc or max(2, int(np.median(_deg)) if _deg else 4)
    euc_nb = _euclidean_neighbor_index(segs, lat, lon, sorted({i for i, _ in cells}), k)

    s_graph = np.array([Rg[i, t] for i, t in cells])
    s_node = np.array([Zn[i, t] for i, t in cells])
    s_spatial = neighbor_residual_at(Xi, euc_nb, cells)

    # keep cells where all three scores exist (drops headwaters)
    ok = np.isfinite(s_graph) & np.isfinite(s_node) & np.isfinite(s_spatial)
    cells = [c for c, k_ in zip(cells, ok) if k_]
    y = np.array([l for l, k_ in zip(label, ok) if k_], float)
    kind = [kk for kk, k_ in zip(kind, ok) if k_]
    scores = {"graph_expectation": s_graph[ok], "node_temporal": s_node[ok],
              "spatial_euclidean": s_spatial[ok]}
    res = evaluate_scores(cells, y, scores, "graph_expectation", "residual graph-residual", seed)
    res.frac_dropped = float((~ok).mean()) if len(ok) else 0.0
    # local spikes vs inherited highs only, without background cells
    kk = np.array(kind)
    contrast = {}
    for m, s in scores.items():
        pos = s[kk == "local"]; neg = s[kk == "inherited"]
        yy = np.r_[np.ones(len(pos)), np.zeros(len(neg))]; ss = np.r_[pos, neg]
        mfin = np.isfinite(ss)
        contrast[m] = float(roc_auc(yy[mfin], ss[mfin])) if yy[mfin].sum() not in (0, mfin.sum()) else float("nan")
    return res, contrast


def run_break(graph: RiverGraph, segs, X, n_viol=300, n_sources=120, n_bg=1500,
           mag=4.0, decay_km=30.0, seed=0, return_scores=False, inject_decay_km=None,
           euc_decay_km=None):
    """Break experiment (neighbour-break test): flow-neighbour residual vs
    Euclidean-neighbour and per-segment temporal scores.

    Positives are isolated spikes (violations of flow coherence), distractors are
    propagated plumes plus background cells. Returns the result and the
    violation-vs-plume ROC-AUC per method (plus raw scores if ``return_scores``).

    ``inject_decay_km`` (default ``decay_km``) sets the plume decay separately from
    the score's decay, to check the result does not rely on the plume being made
    with the score's own decay model.

    ``euc_decay_km`` decay-weights the Euclidean neighbour mean like the flow one.
    The default ``None`` (unweighted) reproduces the published numbers.
    """
    rng = np.random.default_rng(seed)
    inj_decay = inject_decay_km if inject_decay_km is not None else decay_km
    Xi, sources, plume = inject_propagated_plume(X, segs, graph, n_sources, mag, inj_decay, rng)
    plume_cells = list({*plume, *sources})
    viol = _eligible_cells(Xi, rng, n_viol, exclude=set(plume_cells))
    Xi = inject_local_spikes(Xi, viol, mag)
    bg = _eligible_cells(Xi, rng, n_bg, exclude=set(plume_cells) | set(viol))

    cells = viol + plume_cells + bg
    label = [1] * len(viol) + [0] * len(plume_cells) + [0] * len(bg)
    kind = ["violation"] * len(viol) + ["plume"] * len(plume_cells) + ["bg"] * len(bg)

    Zn = node_temporal_score(Xi)
    lat = np.array([graph[s]["lat"] for s in segs])
    lon = np.array([graph[s]["lon"] for s in segs])
    targets = sorted({i for i, _ in cells})
    flow_nb = _flow_neighbor_index(graph, segs, targets, 60.0)
    _deg = [len(v) for v in flow_nb.values() if v]
    k = max(2, int(np.median(_deg)) if _deg else 4)
    euc_nb = _euclidean_neighbor_index(segs, lat, lon, targets, k,
                                       with_dist=euc_decay_km is not None)

    s_graph = neighbor_residual_at(Xi, flow_nb, cells, decay_km=decay_km)
    # The published default weights the flow neighbours by distance but not the
    # Euclidean ones. With both weighted (euc_decay_km=decay_km) the graph still
    # wins, +0.006 [0.001, 0.011] (results_data.ROBUST["break_euc_decay"]).
    s_spatial = neighbor_residual_at(Xi, euc_nb, cells, decay_km=euc_decay_km)
    s_node = np.array([Zn[i, t] for i, t in cells])

    ok = np.isfinite(s_graph) & np.isfinite(s_node) & np.isfinite(s_spatial)
    cells = [c for c, k_ in zip(cells, ok) if k_]
    y = np.array([l for l, k_ in zip(label, ok) if k_], float)
    kind = [kk for kk, k_ in zip(kind, ok) if k_]
    scores = {"graph_neighbors": s_graph[ok], "node_temporal": s_node[ok],
              "spatial_euclidean": s_spatial[ok]}
    res = evaluate_scores(cells, y, scores, "graph_neighbors", "break coherence-violation", seed)
    res.frac_dropped = float((~ok).mean()) if len(ok) else 0.0
    kk = np.array(kind)
    contrast = {}
    for m, s in scores.items():
        pos = s[kk == "violation"]; neg = s[kk == "plume"]
        yy = np.r_[np.ones(len(pos)), np.zeros(len(neg))]; ss = np.r_[pos, neg]
        mfin = np.isfinite(ss)
        contrast[m] = float(roc_auc(yy[mfin], ss[mfin])) if yy[mfin].sum() not in (0, mfin.sum()) else float("nan")
    if return_scores:
        groups = np.array([i for i, _ in cells])
        return res, contrast, {"scores": scores, "y": y, "groups": groups,
                               "strahler": np.array([graph[segs[i]].get("strahler", 0) or 0
                                                     for i, _ in cells])}
    return res, contrast


# ==========================================================================
# recovery: detecting connected multi-reach events
# ==========================================================================
def _union_find(n):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)
    return find, union


def _flow_components(anom_idx, graph, segs, max_hops=4):
    """Connected components of anomalous reaches, linking two reaches when one is
    within max_hops downstream of the other."""
    aset = set(anom_idx)
    pos = {a: p for p, a in enumerate(anom_idx)}
    idx = {s: i for i, s in enumerate(segs)}
    find, union = _union_find(len(anom_idx))
    for a in anom_idx:
        for rec in graph.walk_downstream(segs[a], max_hops):
            j = idx.get(rec["object_id"])
            if j in aset:
                union(pos[a], pos[j])
    comps: Dict[int, List[int]] = {}
    for a in anom_idx:
        comps.setdefault(find(pos[a]), []).append(a)
    return list(comps.values())


def _geo_components(anom_idx, lat, lon, radius_km=6.0):
    """Connected components of anomalous reaches, linking pairs within radius_km.

    Reaches are bucketed on a grid so each one only tests nearby buckets."""
    find, union = _union_find(len(anom_idx))
    # Cells are radius/60 degrees. A 5x5 window then covers every pair within
    # radius_km while cos(lat) >= 1/2 (|lat| below about 60 deg), which holds for
    # Germany and Romania. Higher latitudes would need a wider longitude window.
    cell = max(radius_km / 60.0, 1e-6)
    buckets: Dict[Tuple[int, int], List[int]] = {}
    la = [lat[a] for a in anom_idx]
    lo = [lon[a] for a in anom_idx]
    for p in range(len(anom_idx)):
        buckets.setdefault((int(la[p] // cell), int(lo[p] // cell)), []).append(p)
    for p in range(len(anom_idx)):
        cb = (int(la[p] // cell), int(lo[p] // cell))
        for da in (-2, -1, 0, 1, 2):
            for do in (-2, -1, 0, 1, 2):
                for q in buckets.get((cb[0] + da, cb[1] + do), ()):
                    if q > p and haversine_m(la[p], lo[p], la[q], lo[q]) / 1000.0 <= radius_km:
                        union(p, q)
    comps: Dict[int, List[int]] = {}
    for p, a in enumerate(anom_idx):
        comps.setdefault(find(p), []).append(a)
    return list(comps.values())


def _event_pr(components, true_event_idx, min_size=3):
    """Reach-level precision, recall and F1, treating components of size >=
    min_size as detected events. ``true_event_idx`` holds the injected plume reaches."""
    detected = [c for c in components if len(c) >= min_size]
    det_idx = set().union(*detected) if detected else set()
    true = set(true_event_idx)
    if not det_idx:
        return 0.0, 0.0, 0.0
    tp = len(det_idx & true)
    prec = tp / len(det_idx)
    rec = tp / len(true) if true else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


@dataclass
class RecoveryResult:
    n_runs: int
    flow: Dict[str, float] = _field(default_factory=dict)   # mean prec/rec/f1
    euc: Dict[str, float] = _field(default_factory=dict)
    f1_delta: dict = _field(default_factory=dict)           # flow - euc F1, CI

    def summary(self) -> str:
        v = ("GRAPH WINS" if self.f1_delta["lo"] > 0 else
             ("euclidean wins" if self.f1_delta["hi"] < 0 else "inconclusive"))
        return (f"recovery connected-subgraph events ({self.n_runs} runs)\n\n"
                f"  flow components:      prec {self.flow['prec']:.3f}  rec {self.flow['rec']:.3f}  F1 {self.flow['f1']:.3f}\n"
                f"  euclidean components: prec {self.euc['prec']:.3f}  rec {self.euc['rec']:.3f}  F1 {self.euc['f1']:.3f}\n\n"
                f"  flow - euclidean F1: {self.f1_delta['mean']:+.3f} "
                f"[{self.f1_delta['lo']:+.3f}, {self.f1_delta['hi']:+.3f}]  {v}")


def _inject_flow_plumes(Xi, segs, graph, idx, order, n_plumes, plume_len, mag, std):
    """Inject up to ``n_plumes`` flow plumes (true events) into the month vector
    ``Xi`` in place.

    A plume is a source plus its downstream chain, with magnitude decaying as
    ``exp(-0.3*hop)``. Sources are taken from ``order``, skipping NaN segments
    and chains shorter than 3. Returns the indices that received a plume."""
    plume = set()
    placed = 0
    for i in order:
        if placed >= n_plumes:
            break
        if not np.isfinite(Xi[i]):
            continue
        chain = [i] + [idx[r2["object_id"]] for r2 in graph.walk_downstream(segs[i], plume_len)
                       if r2["object_id"] in idx]
        chain = [c for c in chain if np.isfinite(Xi[c])][:plume_len]
        if len(chain) < 3:
            continue
        for d, c in enumerate(chain):
            Xi[c] += mag * std * np.exp(-d * 0.3)
            plume.add(c)
        placed += 1
    return plume


def _inject_geo_clusters(Xi, segs, graph, idx, order, lat, lon, n_geo, mag, std, exclude):
    """Inject up to ``n_geo`` compact clusters that are not flow-connected (false
    events) into ``Xi`` in place.

    For each seed reach, up to 5 of its 8 nearest reaches are raised, keeping only
    finite ones that are not within 8 hops up or down of it and not in
    ``exclude``. Returns the injected indices."""
    geo = set()
    gplaced = 0
    for i in order:
        if gplaced >= n_geo or not np.isfinite(Xi[i]):
            continue
        d = haversine_km_vec(lat[i], lon[i], lat, lon)
        connected = {idx.get(r2["object_id"]) for r2 in
                     (graph.walk_downstream(segs[i], 8) + graph.walk_upstream(segs[i], 8))}
        near = [j for j in np.argsort(d)[:8]
                if np.isfinite(Xi[j]) and j not in connected and j not in exclude][:5]
        if len(near) < 3:
            continue
        for j in near:
            Xi[j] += mag * std
            geo.add(j)
        gplaced += 1
    return geo


def run_recovery(graph: RiverGraph, segs, X, n_runs=20, n_plumes=4, n_geo=4,
           plume_len=6, mag=5.0, q=98.0, seed=0) -> RecoveryResult:
    """Recovery experiment: in one random month per run, inject flow plumes (true
    events) and compact but disconnected clusters (false events). Flag the top
    (100-q)% of reaches, then compare event detection by flow components vs
    geographic components (F1 against plume reaches)."""
    lat = np.array([graph[s]["lat"] for s in segs])
    lon = np.array([graph[s]["lon"] for s in segs])
    idx = {s: i for i, s in enumerate(segs)}
    std = np.nanstd(X)
    fF1, eF1, fpr, frc, epr, erc = [], [], [], [], [], []
    for r in range(n_runs):
        rng = np.random.default_rng(seed + r)
        t = int(rng.integers(0, X.shape[1]))
        Xi = X[:, t].copy()
        order = rng.permutation(len(segs))
        plume_true = _inject_flow_plumes(Xi, segs, graph, idx, order,
                                         n_plumes, plume_len, mag, std)
        _inject_geo_clusters(Xi, segs, graph, idx, order, lat, lon, n_geo, mag, std,
                             exclude=plume_true)
        # anomalous set: reaches at or above the q-th percentile this month
        thr = np.nanpercentile(Xi, q)
        anom = [i for i in range(len(segs)) if np.isfinite(Xi[i]) and Xi[i] >= thr]
        if len(anom) < 3 or not plume_true:
            continue
        fc = _flow_components(anom, graph, segs)
        gc = _geo_components(anom, lat, lon)
        p1, r1, f1a = _event_pr(fc, plume_true)
        p2, r2v, f2 = _event_pr(gc, plume_true)
        fpr.append(p1); frc.append(r1); fF1.append(f1a)
        epr.append(p2); erc.append(r2v); eF1.append(f2)
    fF1, eF1 = np.array(fF1), np.array(eF1)
    if len(fF1) == 0:                       # no usable runs (e.g. n_plumes=0)
        nan3 = {"prec": float("nan"), "rec": float("nan"), "f1": float("nan")}
        return RecoveryResult(n_runs=0, flow=dict(nan3), euc=dict(nan3),
                        f1_delta={"mean": 0.0, "lo": 0.0, "hi": 0.0})
    diffs = fF1 - eF1
    rng = np.random.default_rng(seed)
    boot = np.array([diffs[rng.integers(0, len(diffs), len(diffs))].mean()
                     for _ in range(2000)]) if len(diffs) else np.array([0.0])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return RecoveryResult(
        n_runs=len(fF1),
        flow={"prec": float(np.mean(fpr)), "rec": float(np.mean(frc)), "f1": float(fF1.mean())},
        euc={"prec": float(np.mean(epr)), "rec": float(np.mean(erc)), "f1": float(eF1.mean())},
        f1_delta={"mean": float(diffs.mean()), "lo": float(lo), "hi": float(hi)})


# ==========================================================================
# denoise: the graph as a filter for flagged anomalies
# ==========================================================================
@dataclass
class DenoiseResult:
    n_runs: int
    raw: Dict[str, float] = _field(default_factory=dict)
    flow: Dict[str, float] = _field(default_factory=dict)
    euc: Dict[str, float] = _field(default_factory=dict)
    prec_delta: dict = _field(default_factory=dict)        # flow - euc precision, CI

    def summary(self) -> str:
        v = ("GRAPH WINS" if self.prec_delta["lo"] > 0 else
             ("euclidean wins" if self.prec_delta["hi"] < 0 else "inconclusive"))
        return (f"denoise anomaly validator / denoiser ({self.n_runs} runs)\n\n"
                f"  raw (no filter):   prec {self.raw['prec']:.3f}  rec {self.raw['rec']:.3f}\n"
                f"  flow-corroborated: prec {self.flow['prec']:.3f}  rec {self.flow['rec']:.3f}  "
                f"noise-reject {self.flow['nrej']:.3f}\n"
                f"  euc-corroborated:  prec {self.euc['prec']:.3f}  rec {self.euc['rec']:.3f}  "
                f"noise-reject {self.euc['nrej']:.3f}\n\n"
                f"  flow - euc precision: {self.prec_delta['mean']:+.3f} "
                f"[{self.prec_delta['lo']:+.3f}, {self.prec_delta['hi']:+.3f}]  {v}")


def run_denoise(graph: RiverGraph, segs, X, n_runs=20, n_plumes=4, n_geo=4,
           plume_len=6, mag=5.0, q=98.0, seed=0) -> DenoiseResult:
    """Denoise experiment: keep a flagged reach only if a flow neighbour is also
    flagged, and compare with the same filter using geographic neighbours. Same
    injections as :func:`run_recovery` (plumes are true, clusters are false)."""
    lat = np.array([graph[s]["lat"] for s in segs])
    lon = np.array([graph[s]["lon"] for s in segs])
    idx = {s: i for i, s in enumerate(segs)}
    std = np.nanstd(X)
    rp, rr, fp, fr, fn, ep, er, en = ([] for _ in range(8))
    fprec_list, eprec_list = [], []
    for r in range(n_runs):
        rng = np.random.default_rng(seed + r)
        t = int(rng.integers(0, X.shape[1]))
        Xi = X[:, t].copy()
        order = rng.permutation(len(segs))
        plume = _inject_flow_plumes(Xi, segs, graph, idx, order,
                                    n_plumes, plume_len, mag, std)
        geonoise = _inject_geo_clusters(Xi, segs, graph, idx, order, lat, lon, n_geo,
                                        mag, std, exclude=plume)
        thr = np.nanpercentile(Xi, q)
        flagged = [i for i in range(len(segs)) if np.isfinite(Xi[i]) and Xi[i] >= thr]
        if len(flagged) < 3 or not plume:
            continue
        # corroborated = in a component of size >= 2
        fkeep = set().union(*[c for c in _flow_components(flagged, graph, segs) if len(c) >= 2] or [set()])
        ekeep = set().union(*[c for c in _geo_components(flagged, lat, lon) if len(c) >= 2] or [set()])
        fl = set(flagged)
        def pr(keep):
            if not keep:
                return 0.0, 0.0
            return len(keep & plume) / len(keep), len(keep & plume) / len(plume)
        rp_, rr_ = len(fl & plume) / len(fl), len(fl & plume) / len(plume)
        fp_, fr_ = pr(fkeep); ep_, er_ = pr(ekeep)
        # noise rejection = fraction of injected cluster reaches not kept
        nrej_f = 1 - (len(fkeep & geonoise) / len(geonoise)) if geonoise else np.nan
        nrej_e = 1 - (len(ekeep & geonoise) / len(geonoise)) if geonoise else np.nan
        rp.append(rp_); rr.append(rr_); fp.append(fp_); fr.append(fr_)
        ep.append(ep_); er.append(er_); fn.append(nrej_f); en.append(nrej_e)
        fprec_list.append(fp_); eprec_list.append(ep_)
    fa, ea = np.array(fprec_list), np.array(eprec_list)
    if len(fa) == 0:                        # no usable runs (e.g. n_plumes=0)
        nan3 = {"prec": float("nan"), "rec": float("nan"), "nrej": float("nan")}
        return DenoiseResult(n_runs=0, raw={"prec": float("nan"), "rec": float("nan")},
                        flow=dict(nan3), euc=dict(nan3),
                        prec_delta={"mean": 0.0, "lo": 0.0, "hi": 0.0})
    diffs = fa - ea
    rng = np.random.default_rng(seed)
    boot = np.array([diffs[rng.integers(0, len(diffs), len(diffs))].mean()
                     for _ in range(2000)]) if len(diffs) else np.array([0.0])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return DenoiseResult(
        n_runs=len(fprec_list),
        raw={"prec": float(np.mean(rp)), "rec": float(np.mean(rr))},
        flow={"prec": float(np.mean(fp)), "rec": float(np.mean(fr)), "nrej": float(np.nanmean(fn))},
        euc={"prec": float(np.mean(ep)), "rec": float(np.mean(er)), "nrej": float(np.nanmean(en))},
        prec_delta={"mean": float(diffs.mean()), "lo": float(lo), "hi": float(hi)})


# ==========================================================================
# tracing: recovering the source of a propagated event
# ==========================================================================
@dataclass
class TracingResult:
    n_events: int
    top1: Dict[str, float] = _field(default_factory=dict)
    mrr: Dict[str, float] = _field(default_factory=dict)
    top1_delta: dict = _field(default_factory=dict)        # spatio-temporal - static

    def summary(self) -> str:
        v = ("SPATIO-TEMPORAL WINS" if self.top1_delta["lo"] > 0 else
             ("static wins" if self.top1_delta["hi"] < 0 else "inconclusive"))
        lines = [f"tracing spatio-temporal event attribution ({self.n_events} events)\n"]
        for m in self.top1:
            lines.append(f"  {m:<18} top-1 {self.top1[m]:.3f}   MRR {self.mrr[m]:.3f}")
        lines.append(f"\n  spatio-temporal - static top-1: {self.top1_delta['mean']:+.3f} "
                     f"[{self.top1_delta['lo']:+.3f}, {self.top1_delta['hi']:+.3f}]  {v}")
        return "\n".join(lines)


def run_tracing(graph: RiverGraph, segs, X, n_events=120, plume_len=8, mag=6.0,
           decay_km=30.0, seed=0, inject_hop_decay=0.3) -> TracingResult:
    """Tracing experiment: inject a source spike that propagates downstream, then
    rank candidate sources from the most-downstream detection.

    Methods: static upstream tracing, spatio-temporal (tracing restricted to
    candidates that are themselves in the month's top 10%), Euclidean (the same
    candidates ranked by distance) and a reversed-direction sanity check."""
    from ..propagation import compute_field, trace_sources
    idx = {s: i for i, s in enumerate(segs)}
    lat = np.array([graph[s]["lat"] for s in segs])
    lon = np.array([graph[s]["lon"] for s in segs])
    flows = precompute_flows(graph)                  # static, so computed once
    std = np.nanstd(X)
    hits = {m: [] for m in ["static", "spatio_temporal", "euclidean", "reversed"]}
    rr = {m: [] for m in hits}
    rng = np.random.default_rng(seed)
    placed = 0
    for i in rng.permutation(len(segs)):
        if placed >= n_events:
            break
        t = int(rng.integers(0, X.shape[1]))
        if not np.isfinite(X[i, t]):
            continue
        chain = [idx[r2["object_id"]] for r2 in graph.walk_downstream(segs[i], plume_len)
                 if r2["object_id"] in idx and np.isfinite(X[idx[r2["object_id"]], t])]
        if len(chain) < 3:
            continue
        Xi = X[:, t].copy()
        Xi[i] += mag * std                       # source
        for d, c in enumerate(chain):
            Xi[c] += mag * std * np.exp(-(d + 1) * inject_hop_decay)
        detect = chain[-1]                        # most-downstream reach is the detection
        conc = {segs[k]: Xi[k] for k in range(len(segs)) if np.isfinite(Xi[k])}
        field = compute_field(conc, graph, flows=flows)
        srcs = trace_sources(segs[detect], field, graph, top_k=30)
        ranked = [s["object_id"] for s in srcs]
        def rank_of(lst, target):
            return (lst.index(target) + 1) if target in lst else None
        tgt = segs[i]
        r_static = rank_of(ranked, tgt)
        # spatio-temporal: keep only candidates that are themselves anomalous this month
        thr = np.nanpercentile(Xi, 90)
        ranked_st = [o for o in ranked if np.isfinite(Xi[idx[o]]) and Xi[idx[o]] >= thr]
        r_st = rank_of(ranked_st, tgt)
        # Euclidean: rank the same trace_sources candidates by distance to the
        # detection, so both methods see identical candidates. Ranking the 30
        # globally nearest segments instead would often exclude the true source.
        dd = haversine_km_vec(lat[detect], lon[detect], lat, lon)
        eu = sorted(ranked, key=lambda o: dd[idx[o]])
        r_eu = rank_of(eu, tgt)
        # Reversed direction: rank reaches downstream of the detection. The source
        # is upstream, so top-1 and MRR are 0 by construction. This is only a
        # sanity check, not a ranking baseline (unlike the reversed control in
        # attribution.py).
        down = [r2["object_id"] for r2 in graph.walk_downstream(segs[detect], plume_len)
                if r2["object_id"] in idx and np.isfinite(Xi[idx[r2["object_id"]]])]
        rev = sorted(down, key=lambda o: -Xi[idx[o]])
        r_rev = rank_of(rev, tgt)
        for m, rk in [("static", r_static), ("spatio_temporal", r_st),
                      ("euclidean", r_eu), ("reversed", r_rev)]:
            hits[m].append(1.0 if rk == 1 else 0.0)
            rr[m].append(1.0 / rk if rk else 0.0)
        placed += 1
    res = TracingResult(n_events=placed)
    for m in hits:
        res.top1[m] = float(np.mean(hits[m])) if hits[m] else float("nan")
        res.mrr[m] = float(np.mean(rr[m])) if rr[m] else float("nan")
    if placed == 0:                          # no events injected (e.g. chains too short)
        res.top1_delta = {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
        return res
    a = np.array(hits["spatio_temporal"]); b = np.array(hits["static"])
    diffs = a - b
    boot = np.array([diffs[rng.integers(0, len(diffs), len(diffs))].mean()
                     for _ in range(2000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    res.top1_delta = {"mean": float(diffs.mean()), "lo": float(lo), "hi": float(hi)}
    return res


def evaluate_scores(cells, y, scores: Dict[str, np.ndarray], graph_method: str,
                    name: str, seed: int = 0, n_boot: int = 2000) -> AnomalyResult:
    """Per-method ROC-AUC, PR-AUC and permutation p-value, plus graph-vs-baseline
    ROC-AUC deltas with a segment cluster bootstrap.

    ``cells``, ``y`` (1 = anomaly) and each array in ``scores`` are parallel."""
    y = np.asarray(y, dtype=float)
    seg_of = np.array([i for i, _ in cells])
    res = AnomalyResult(name=name, n_pos=int(y.sum()), n_neg=int((y == 0).sum()),
                        graph_method=graph_method)
    for m, s in scores.items():
        auc, ap, p = _auc_perm(y, np.asarray(s, float), seed=seed)
        res.auc[m], res.ap[m], res.perm_p[m] = auc, ap, p

    gs = np.asarray(scores[graph_method], float)
    uniq = np.unique(seg_of)
    rows_by = {u: np.where(seg_of == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    for b, bs in scores.items():
        if b == graph_method:
            continue
        bs = np.asarray(bs, float)
        boot = []
        for _ in range(n_boot):
            samp = rng.choice(uniq, size=len(uniq), replace=True)
            ri = np.concatenate([rows_by[u] for u in samp])
            yy = y[ri]
            mg = np.isfinite(gs[ri]); mb = np.isfinite(bs[ri])
            if yy[mg].sum() in (0, mg.sum()) or yy[mb].sum() in (0, mb.sum()):
                continue
            boot.append(roc_auc(yy[mg], gs[ri][mg]) - roc_auc(yy[mb], bs[ri][mb]))
        if boot:
            lo, hi = np.percentile(boot, [2.5, 97.5])
            res.boot_delta[b] = {"mean": float(np.mean(boot)), "lo": float(lo), "hi": float(hi)}
    return res
