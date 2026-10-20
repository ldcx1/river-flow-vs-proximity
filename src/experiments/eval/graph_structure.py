"""Adjacency matrices, spatial statistics and placebo nulls shared by the
graph-vs-proximity experiments. Pure numpy and RiverGraph, no model.

A Euclidean control must have the same mean degree as the flow graph, or the
comparison is unfair. :func:`euclidean_weights_matched` builds such a control.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from ..graph import RiverGraph
from ..osm import haversine_km_vec


# --------------------------------------------------------------------------
# Adjacency construction over a set of (monitored) segments
# --------------------------------------------------------------------------
def flow_neighbors(graph: RiverGraph, segments: Sequence[str],
                   max_dist_km: float = 50.0, max_hops: int = 80
                   ) -> Dict[str, Dict[str, float]]:
    """For each segment in ``segments``, the other listed segments reachable
    along the flow network (up or down) within ``max_dist_km``, mapped to their
    flow distance in km. Monitored segments are sparse, so "flow-adjacent" means
    reachable within a distance, not a shared edge."""
    sset = set(segments)
    out: Dict[str, Dict[str, float]] = {s: {} for s in segments}
    for s in segments:
        if s not in graph:
            continue
        for rec in graph.walk_downstream(s, max_hops) + graph.walk_upstream(s, max_hops):
            oid = rec["object_id"]
            d_km = rec["flow_distance_m"] / 1000.0
            if oid != s and oid in sset and d_km <= max_dist_km:
                prev = out[s].get(oid)
                if prev is None or d_km < prev:
                    out[s][oid] = d_km
    return out


def flow_weights(graph: RiverGraph, segments: Sequence[str],
                 max_dist_km: float = 50.0, weight: str = "binary"
                 ) -> Tuple[sp.csr_matrix, List[str]]:
    """Symmetric flow-connectivity weight matrix over ``segments`` as sparse CSR
    (a dense matrix does not fit in memory at national scale). ``weight='binary'``
    gives 1 if flow-connected within the distance, ``'inv_dist'`` gives 1/km.
    Returns ``(W, ordered_segments)``."""
    segs = list(segments)
    idx = {s: i for i, s in enumerate(segs)}
    n = len(segs)
    nb = flow_neighbors(graph, segs, max_dist_km)
    # Deduplicate undirected pairs keeping the max weight, then emit each pair
    # symmetrically so coo summation does not double-count.
    pair_w: Dict[Tuple[int, int], float] = {}
    for s, neigh in nb.items():
        i = idx[s]
        for t, d in neigh.items():
            j = idx[t]
            if i == j:
                continue
            key = (i, j) if i < j else (j, i)
            w = 1.0 if weight == "binary" else (1.0 / max(d, 1e-3))
            pair_w[key] = max(pair_w.get(key, 0.0), w)
    rows, cols, data = [], [], []
    for (i, j), w in pair_w.items():
        rows += [i, j]; cols += [j, i]; data += [w, w]
    W = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return W, segs


def euclidean_weights(graph: RiverGraph, segments: Sequence[str], k: int
                      ) -> Tuple[sp.csr_matrix, List[str]]:
    """Symmetric binary k-nearest-neighbour matrix (sparse CSR) over ``segments``
    by haversine distance between centroids. Returns ``(W, ordered_segments)``.

    An edge is kept if either endpoint is in the other's kNN, so the realised
    mean degree exceeds ``k`` when neighbour relations are not mutual (by about
    40-50% at ``k>=2``). For a control matched on realised degree, use
    :func:`euclidean_weights_matched`."""
    segs = list(segments)
    n = len(segs)
    lat = np.array([graph[s]["lat"] for s in segs])
    lon = np.array([graph[s]["lon"] for s in segs])
    kk = min(k, n - 1)
    rows, cols = [], []
    for i in range(n):
        d = haversine_km_vec(lat[i], lon[i], lat, lon)
        d[i] = np.inf
        nn = np.argpartition(d, kk)[:kk]
        rows.extend([i] * len(nn)); cols.extend(nn.tolist())
        rows.extend(nn.tolist()); cols.extend([i] * len(nn))   # OR-symmetrise
    W = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    W.data[:] = 1.0                    # clamp OR-symmetrisation duplicates to binary
    return W, segs


def euclidean_weights_matched(graph: RiverGraph, segments: Sequence[str],
                              target_degree: float, k_max: int = 8
                              ) -> Tuple[sp.csr_matrix, List[str]]:
    """Euclidean kNN control whose realised mean degree is closest to
    ``target_degree`` (usually the flow graph's mean degree on the same nodes).

    The symmetric kNN degree exceeds ``k``, so ``k=round(target_degree)`` would
    over-connect the control and understate a flow advantage. This searches
    ``k`` in ``1..k_max`` and stops once the degree overshoots, since it does
    not decrease with ``k``. Returns ``(W, ordered_segments)``."""
    segs = list(segments)
    kmax = max(1, min(k_max, len(segs) - 1))
    best_gap, best_W = None, None
    for k in range(1, kmax + 1):
        W, _ = euclidean_weights(graph, segs, k)
        md = mean_degree(W)
        gap = abs(md - target_degree)
        if best_gap is None or gap < best_gap:
            best_gap, best_W = gap, W
        if md >= target_degree:   # larger k only overshoots further
            break
    return best_W, segs


def mean_degree(W) -> float:
    """Mean number of neighbours per node. Works on sparse CSR or a dense array."""
    if sp.issparse(W):
        return float(np.asarray(W.getnnz(axis=1)).mean())
    return float((np.asarray(W) > 0).sum(axis=1).mean())


# --------------------------------------------------------------------------
# Spatial statistics
# --------------------------------------------------------------------------
def morans_I(values: Sequence[float], W: np.ndarray) -> float:
    """Global Moran's I of ``values`` under weight matrix ``W``. +1 = strong
    positive autocorrelation, ~0 = none, <0 = anti. NaN if W or variance is 0."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    z = x - x.mean()
    S0 = W.sum()
    den = (z ** 2).sum()
    if S0 == 0 or den == 0:
        return float("nan")
    num = z @ (W @ z)
    return float((n / S0) * (num / den))


def join_count_bb(labels: Sequence[int], W: np.ndarray) -> float:
    """Number of "both-positive" joins (BB) under binary ``W``: 0.5 * y' W y.
    Use with a permutation test for significance (planted clusters raise BB)."""
    y = np.asarray(labels, dtype=float)
    return float(0.5 * (y @ (W @ y)))


# --------------------------------------------------------------------------
# Placebo / null machinery
# --------------------------------------------------------------------------
def permutation_test(stat_of_x: Callable[[np.ndarray], float], x: Sequence[float],
                     n: int = 2000, seed: int = 0) -> dict:
    """Permutation test for a spatial statistic. Shuffles ``x`` across nodes
    ``n`` times with the weights fixed and compares the observed statistic to
    that null. Returns observed, null mean and sd, z, and a two-sided p."""
    rng = np.random.default_rng(seed)
    xx = np.asarray(x, dtype=float)
    obs = stat_of_x(xx)
    null = np.array([stat_of_x(rng.permutation(xx)) for _ in range(n)])
    mu, sd = float(null.mean()), float(null.std(ddof=1))
    z = (obs - mu) / sd if sd > 0 else float("nan")
    p = float((np.sum(np.abs(null - mu) >= abs(obs - mu)) + 1) / (n + 1))
    return {"observed": obs, "null_mean": mu, "null_sd": sd, "z": z, "p": p}


def rewire_degree_preserving(W: np.ndarray, seed: int = 0,
                             n_swaps: int = None) -> np.ndarray:
    """Degree-preserving double-edge-swap randomisation of a binary symmetric
    adjacency. Each node keeps its degree but the wiring is random, giving a
    "same connectivity, wrong structure" null for autocorrelation."""
    A = W.toarray() if sp.issparse(W) else np.asarray(W)   # dense, small node sets only
    A = (A > 0).astype(int)
    np.fill_diagonal(A, 0)
    edges = [(i, j) for i in range(len(A)) for j in range(i + 1, len(A)) if A[i, j]]
    if not edges:
        return A.astype(float)
    rng = np.random.default_rng(seed)
    m = len(edges)
    n_swaps = n_swaps if n_swaps is not None else 10 * m
    eset = set(edges)

    def has(a, b):
        return (a, b) in eset or (b, a) in eset

    for _ in range(n_swaps):
        ei, ej = rng.integers(0, m), rng.integers(0, m)
        if ei == ej:
            continue
        a, b = edges[ei]
        c, d = edges[ej]
        if len({a, b, c, d}) < 4:
            continue
        # swap endpoints: (a,b),(c,d) -> (a,d),(c,b)
        if has(a, d) or has(c, b):
            continue
        eset.discard((a, b)); eset.discard((c, d))
        eset.add((a, d)); eset.add((c, b))
        edges[ei] = (a, d); edges[ej] = (c, b)
    out = np.zeros_like(A, dtype=float)
    for (i, j) in eset:
        out[i, j] = out[j, i] = 1.0
    return out
