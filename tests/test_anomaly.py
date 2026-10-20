"""Graph anomaly scores on toy graphs. The graph-expectation residual must
separate a local injection from a reach that only inherits a high value. The
directed-neighbour residual must flag an isolated break but not a coherent plume
that propagates downstream."""

import numpy as np

from experiments.eval.anomaly import (
    _euclidean_neighbor_index,
    _flow_components,
    _flow_neighbor_index,
    _geo_components,
    graph_expectation_residual,
    neighbor_residual_at,
    node_temporal_score,
)
from experiments.graph import RiverGraph


def test_graph_residual_injection_vs_inheritance(diamond_graph):
    segs = ["H1", "H2", "C", "O"]
    T = 5
    X = np.zeros((4, T))
    X[0, :] = 2.0   # H1
    X[1, :] = 4.0   # H2 -> expected at C = flow-weighted mean = 3
    X[2, :] = 3.0   # C inherits the expectation -> residual ~ 0
    X[3, :] = 3.0   # O
    ci = segs.index("C")
    R = graph_expectation_residual(X, segs, diamond_graph)
    assert np.nanmax(np.abs(R[ci])) < 0.1            # inherited-high -> ~0 residual

    Xinj = X.copy(); Xinj[ci, :] = 3.0 + 5.0          # local injection on top
    Rinj = graph_expectation_residual(Xinj, segs, diamond_graph)
    assert np.nanmin(Rinj[ci]) > 4.0                  # injection -> ~5 residual


def test_node_temporal_flags_a_spike():
    X = np.ones((1, 12)) * 2.0
    X[0, 6] = 12.0
    Z = node_temporal_score(X)
    assert np.argmax(Z[0]) == 6 and Z[0, 6] > 3.0


def _two_chains(L=6):
    nodes = {}
    for tag, lon in (("a", 0.0), ("b", 0.0008)):
        for k in range(L):
            nodes[f"{tag}{k}"] = {
                "up": [f"{tag}{k-1}"] if k > 0 else [],
                "next_down": f"{tag}{k+1}" if k < L - 1 else None,
                "strahler": 1, "length_m": 2000.0,
                "lat": k * 0.02, "lon": lon, "river_id": tag, "river_name": tag,
            }
    return RiverGraph(nodes), L


def test_flow_neighbor_residual_separates_violation_from_plume():
    g, L = _two_chains()
    segs = sorted(g.nodes)
    idx = {s: i for i, s in enumerate(segs)}
    T = 6
    X = np.ones((len(segs), T))                       # flat background
    t0 = 3
    # coherent plume in chain A: a2, a3, a4 elevated (connected run)
    for s, v in (("a2", 11.0), ("a3", 9.0), ("a4", 7.0)):
        X[idx[s], t0] = v
    # isolated break in chain B: only b2 elevated
    X[idx["b2"], t0] = 11.0

    cells = [(idx["a3"], t0), (idx["b2"], t0)]   # [plume, violation]
    lat = np.array([g[s]["lat"] for s in segs])
    lon = np.array([g[s]["lon"] for s in segs])
    fnb = _flow_neighbor_index(g, segs, [idx["a3"], idx["b2"]], max_dist_km=10.0)
    enb = _euclidean_neighbor_index(segs, lat, lon, [idx["a3"], idx["b2"]], k=4)
    sf = neighbor_residual_at(X, fnb, cells, decay_km=30.0)
    se = neighbor_residual_at(X, enb, cells)
    # flow: violation residual > plume residual (the plume cell's directed
    # neighbours are also elevated, so the network expects it)
    assert sf[1] > sf[0]
    # flow separates violation from plume more than Euclidean does (Euclidean
    # picks up the adjacent other chain, blurring the contrast)
    assert (sf[1] - sf[0]) > (se[1] - se[0])


def test_components_flow_groups_plume_geo_merges_disconnected():
    g, L = _two_chains()
    segs = sorted(g.nodes)
    idx = {s: i for i, s in enumerate(segs)}
    # anomalous: a connected flow plume (a2,a3,a4) + an isolated cross-river b2
    anom = [idx["a2"], idx["a3"], idx["a4"], idx["b2"]]
    fc = _flow_components(anom, g, segs, max_hops=4)
    sizes = sorted(len(c) for c in fc)
    assert sizes == [1, 3]                          # plume grouped, b2 singleton
    # a2 and b2 are ~60 m apart, so geography merges the disconnected pair
    lat = np.array([g[s]["lat"] for s in segs])
    lon = np.array([g[s]["lon"] for s in segs])
    gc = _geo_components([idx["a2"], idx["b2"]], lat, lon, radius_km=1.0)
    assert len(gc) == 1                             # geography wrongly merges them


def test_geo_components_match_bruteforce():
    """The grid-bucketed _geo_components must match a brute-force all-pairs
    union-find exactly."""
    from experiments.eval.anomaly import _geo_components, _union_find
    from experiments.osm import haversine_m
    rng = np.random.default_rng(0)
    n, radius = 80, 9.0
    lat = 47.0 + 8.0 * rng.random(n)                # German-range latitudes
    lon = 6.0 + 9.0 * rng.random(n)
    anom = list(range(n))
    grid = _geo_components(anom, lat, lon, radius_km=radius)
    find, union = _union_find(n)
    for p in range(n):
        for q in range(p + 1, n):
            if haversine_m(lat[p], lon[p], lat[q], lon[q]) / 1000.0 <= radius:
                union(p, q)
    bf = {}
    for p in range(n):
        bf.setdefault(find(p), []).append(p)
    part = lambda comps: {frozenset(c) for c in comps}
    assert part(grid) == part(bf.values())
