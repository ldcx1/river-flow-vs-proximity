"""Spatial statistics and placebo controls on hand-built graphs: Moran's I on
gradients, noise and alternating patterns, join counts, degree-preserving
rewiring, degree-matched Euclidean graphs, and flow vs Euclidean edges when the
network and geography disagree."""

import numpy as np

from experiments.eval.graph_structure import (
    euclidean_weights,
    euclidean_weights_matched,
    flow_weights,
    join_count_bb,
    mean_degree,
    morans_I,
    permutation_test,
    rewire_degree_preserving,
)
from experiments.graph import RiverGraph


def _chain(n, lat0=0.0, lon=0.0, dlat=0.02):
    """Straight chain n0->n1->...->n{n-1}, 2 km reaches, evenly spaced in lat."""
    nodes = {}
    for i in range(n):
        nodes[f"n{i}"] = {
            "up": [f"n{i-1}"] if i > 0 else [],
            "next_down": f"n{i+1}" if i < n - 1 else None,
            "strahler": 1, "length_m": 2000.0,
            "lat": lat0 + i * dlat, "lon": lon,
            "river_id": "R", "river_name": "R",
        }
    return RiverGraph(nodes)


def test_morans_I_gradient_vs_noise_vs_anti():
    g = _chain(20)
    segs = [f"n{i}" for i in range(20)]
    W, _ = flow_weights(g, segs, max_dist_km=5.0)
    # monotone gradient -> strong positive autocorrelation
    assert morans_I(np.arange(20.0), W) > 0.5
    # alternating high/low -> negative autocorrelation
    alt = np.array([i % 2 for i in range(20)], dtype=float)
    assert morans_I(alt, W) < 0.0


def test_permutation_test_significant_for_gradient_only():
    g = _chain(20)
    segs = [f"n{i}" for i in range(20)]
    W, _ = flow_weights(g, segs, max_dist_km=5.0)
    grad = permutation_test(lambda x: morans_I(x, W), np.arange(20.0), n=999, seed=0)
    assert grad["observed"] > 0.5 and grad["p"] < 0.05      # structured -> significant
    rng = np.random.default_rng(0)
    noise = permutation_test(lambda x: morans_I(x, W), rng.random(20), n=999, seed=1)
    assert noise["p"] > 0.05                                # noise -> not significant


def test_join_count_finds_planted_cluster():
    g = _chain(20)
    segs = [f"n{i}" for i in range(20)]
    W, _ = flow_weights(g, segs, max_dist_km=5.0)
    clustered = np.zeros(20); clustered[6:12] = 1          # 6 consecutive positives
    res = permutation_test(lambda y: join_count_bb(y, W), clustered, n=999, seed=0)
    assert res["z"] > 2 and res["p"] < 0.05                # more BB joins than chance


def test_rewire_preserves_degree():
    g = _chain(20)
    segs = [f"n{i}" for i in range(20)]
    W, _ = flow_weights(g, segs, max_dist_km=5.0)
    R = rewire_degree_preserving(W, seed=0)
    Wd = W.toarray()                                        # W is sparse CSR
    assert np.array_equal(np.sort((Wd > 0).sum(1)), np.sort((R > 0).sum(1)))


def test_euclidean_degree_matches_flow():
    g = _chain(20)
    segs = [f"n{i}" for i in range(20)]
    Wf, _ = flow_weights(g, segs, max_dist_km=5.0)
    k = round(mean_degree(Wf))
    We, _ = euclidean_weights(g, segs, k)
    # symmetrised kNN mean degree is close to k (>= k due to OR-symmetrisation)
    assert abs(mean_degree(We) - mean_degree(Wf)) <= 1.0


def test_euclidean_weights_matched_no_worse_than_round_based():
    # Realised kNN degree exceeds k (OR-symmetrisation), so k=round(target) can
    # over-connect. The matched control must be at least as close to the flow
    # graph's mean degree as the round-based choice.
    g = _chain(20)
    segs = [f"n{i}" for i in range(20)]
    Wf, _ = flow_weights(g, segs, max_dist_km=5.0)
    target = mean_degree(Wf)
    Wm, _ = euclidean_weights_matched(g, segs, target_degree=target)
    Wr, _ = euclidean_weights(g, segs, max(1, round(target)))
    assert abs(mean_degree(Wm) - target) <= abs(mean_degree(Wr) - target) + 1e-9


def test_euclidean_weights_matched_identical_to_k1_when_target_low():
    # With flow degree ~1 (sparse in-situ reaches) the matched control equals
    # euclidean_weights(..., 1) exactly.
    g = _chain(20)
    segs = [f"n{i}" for i in range(20)]
    Wm, _ = euclidean_weights_matched(g, segs, target_degree=1.0)
    W1, _ = euclidean_weights(g, segs, 1)
    assert (Wm != W1).nnz == 0


def test_flow_differs_from_euclidean_when_geography_disagrees():
    # two parallel chains placed close in space but not flow-connected
    a = _chain(8, lat0=0.0, lon=0.0).nodes
    b = _chain(8, lat0=0.0, lon=0.0005).nodes      # ~50 m away, separate river
    b = {f"m{i}": {**v, "up": [f"m{i-1}"] if i > 0 else [],
                   "next_down": f"m{i+1}" if i < 7 else None}
         for i, (k, v) in enumerate(b.items())}
    g = RiverGraph({**a, **b})
    segs = list(a) + list(b)
    Wf, _ = flow_weights(g, segs, max_dist_km=5.0)
    We, _ = euclidean_weights(g, segs, k=2)
    Wf, We = Wf.toarray(), We.toarray()
    # no flow edge crosses the two rivers, Euclidean kNN edges do
    cross_flow = sum(Wf[i, j] for i in range(8) for j in range(8, 16))
    cross_euc = sum(We[i, j] for i in range(8) for j in range(8, 16))
    assert cross_flow == 0
    assert cross_euc > 0
