"""Gap-filling and static propagation on a toy chain. With a smooth downstream
gradient each experiment must detect the structure. With noise it must not."""

import numpy as np

from experiments.eval.gapfill import gapfill_eval
from experiments.eval.propagation_test import propagation_test
from experiments.graph import RiverGraph


def _gradient_chain(n=30):
    """n0->...->n{n-1}, 2 km reaches, value = position (smooth downstream gradient)."""
    nodes, values = {}, {}
    for i in range(n):
        nodes[f"n{i}"] = {
            "up": [f"n{i-1}"] if i > 0 else [],
            "next_down": f"n{i+1}" if i < n - 1 else None,
            "strahler": 1, "length_m": 2000.0,
            "lat": i * 0.02, "lon": 0.0, "river_id": "R", "river_name": "R",
        }
        values[f"n{i}"] = float(i)
    return RiverGraph(nodes), values


def test_gapfill_detects_gradient_structure():
    g, vals = _gradient_chain(30)
    res = gapfill_eval(g, vals, name="grad", max_dist_km=12.0, decay_km=8.0,
                       binary=False, n_boot=500)
    # graph propagation must beat the global mean AND the permuted-value placebo
    assert res.boot_delta["global"]["lo"] > 0      # graph < global RMSE, significantly
    assert res.boot_delta["permuted"]["mean"] > 0  # true structure beats permuted


def test_propagation_detects_gradient_structure():
    g, vals = _gradient_chain(40)
    res = propagation_test(g, vals, name="grad", max_dist_km=12.0, sample=40, n_boot=300)
    assert res.r_flow > 0.8                         # adjacent reaches strongly correlated
    assert abs(res.r_random) < 0.4                  # random pairs uncorrelated


def test_gapfill_finds_no_structure_in_noise():
    # negative control: random values -> graph should not beat the global mean
    g, _ = _gradient_chain(30)
    rng = np.random.default_rng(0)
    noise = {f"n{i}": float(rng.random()) for i in range(30)}
    res = gapfill_eval(g, noise, name="noise", max_dist_km=12.0, n_boot=500)
    assert res.boot_delta["global"]["lo"] <= 0      # no real structure to exploit


def test_propagation_finds_no_structure_in_noise():
    # random field: the flow-minus-Euclidean CI must not lie above 0
    g, _ = _gradient_chain(40)
    rng = np.random.default_rng(0)
    noise = {f"n{i}": float(rng.standard_normal()) for i in range(40)}
    res = propagation_test(g, noise, name="noise", max_dist_km=12.0, sample=40, n_boot=300)
    assert res.boot_lo <= 0
