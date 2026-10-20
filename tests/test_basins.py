"""Basin decomposition and accretion: basins are rooted at outlets, ordered
largest first, and the accretion curve adds them cumulatively."""

from experiments.eval.basins import (
    accretion_curve,
    basin_of,
    basins,
    random_order_curves,
)
from experiments.graph import RiverGraph


def _two_basins():
    """Basin A: a0->a1->a2 (outlet a2). Basin B: b0->b1 (outlet b1). Disconnected."""
    nodes = {
        "a0": {"up": [], "next_down": "a1", "strahler": 1, "length_m": 1000, "lat": 0, "lon": 0},
        "a1": {"up": ["a0"], "next_down": "a2", "strahler": 1, "length_m": 1000, "lat": 0, "lon": 0},
        "a2": {"up": ["a1"], "next_down": None, "strahler": 2, "length_m": 1000, "lat": 0, "lon": 0},
        "b0": {"up": [], "next_down": "b1", "strahler": 1, "length_m": 1000, "lat": 1, "lon": 1},
        "b1": {"up": ["b0"], "next_down": None, "strahler": 1, "length_m": 1000, "lat": 1, "lon": 1},
    }
    return RiverGraph(nodes)


def test_basin_of_groups_by_outlet():
    g = _two_basins()
    bof = basin_of(g)
    assert bof["a0"] == bof["a1"] == bof["a2"] == "a2"
    assert bof["b0"] == bof["b1"] == "b1"


def test_basins_sorted_largest_first():
    g = _two_basins()
    bl = basins(g)
    assert len(bl) == 2
    assert bl[0][0] == "a2" and len(bl[0][1]) == 3       # basin A first (3 segments)
    assert bl[1][0] == "b1" and len(bl[1][1]) == 2


def test_accretion_includes_basins_cumulatively():
    g = _two_basins()
    bl = basins(g)
    seen_sizes = []

    def metric(included):
        seen_sizes.append(len(included))
        return float(len(included)), float("nan"), float("nan")

    curve = accretion_curve(bl, metric)
    assert [p.k for p in curve] == [1, 2]
    assert [p.n_segments for p in curve] == [3, 5]       # 3, then 3+2
    assert seen_sizes == [3, 5]


def test_basins_restrict_subset():
    g = _two_basins()
    bl = basins(g, restrict={"a0", "a1"})                # only part of basin A
    assert len(bl) == 1 and bl[0][0] == "a2" and len(bl[0][1]) == 2


def test_cycle_is_guarded():
    nodes = {
        "x": {"up": ["y"], "next_down": "y", "strahler": 1, "length_m": 1, "lat": 0, "lon": 0},
        "y": {"up": ["x"], "next_down": "x", "strahler": 1, "length_m": 1, "lat": 0, "lon": 0},
    }
    g = RiverGraph(nodes)
    bof = basin_of(g)                                    # must terminate, not hang
    assert set(bof) == {"x", "y"}


def test_random_order_curves_same_final_set_varied_orderings():
    g = _two_basins()
    bl = basins(g)

    def metric(included):
        return float(len(included)), float("nan"), float("nan")

    curves = random_order_curves(bl, metric, n_reps=5, seed=0)
    assert len(curves) == 5
    for c in curves:
        assert len(c) == 2                               # one accretion point per basin
        assert c[-1].n_segments == 5                     # every ordering ends on the full set
