"""RiverGraph: topological order, walks, deduplication, and no shared state
between two graph instances."""

from experiments.graph import RiverGraph


def test_topo_order_headwaters_before_outlet(diamond_graph):
    order = list(diamond_graph.topo_order())
    assert set(order) == {"H1", "H2", "C", "O"}
    assert order.index("H1") < order.index("C")
    assert order.index("H2") < order.index("C")
    assert order.index("C") < order.index("O")


def test_walk_upstream_collects_all_and_distances(diamond_graph):
    up = {r["object_id"]: r for r in diamond_graph.walk_upstream("O")}
    assert set(up) == {"C", "H1", "H2"}          # start O not included
    assert up["C"]["flow_distance_m"] == 1000.0  # length of O
    assert up["H1"]["flow_distance_m"] == 2000.0  # O + C lengths
    assert up["C"]["depth"] == 1 and up["H1"]["depth"] == 2


def test_walk_downstream_is_a_path(diamond_graph):
    down = [r["object_id"] for r in diamond_graph.walk_downstream("H1")]
    assert down == ["C", "O"]


def test_walks_dedup_on_converging_paths(converging_graph):
    # Z is reachable from X via both A and B. It must appear exactly once.
    up = [r["object_id"] for r in converging_graph.walk_upstream("X")]
    assert sorted(up) == ["A", "B", "Z"]
    assert up.count("Z") == 1


def test_missing_node_returns_empty(diamond_graph):
    assert diamond_graph.walk_upstream("nope") == []
    assert diamond_graph.walk_downstream("nope") == []


def test_no_cross_graph_state_leak():
    g1 = RiverGraph({"a": {"up": [], "next_down": None, "strahler": 1}})
    g2 = RiverGraph({
        "x": {"up": [], "next_down": "y", "strahler": 1},
        "y": {"up": ["x"], "next_down": None, "strahler": 1},
    })
    assert len(g1.topo_order()) == 1
    assert len(g2.topo_order()) == 2
    # caches are independent objects
    assert g1.topo_order() is g1.topo_order()
    assert g1._topo != g2._topo


def test_subgraph_river_filters_by_river_id():
    g = RiverGraph({
        "a": {"up": [], "next_down": None, "strahler": 1, "river_id": "R1"},
        "b": {"up": [], "next_down": None, "strahler": 1, "river_id": "R2"},
    })
    assert g.subgraph_river("R1") == ["a"]
