"""EU-Hydro cycle-breaking + graph assembly."""
from experiments.ingest import euhydro


def test_enforce_acyclic_breaks_two_cycle_at_outlet():
    flows = {"a": "b", "b": "a"}                       # bad 2-cycle
    prio = lambda o: ({"a": 2, "b": 1}[o], 0.0, o)     # a = outlet (higher order)
    ac, broken = euhydro.enforce_acyclic(flows, prio)
    assert len(broken) == 1
    assert ac.get("a") is None                         # outlet's edge dropped
    assert ac.get("b") == "a"


def test_enforce_acyclic_leaves_chain_untouched():
    flows = {"a": "b", "b": "c"}
    ac, broken = euhydro.enforce_acyclic(flows, lambda o: (0, 0, o))
    assert broken == [] and ac == {"a": "b", "b": "c"}


def test_assemble_graph_up_down_and_attrs():
    recs = [
        {"object_id": "a", "strahler": 1, "length_m": 10, "lat": 50.0, "lon": 8.0,
         "river_name": "R", "next_down": "c"},
        {"object_id": "b", "strahler": 1, "length_m": 20, "lat": 50.1, "lon": 8.1,
         "river_name": None, "next_down": "c"},
        {"object_id": "c", "strahler": 2, "length_m": 30, "lat": 50.2, "lon": 8.2,
         "river_name": "R", "next_down": None},
    ]
    g = euhydro.assemble_graph(recs)
    assert g["a"]["next_down"] == "c" and g["c"]["next_down"] is None
    assert g["c"]["up"] == ["a", "b"]
    assert g["a"]["strahler"] == 1 and g["c"]["strahler"] == 2
    assert g["b"]["river_name"] is None


def test_assemble_graph_drops_dangling_edge():
    recs = [{"object_id": "a", "strahler": 1, "length_m": 1, "lat": 50, "lon": 8,
             "river_name": None, "next_down": "ZZZ"}]        # target not in universe
    assert euhydro.assemble_graph(recs)["a"]["next_down"] is None


def test_assemble_graph_reports_broken_cycles():
    recs = [{"object_id": o, "strahler": 1, "length_m": 1.0, "lat": 0, "lon": 0,
             "next_down": nd} for o, nd in (("a", "b"), ("b", "a"), ("c", None))]
    broken = []
    euhydro.assemble_graph(recs, broken=broken)
    assert len(broken) == 1
