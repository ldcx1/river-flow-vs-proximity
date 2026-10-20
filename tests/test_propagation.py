"""The flow-weighted upstream prior. compute_field and trace_sources are pinned
to values worked out by hand on the diamond graph."""

import pytest

from experiments import compute_field, trace_sources
from experiments.discharge import SurfaceAreaDischarge


def test_compute_field_mixing(diamond_graph):
    # Equal-flow headwaters at 2 and 4 -> confluence expects their mean (3).
    conc = {"H1": 2.0, "H2": 4.0, "C": 10.0}  # O unobserved
    field = compute_field(conc, diamond_graph)

    # headwaters: no upstream -> expected None, all observed treated as injected
    assert field["H1"]["expected"] is None
    assert field["H1"]["injected_load"] == 2.0
    assert field["H2"]["injected_load"] == 4.0

    # confluence: flow-weighted blend of 2 and 4 (equal flow) = 3.0
    assert field["C"]["expected"] == 3.0
    assert field["C"]["injected_load"] == 7.0      # 10 observed - 3 expected
    assert field["C"]["propagated"] == 10.0

    # outlet: unobserved -> carries the confluence value through, injects nothing.
    # `expected` is rounded to 4dp. `propagated` keeps full float precision so
    # downstream mixing stays accurate, hence approx for it.
    assert field["O"]["observed"] is None
    assert field["O"]["expected"] == 10.0
    assert field["O"]["injected_load"] == 0.0
    assert field["O"]["propagated"] == pytest.approx(10.0)


def test_injected_mass_is_load_times_flow(diamond_graph):
    conc = {"H1": 2.0, "H2": 4.0, "C": 10.0}
    field = compute_field(conc, diamond_graph)
    q_c = SurfaceAreaDischarge().flow(diamond_graph["C"])
    assert field["C"]["injected_mass"] == pytest.approx(7.0 * q_c, abs=1e-3)


def test_no_negative_injection(diamond_graph):
    # A confluence cleaner than its inflow must not report negative injection.
    conc = {"H1": 8.0, "H2": 8.0, "C": 1.0}
    field = compute_field(conc, diamond_graph)
    assert field["C"]["expected"] == 8.0
    assert field["C"]["injected_load"] == 0.0      # max(0, 1 - 8)


def test_trace_sources_ranks_strongest_upstream_first(diamond_graph):
    conc = {"H1": 2.0, "H2": 4.0, "C": 10.0}
    field = compute_field(conc, diamond_graph)
    sources = trace_sources("O", field, diamond_graph, min_load=0.4)
    ids = [s["object_id"] for s in sources]
    # C injects the most mass and is closest, so it ranks first. H2 > H1 (4 vs 2).
    assert ids[0] == "C"
    assert ids == ["C", "H2", "H1"]
    # relative shares are bounded and sum to ~100
    assert sum(s["contribution_pct"] for s in sources) == pytest.approx(100.0, abs=0.5)


def test_trace_sources_min_load_filters(diamond_graph):
    conc = {"H1": 2.0, "H2": 4.0, "C": 10.0}
    field = compute_field(conc, diamond_graph)
    # raise the threshold above H1's injected load (2.0) -> H1 drops out
    sources = trace_sources("O", field, diamond_graph, min_load=3.0)
    ids = [s["object_id"] for s in sources]
    assert "H1" not in ids
    assert set(ids) == {"C", "H2"}


def test_downstream_impact_decays_with_distance():
    from experiments.graph import RiverGraph
    from experiments.propagation import downstream_impact
    # straight chain n0->...->n8. A load injected at n0 is felt downstream, decaying
    # with flow distance, and never attributed back to the source itself.
    nodes = {f"n{i}": {"up": [f"n{i-1}"] if i > 0 else [],
                       "next_down": f"n{i+1}" if i < 8 else None,
                       "strahler": 1, "length_m": 3000.0,
                       "lat": i * 0.02, "lon": 0.0, "river_id": "R", "river_name": "R"}
             for i in range(9)}
    g = RiverGraph(nodes)
    conc = {f"n{i}": 0.1 for i in range(9)}
    conc["n0"] = 5.0                                        # n0 injects load
    field = compute_field(conc, g)
    imp = downstream_impact("n0", field, g, decay_km=40.0, min_contribution=0.0)
    ids = [r["object_id"] for r in imp]
    assert ids and "n0" not in ids                         # only downstream reaches
    assert ids == [f"n{i}" for i in range(1, 9)]           # in downstream order
    assert imp[0]["contribution"] > imp[-1]["contribution"]   # decays with distance
