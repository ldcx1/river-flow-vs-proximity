"""Shared fixtures: small hand-built river graphs, independent of the real data."""

import os
import sys

import pytest

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from experiments.graph import RiverGraph  # noqa: E402


def _node(up, next_down, strahler, length_m=1000.0, lat=0.0, lon=0.0,
          river_id="R", river_name="R"):
    return {"up": up, "next_down": next_down, "strahler": strahler,
            "length_m": length_m, "lat": lat, "lon": lon,
            "river_id": river_id, "river_name": river_name}


@pytest.fixture
def diamond_graph():
    """Two headwaters H1, H2 -> confluence C -> outlet O. Small enough to work the
    flow-prior mixing out by hand.

        H1 \\
             C --> O
        H2 /
    """
    nodes = {
        "H1": _node([], "C", 1, lat=1.0, lon=1.0),
        "H2": _node([], "C", 1, lat=1.0, lon=2.0),
        "C":  _node(["H1", "H2"], "O", 2, lat=2.0, lon=1.5),
        "O":  _node(["C"], None, 2, lat=3.0, lon=1.5),
    }
    return RiverGraph(nodes)


@pytest.fixture
def converging_graph():
    """Z is reachable from X along two upstream paths (via A and via B). Used to
    check that walks deduplicate.

        Z --> A \\
                  X
        Z --> B /
    """
    nodes = {
        "Z": _node([], "A", 1),          # next_down unused, walks use 'up'
        "A": _node(["Z"], "X", 1),
        "B": _node(["Z"], "X", 1),
        "X": _node(["A", "B"], None, 2),
    }
    return RiverGraph(nodes)
