"""River-segment flow graph: loading, upstream/downstream walks, topological order.

Caches live on the instance, so several graphs (e.g. Germany and Romania) can be
used in one process.

Each node is a 1-3 km EU-Hydro reach::

    {
      "next_down": object_id | None,   # single downstream reach (acyclic)
      "up": [object_id, ...],          # immediate upstream reaches
      "strahler": int,
      "length_m": float,
      "lat": float, "lon": float,      # reach centroid
      "river_id": str, "river_name": str
    }

The graph is a forest of basin DAGs, one tree per outlet. Confluences are reaches
that share a ``next_down``. It is read-only after load.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Dict, List, Optional


class RiverGraph:
    """A river-segment DAG. Construct from a ``{object_id: node}`` mapping, or
    load from JSON with :meth:`from_json`.

    Walks are bounded by ``max_depth`` (in reaches, not metres) and skip visited
    reaches, so a braided confluence is not counted twice.
    """

    def __init__(self, nodes: Dict[str, dict]):
        self.nodes: Dict[str, dict] = nodes
        self._topo: Optional[tuple] = None

    # -- construction --------------------------------------------------------
    @classmethod
    def from_json(cls, path: str) -> "RiverGraph":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    # -- mapping-ish accessors ----------------------------------------------
    def __contains__(self, object_id: str) -> bool:
        return object_id in self.nodes

    def __getitem__(self, object_id: str) -> dict:
        return self.nodes[object_id]

    def __len__(self) -> int:
        return len(self.nodes)

    def get(self, object_id: str, default=None):
        return self.nodes.get(object_id, default)

    # -- walks ---------------------------------------------------------------
    def walk_upstream(self, object_id: str, max_depth: int = 200) -> List[dict]:
        """Every reach upstream of ``object_id`` (breadth-first), each annotated
        with ``depth`` (hops) and ``flow_distance_m`` (summed reach lengths from
        the start). The start segment is not included.

        Each step to an upstream neighbour adds the length of the reach being
        left (the current node).
        """
        g = self.nodes
        if object_id not in g:
            return []
        out: List[dict] = []
        seen = {object_id}
        frontier = deque([(object_id, 0, 0.0)])
        while frontier:
            cur, depth, dist = frontier.popleft()
            if depth >= max_depth:
                continue
            for up in g[cur]["up"]:
                if up in seen:
                    continue
                seen.add(up)
                d = dist + g[cur].get("length_m", 0.0)
                rec = dict(g[up], object_id=up, depth=depth + 1,
                           flow_distance_m=round(d, 1))
                out.append(rec)
                frontier.append((up, depth + 1, d))
        return out

    def walk_downstream(self, object_id: str, max_depth: int = 400) -> List[dict]:
        """The downstream chain from ``object_id`` (single-valued, so a path,
        not a tree), each reach annotated with ``depth`` and ``flow_distance_m``.
        The start segment is not included."""
        g = self.nodes
        if object_id not in g:
            return []
        out: List[dict] = []
        seen = {object_id}
        cur, depth, dist = object_id, 0, 0.0
        while depth < max_depth:
            nd = g[cur].get("next_down")
            if not nd or nd in seen or nd not in g:
                break
            seen.add(nd)
            dist += g[cur].get("length_m", 0.0)
            out.append(dict(g[nd], object_id=nd, depth=depth + 1,
                            flow_distance_m=round(dist, 1)))
            cur, depth = nd, depth + 1
        return out

    def topo_order(self) -> tuple:
        """Object ids in upstream->downstream order (Kahn's algorithm).
        Headwaters come first, so a single forward pass sees every upstream
        contributor before the reach itself. Cached after the first call.
        """
        if self._topo is not None:
            return self._topo
        g = self.nodes
        indeg = {oid: len(n["up"]) for oid, n in g.items()}
        queue = [oid for oid, d in indeg.items() if d == 0]
        order: List[str] = []
        while queue:
            oid = queue.pop()
            order.append(oid)
            nd = g[oid].get("next_down")
            if nd and nd in indeg:
                indeg[nd] -= 1
                if indeg[nd] == 0:
                    queue.append(nd)
        # Nodes left in an unbroken cycle are appended in arbitrary order so
        # callers still see every reach. The graph is built acyclic, so this is
        # normally empty.
        if len(order) < len(g):
            seen = set(order)
            order.extend(oid for oid in g if oid not in seen)
        self._topo = tuple(order)
        return self._topo

    def subgraph_river(self, river_id: str) -> List[str]:
        """All segment ids belonging to one ``river_id``."""
        return [oid for oid, n in self.nodes.items() if n.get("river_id") == river_id]
