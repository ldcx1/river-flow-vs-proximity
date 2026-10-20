"""Build the river-segment flow DAG from the EU-Hydro network.

Reading the ``River_Net_l`` layer of the EU-Hydro geodatabase needs
``geopandas`` and lives in ``scripts/ingest/build_graph.py``. This module holds
the pure part: turning per-reach records into the ``segment_graph_de.json``
structure.

A ``next_down`` relation only cycles on bad source data (EU-Hydro sometimes
records X->Y and Y->X). For each cycle, the edge leaving the basin outlet
(highest Strahler, then length, then id) is dropped.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple


def enforce_acyclic(flows_into: Dict[str, Optional[str]],
                    priority: Callable[[str], tuple]
                    ) -> Tuple[Dict[str, str], list]:
    """Remove all cycles from a functional ``river -> downstream`` map.

    Returns ``(acyclic_flows, broken)`` where ``broken`` lists ``(outlet, cycle)``
    for each removed edge. The cycle member with the largest ``priority`` key is
    the basin outlet, and its outgoing edge is dropped.
    """
    nxt = {k: v for k, v in flows_into.items() if v is not None}
    state: Dict[str, int] = {}   # 0 unvisited, 1 on-path, 2 settled
    broken = []
    for start in list(nxt.keys()):
        if state.get(start, 0) != 0:
            continue
        path: List[str] = []
        index: Dict[str, int] = {}
        cur: Optional[str] = start
        while cur is not None and state.get(cur, 0) == 0 and cur in nxt:
            state[cur] = 1
            index[cur] = len(path)
            path.append(cur)
            cur = nxt.get(cur)
        if cur is not None and state.get(cur, 0) == 1:
            cycle = path[index[cur]:]
            outlet = max(cycle, key=priority)
            nxt.pop(outlet, None)
            broken.append((outlet, cycle))
        for n in path:
            state[n] = 2
    return nxt, broken


def assemble_graph(records: Iterable[dict], broken: Optional[list] = None) -> Dict[str, dict]:
    """Turn per-reach records into the ``segment_graph`` dict.

    Each record needs: ``object_id``, ``strahler``, ``length_m``, ``lat``,
    ``lon``, ``river_name`` (may be None), ``next_down`` (downstream id or None).
    Cycles are broken deterministically. ``up`` lists come from the surviving
    ``next_down`` edges. Output matches ``segment_graph_de.json``. If ``broken``
    is a list, the removed cycles are appended to it.
    """
    recs = list(records)
    attrs: Dict[str, dict] = {}
    for r in recs:
        attrs[r["object_id"]] = {
            "strahler": int(r.get("strahler") or 1),
            "length_m": float(r.get("length_m") or 0.0),
            "lat": round(float(r["lat"]), 5), "lon": round(float(r["lon"]), 5),
            "river_name": r.get("river_name") or None,
        }
    universe = set(attrs)
    flows: Dict[str, Optional[str]] = {}
    for r in recs:
        oid, nd = r["object_id"], r.get("next_down")
        if nd and nd in universe and nd != oid:
            flows[oid] = nd

    def prio(o: str) -> tuple:
        a = attrs.get(o, {})
        return (a.get("strahler", 1), a.get("length_m", 0.0), o)

    flows, removed = enforce_acyclic(flows, priority=prio)
    if broken is not None:
        broken.extend(removed)
    up = defaultdict(list)
    for s, d in flows.items():
        up[d].append(s)
    return {oid: {
        "next_down": flows.get(oid), "up": sorted(up.get(oid, [])),
        "strahler": a["strahler"], "length_m": round(a["length_m"], 1),
        "lat": a["lat"], "lon": a["lon"], "river_name": a["river_name"],
    } for oid, a in attrs.items()}
