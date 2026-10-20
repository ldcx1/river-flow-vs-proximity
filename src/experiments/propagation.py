"""Flow-weighted pollution mixing, injected-load attribution, source tracing.

This is a flow-weighted upstream-influence prior on the segment DAG, not a
calibrated mass-transport model: there are no measured discharges, travel times
or dilution constants. One upstream-to-downstream pass gives every reach:

  flow            discharge-model proxy (surface-area by default)
  expected        flow-weighted blend of the concentrations arriving from the
                  immediate upstream reaches: sum(Q_u * C_u) / sum(Q_u)
  observed        satellite reading at the reach (may be absent)
  injected_load   max(0, observed - expected), the concentration the reach adds
                  beyond what flowed in. Such reaches are candidate sources.
  injected_mass   injected_load * Q, the quantity that is diluted downstream

Source tracing walks upstream from a target, keeps reaches with injected load,
and ranks them by how much of it reaches the target after dilution by the
target's flow and a distance decay.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from .discharge import DischargeModel, default_discharge
from .graph import RiverGraph

# Injected load below this (concentration units) is treated as noise.
DEFAULT_MIN_LOAD = 0.4
# Distance over which an injected load attenuates to 1/e downstream (km).
DEFAULT_DECAY_KM = 40.0


def precompute_flows(graph: RiverGraph,
                     discharge: Optional[DischargeModel] = None) -> Dict[str, float]:
    """Per-segment flow proxy for every node. Pass the result as ``flows`` to
    :func:`compute_field` to avoid recomputing it on repeated calls."""
    disch = discharge or default_discharge()
    return {oid: disch.flow(graph[oid]) for oid in graph.topo_order()}


def compute_field(
    concentrations: Dict[str, float],
    graph: RiverGraph,
    discharge: Optional[DischargeModel] = None,
    flows: Optional[Dict[str, float]] = None,
) -> Dict[str, dict]:
    """One forward mixing pass over the whole graph. Returns per-segment
    ``{flow, observed, expected, injected_load, injected_mass, propagated}``.

    Segments with no observation carry the expected concentration through, so
    confluences downstream still mix. ``flows`` is an optional precomputed
    result of :func:`precompute_flows`.
    """
    disch = discharge or default_discharge()
    field: Dict[str, dict] = {}

    for oid in graph.topo_order():
        node = graph[oid]
        q = flows[oid] if flows is not None else disch.flow(node)
        ups = node.get("up", [])

        # Flow-weighted blend of each upstream reach's propagated concentration
        # (observed if measured, else the value carried through).
        num = 0.0
        den = 0.0
        for u in ups:
            fu = field.get(u)
            if not fu:
                continue
            cu = fu["propagated"]
            qu = fu["flow"]
            if cu is None:
                continue
            num += cu * qu
            den += qu
        expected = (num / den) if den > 0 else None

        observed = concentrations.get(oid)
        if observed is not None:
            base = expected if expected is not None else 0.0
            injected = max(0.0, observed - base)
            propagated = observed
        else:
            injected = 0.0
            propagated = expected  # may be None at an unobserved headwater

        field[oid] = {
            "flow": q,
            "observed": observed,
            "expected": round(expected, 4) if expected is not None else None,
            "injected_load": round(injected, 4),
            "injected_mass": round(injected * q, 4),
            "propagated": propagated,
        }
    return field


def _decay(distance_m: float, decay_km: float) -> float:
    if not decay_km or decay_km <= 0:
        return 1.0
    return math.exp(-(distance_m / 1000.0) / decay_km)


def trace_sources(
    target_oid: str,
    field: Dict[str, dict],
    graph: RiverGraph,
    discharge: Optional[DischargeModel] = None,
    max_depth: int = 200,
    min_load: float = DEFAULT_MIN_LOAD,
    decay_km: float = DEFAULT_DECAY_KM,
    top_k: int = 12,
) -> List[dict]:
    """Rank upstream reaches by how much injected load still reaches ``target``.

    Each source's raw ``contribution`` is its injected mass attenuated over the
    flow distance and diluted by the target's flow
    (``injected_mass * decay(distance) / Q_target``). The reported
    ``contribution_pct`` is each source's share of the total attributed upstream
    load, so listed sources sum to at most 100 (the rest is in sources beyond
    ``top_k``). Sorted strongest first.

    Shares are not computed against the target's observed concentration, because
    ``contribution`` comes from an uncalibrated flow proxy and that ratio could
    exceed 100.
    """
    disch = discharge or default_discharge()
    tgt = field.get(target_oid)
    _q = (tgt or {}).get("flow")
    q_target = _q if _q is not None else disch.flow(graph.get(target_oid, {}))
    out: List[dict] = []
    for rec in graph.walk_upstream(target_oid, max_depth):
        oid = rec["object_id"]
        f = field.get(oid)
        if not f or f["injected_load"] < min_load:
            continue
        dist = rec["flow_distance_m"]
        dec = _decay(dist, decay_km)
        contribution = f["injected_mass"] * dec / q_target if q_target else 0.0
        out.append({
            "object_id": oid,
            "river_id": rec.get("river_id"),
            "river_name": rec.get("river_name"),
            "lat": rec.get("lat"),
            "lon": rec.get("lon"),
            "strahler": rec.get("strahler"),
            "distance_km": round(dist / 1000.0, 2),
            "observed": f["observed"],
            "injected_load": f["injected_load"],
            "flow": f["flow"],
            "decay": round(dec, 4),
            "contribution": round(contribution, 4),
        })
    out.sort(key=lambda r: r["contribution"], reverse=True)

    total = sum(r["contribution"] for r in out)
    if total > 0:
        for r in out:
            r["contribution_pct"] = round(100.0 * r["contribution"] / total, 1)
    return out[:top_k]


def downstream_impact(
    source_oid: str,
    field: Dict[str, dict],
    graph: RiverGraph,
    discharge: Optional[DischargeModel] = None,
    max_depth: int = 400,
    decay_km: float = DEFAULT_DECAY_KM,
    min_contribution: float = 0.05,
) -> List[dict]:
    """Mirror of :func:`trace_sources`: how far the load injected at
    ``source_oid`` is still felt downstream. Each downstream reach gets the
    estimated increment this source adds there."""
    disch = discharge or default_discharge()
    f0 = field.get(source_oid)
    if not f0:
        return []
    mass = f0["injected_mass"]
    if mass <= 0:
        return []
    out: List[dict] = []
    for rec in graph.walk_downstream(source_oid, max_depth):
        oid = rec["object_id"]
        fd = field.get(oid, {})
        _q = fd.get("flow")
        q_down = _q if _q is not None else disch.flow(rec)
        dec = _decay(rec["flow_distance_m"], decay_km)
        contribution = mass * dec / q_down if q_down else 0.0
        if contribution < min_contribution:
            continue
        share = None
        if fd.get("observed"):
            share = round(100.0 * contribution / fd["observed"], 1)
        out.append({
            "object_id": oid,
            "river_id": rec.get("river_id"),
            "river_name": rec.get("river_name"),
            "lat": rec.get("lat"),
            "lon": rec.get("lon"),
            "distance_km": round(rec["flow_distance_m"] / 1000.0, 2),
            "observed": fd.get("observed"),
            "contribution": round(contribution, 4),
            "contribution_pct": share,
        })
    return out
