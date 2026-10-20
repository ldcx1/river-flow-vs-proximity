"""Basins: basin-accretion scaling analysis.

Decompose the graph into outlet-rooted basins, order them by size, and re-run a
headline metric on the cumulative top-k basins (k=1..N) to see how progressively
smaller basins move the result. A random-order ensemble separates "basin size
composition matters" from "just more data".

Graph-only and metric-agnostic: the caller supplies a ``metric_fn(included_segments)``
that recomputes its experiment's primary statistic on the restricted segment set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..graph import RiverGraph


def basin_of(graph: RiverGraph) -> Dict[str, str]:
    """Map each segment to its terminal outlet id (follow next_down to a node with
    no downstream, cycle-guarded). Segments sharing an outlet form one basin."""
    memo: Dict[str, str] = {}
    for start in graph.nodes:
        if start in memo:
            continue
        path: List[str] = []
        cur = start
        res = None
        while True:
            if cur in memo:
                res = memo[cur]
                break
            if cur in path:                       # cycle -> treat as its own outlet
                res = cur
                break
            path.append(cur)
            nd = graph[cur].get("next_down")
            if not nd or nd not in graph:
                res = cur
                break
            cur = nd
        for s in path:
            memo[s] = res
    return memo


def basins(graph: RiverGraph, restrict: Optional[Set[str]] = None
           ) -> List[Tuple[str, List[str]]]:
    """List of (outlet_id, members) basins, restricted to ``restrict`` segments if
    given, sorted largest-first by member count."""
    bof = basin_of(graph)
    groups: Dict[str, List[str]] = {}
    for s, outlet in bof.items():
        if restrict is not None and s not in restrict:
            continue
        groups.setdefault(outlet, []).append(s)
    return sorted(groups.items(), key=lambda kv: -len(kv[1]))


@dataclass
class AccretionPoint:
    k: int                 # number of basins included
    n_segments: int        # cumulative segments
    value: float           # the tracked metric
    lo: float = float("nan")
    hi: float = float("nan")


def accretion_curve(ordered_basins: Sequence[Tuple[str, List[str]]],
                    metric_fn: Callable[[Set[str]], Tuple[float, float, float]]
                    ) -> List[AccretionPoint]:
    """For k=1..N, include the first k basins' segments and call
    ``metric_fn(included)`` -> (value, lo, hi). Returns the curve."""
    included: Set[str] = set()
    curve: List[AccretionPoint] = []
    for k, (_, members) in enumerate(ordered_basins, 1):
        included |= set(members)
        v, lo, hi = metric_fn(included)
        curve.append(AccretionPoint(k=k, n_segments=len(included), value=v, lo=lo, hi=hi))
    return curve


def random_order_curves(all_basins: Sequence[Tuple[str, List[str]]],
                        metric_fn: Callable[[Set[str]], Tuple[float, float, float]],
                        n_reps: int = 10, seed: int = 0) -> List[List[AccretionPoint]]:
    """Accretion curves under ``n_reps`` random basin orderings (the pure
    more-data reference vs the size-ordered curve)."""
    rng = np.random.default_rng(seed)
    bl = list(all_basins)
    out = []
    for _ in range(n_reps):
        idx = rng.permutation(len(bl))
        out.append(accretion_curve([bl[i] for i in idx], metric_fn))
    return out


def summarize(curve: List[AccretionPoint]) -> str:
    lines = [f"  {'k':>3} {'segs':>7} {'metric':>10} {'95% CI':>22}"]
    for p in curve:
        ci = (f"[{p.lo:+.3f}, {p.hi:+.3f}]" if np.isfinite(p.lo) else "")
        lines.append(f"  {p.k:>3} {p.n_segments:>7} {p.value:>10.3f} {ci:>22}")
    return "\n".join(lines)
