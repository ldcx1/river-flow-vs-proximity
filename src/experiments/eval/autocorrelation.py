"""Is water quality more autocorrelated along the flow network than along
geographic proximity?

If so, the river graph organises the data in a way plain geography does not.
This is a model-free test. The control is a degree-matched Euclidean kNN
adjacency. Significance comes from permuting the per-segment values (the graph
stays fixed), and the flow-minus-Euclidean gap is permutation-tested too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping

import numpy as np

from ..graph import RiverGraph
from .graph_structure import (
    euclidean_weights_matched,
    flow_weights,
    mean_degree,
    morans_I,
    permutation_test,
    rewire_degree_preserving,
)


@dataclass
class AutocorrResult:
    name: str
    n_segments: int
    flow_degree: float
    euc_degree: float
    I_flow: float
    I_euc: float
    p_flow: float          # is flow autocorrelation > chance?
    I_diff: float          # I_flow - I_euc
    p_diff: float          # is the flow-minus-Euclidean gap above chance?
    I_rewired: float       # flow I under degree-preserving rewire (placebo)

    def line(self) -> str:
        return (f"{self.name:<16} n={self.n_segments:<4} "
                f"I_flow={self.I_flow:+.3f}(p={self.p_flow:.3f})  "
                f"I_euc={self.I_euc:+.3f}  diff={self.I_diff:+.3f}(p={self.p_diff:.3f})  "
                f"I_rewired={self.I_rewired:+.3f}  "
                f"[mean-deg flow={self.flow_degree:.1f} euc={self.euc_degree:.1f}]")


def morans_autocorr(graph: RiverGraph, seg_values: Mapping[str, float],
                    name: str = "", max_dist_km: float = 50.0,
                    n_perm: int = 2000, seed: int = 0,
                    n_rewire: int = 25) -> AutocorrResult:
    """Moran's I of ``seg_values`` under flow vs degree-matched Euclidean
    adjacency, with permutation significance and a rewire placebo. Restricts to
    segments that have at least one flow-neighbour (so flow and Euclidean are
    scored on the same node set)."""
    segs = [s for s, v in seg_values.items() if s in graph and np.isfinite(v)]
    Wf_all, segs = flow_weights(graph, segs, max_dist_km)
    deg = np.asarray(Wf_all.getnnz(axis=1)).ravel()
    keep = [s for i, s in enumerate(segs) if deg[i] > 0]
    Wf, keep = flow_weights(graph, keep, max_dist_km)
    x = np.array([seg_values[s] for s in keep], dtype=float)
    # Match the flow graph's realised mean degree exactly (not rounded), so the
    # Euclidean control is not over-connected.
    We, _ = euclidean_weights_matched(graph, keep, target_degree=mean_degree(Wf))

    I_flow = morans_I(x, Wf)
    I_euc = morans_I(x, We)
    pf = permutation_test(lambda v: morans_I(v, Wf), x, n=n_perm, seed=seed)
    pd = permutation_test(lambda v: morans_I(v, Wf) - morans_I(v, We), x,
                          n=n_perm, seed=seed)
    I_rew = float(np.mean([morans_I(x, rewire_degree_preserving(Wf, seed=s))
                           for s in range(n_rewire)]))
    return AutocorrResult(
        name=name or "value", n_segments=len(keep),
        flow_degree=mean_degree(Wf), euc_degree=mean_degree(We),
        I_flow=I_flow, I_euc=I_euc, p_flow=pf["p"],
        I_diff=I_flow - I_euc, p_diff=pd["p"], I_rewired=I_rew)


def segment_means(readings: List[Mapping[str, str]], determinand: str,
                  graph: RiverGraph) -> Dict[str, float]:
    """Mean reported value of ``determinand`` per segment (segments in graph)."""
    acc: Dict[str, List[float]] = {}
    for r in readings:
        if r["determinand"] != determinand or r["segment"] not in graph:
            continue
        try:
            acc.setdefault(r["segment"], []).append(float(r["value"]))
        except (ValueError, TypeError):
            continue
    return {s: float(np.mean(v)) for s, v in acc.items() if v}
