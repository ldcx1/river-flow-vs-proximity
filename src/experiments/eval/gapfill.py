"""Gap-filling at unmonitored reaches.

Can a reach's water quality be estimated from other segments' observations
along the network better than by Euclidean interpolation? A node-only model
cannot do this at all, since a reach without a station has no features.

Leave-one-segment-out. For held-out S, predict from the other observed segments:
  g1 graph     flow-distance-decayed mean of network neighbours within D km
  g2 euclidean inverse-distance mean of the k nearest by geography (k matched)
  g3 global    mean of all other observed segments
  g4 placebo   g1 on a graph where the values are permuted across segments
Report RMSE/MAE (and ROC-AUC for a binary target), with a segment bootstrap on
RMSE(control) - RMSE(graph) (positive => graph better).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as _field
from typing import Dict, Mapping

import numpy as np

from ..graph import RiverGraph
from ..metrics import roc_auc
from ..osm import haversine_km_vec
from .graph_structure import flow_neighbors


@dataclass
class GapfillResult:
    name: str
    n: int
    binary: bool
    rmse: Dict[str, float] = _field(default_factory=dict)
    mae: Dict[str, float] = _field(default_factory=dict)
    auc: Dict[str, float] = _field(default_factory=dict)
    boot_delta: Dict[str, dict] = _field(default_factory=dict)  # rmse(ctrl) - rmse(graph)

    def summary(self) -> str:
        methods = ["graph", "euclidean", "global", "permuted"]
        lines = [f"gap-filling for '{self.name}'  (n={self.n}, "
                 f"{'binary' if self.binary else 'continuous'})\n",
                 f"  {'method':<12}{'RMSE':>9}{'MAE':>9}" + ("   ROC-AUC" if self.binary else "")]
        for m in methods:
            row = f"  {m:<12}{self.rmse[m]:>9.3f}{self.mae[m]:>9.3f}"
            if self.binary:
                # The leave-one-out global mean falls as the held-out label rises,
                # so its ROC-AUC is 0 by construction and carries no information.
                row += (f"{'n/a':>10}" if m == "global"
                        else f"{self.auc.get(m, float('nan')):>10.3f}")
            lines.append(row)
        lines.append("\n  graph vs controls (RMSE reduction, segment bootstrap 95% CI;"
                     " positive => graph better):")
        for c, d in self.boot_delta.items():
            verdict = "GRAPH WINS" if d["lo"] > 0 else ("control wins" if d["hi"] < 0
                                                        else "inconclusive")
            lines.append(f"    vs {c:<10} {d['mean']:+.3f}  [{d['lo']:+.3f}, {d['hi']:+.3f}]  {verdict}")
        return "\n".join(lines)


def gapfill_eval(graph: RiverGraph, seg_values: Mapping[str, float], name: str = "",
                 max_dist_km: float = 50.0, decay_km: float = 30.0,
                 binary: bool = False, seed: int = 0, n_boot: int = 2000) -> GapfillResult:
    segs = [s for s, v in seg_values.items() if s in graph and np.isfinite(v)]
    x = np.array([seg_values[s] for s in segs], dtype=float)
    n = len(segs)
    idx = {s: i for i, s in enumerate(segs)}
    lat = np.array([graph[s]["lat"] for s in segs])
    lon = np.array([graph[s]["lon"] for s in segs])

    nb = flow_neighbors(graph, segs, max_dist_km)             # seg -> {seg: dist_km}
    # euclidean distances (km) among monitored segments (vectorised great-circle)
    D = np.zeros((n, n))
    for i in range(n):
        D[i] = haversine_km_vec(lat[i], lon[i], lat, lon)
    deg = [len(nb[s]) for s in segs]
    k = max(1, int(np.median([d for d in deg if d > 0]) if any(deg) else 3))

    rng = np.random.default_rng(seed)
    xperm = rng.permutation(x)                                 # placebo value field

    def predict(i, values):
        s = segs[i]
        ws = [(values[idx[t]], math.exp(-d / decay_km)) for t, d in nb[s].items() if t in idx]
        # No network neighbour: fall back to the mean of the other segments.
        # Self is excluded so the held-out value does not leak.
        g1 = (sum(v * w for v, w in ws) / sum(w for _, w in ws)) if ws \
            else float(np.delete(values, i).mean())
        return g1

    preds = {m: np.empty(n) for m in ["graph", "euclidean", "global", "permuted"]}
    for i in range(n):
        others = np.array([j for j in range(n) if j != i])
        preds["graph"][i] = predict(i, x)
        preds["permuted"][i] = predict(i, xperm)
        preds["global"][i] = x[others].mean()
        # euclidean: k nearest others, inverse-distance (decay) weighted
        dd = D[i, others]
        order = others[np.argsort(dd)[:k]]
        w = np.exp(-D[i, order] / decay_km)
        preds["euclidean"][i] = float((x[order] * w).sum() / w.sum()) if w.sum() > 0 else x[others].mean()

    res = GapfillResult(name=name or "value", n=n, binary=binary)
    for m, p in preds.items():
        err = p - x
        res.rmse[m] = float(np.sqrt((err ** 2).mean()))
        res.mae[m] = float(np.abs(err).mean())
        if binary and 0 < x.sum() < n:
            res.auc[m] = float(roc_auc(x, p))
    # segment bootstrap on RMSE(control) - RMSE(graph)
    eg = (preds["graph"] - x) ** 2
    for c in ["euclidean", "global", "permuted"]:
        ec = (preds[c] - x) ** 2
        boot = []
        for _ in range(n_boot):
            b = rng.integers(0, n, n)
            boot.append(math.sqrt(ec[b].mean()) - math.sqrt(eg[b].mean()))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        res.boot_delta[c] = {"mean": float(np.sqrt(ec.mean()) - np.sqrt(eg.mean())),
                             "lo": float(lo), "hi": float(hi)}
    return res
