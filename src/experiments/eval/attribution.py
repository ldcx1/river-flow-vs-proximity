"""Source attribution with connectivity controls.

For each observed exceedance, does walking the graph upstream (trace_sources)
find reported pressures better than controls that use the same data but wrong or
no structure?

  m1 graph-upstream   trace_sources on the true flow field (the claim)
  m2 euclidean        nearest reaches by geographic distance (proximity, no flow)
  m3 random           random reaches (chance)
  m4 permuted-field   trace_sources on a field with concentrations shuffled across
                      segments (topology intact, pollution pattern destroyed)
  m5 reversed         rank downstream reaches (wrong flow direction)

A reach is "pressure-aligned" if a facility lies within R metres. Each method is
scored by top-k hit rate and MRR, with a bootstrap over targets for the
m1-vs-control gaps. If m1 beats m2, m4 and m5, the gain comes from connectivity
rather than proximity, the field alone, or undirected structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Dict, List, Mapping, Sequence

import numpy as np

from ..graph import RiverGraph
from ..osm import FacilityIndex, haversine_km_vec
from ..propagation import compute_field, precompute_flows, trace_sources

KS = (1, 3, 5)


def _aligned(index: FacilityIndex, lat, lon, radius_m: float) -> int:
    if lat is None or lon is None:
        return 0
    return 1 if index.nearby(lat, lon, radius_m=radius_m, limit=1) else 0


def _hit_vector(index, reaches: List[dict], radius_m: float) -> List[int]:
    return [_aligned(index, r.get("lat"), r.get("lon"), radius_m) for r in reaches]


def _rank_euclidean(target: str, graph: RiverGraph, seg_ids, seg_lat, seg_lon,
                    top_k: int) -> List[dict]:
    t = graph[target]
    d = haversine_km_vec(t["lat"], t["lon"], seg_lat, seg_lon)
    # Take the nearest top_k+1 (may include the target itself), then drop it.
    kk = min(top_k + 1, len(d))
    cand = np.argpartition(d, kk - 1)[:kk]
    order = cand[np.argsort(d[cand])]
    out = []
    for j in order:
        if seg_ids[j] == target:
            continue
        out.append({"object_id": seg_ids[j], "lat": seg_lat[j], "lon": seg_lon[j]})
        if len(out) >= top_k:
            break
    return out


def _rank_reversed(target: str, graph: RiverGraph, field, top_k: int) -> List[dict]:
    """Wrong-direction control: rank downstream reaches by propagated
    concentration, strongest first.

    This ranks by ``propagated``, not ``injected_load``. Downstream of a polluted
    reach the mixing model already expects a high concentration, so
    ``injected_load`` is near zero there and the list would almost always be
    empty. An empty list scores MRR 0 and would hand the graph an unfair win.
    Ranking by concentration keeps the control non-empty, so m1 and m5 differ
    only in direction."""
    recs = []
    for rec in graph.walk_downstream(target, 200):
        f = field.get(rec["object_id"])
        if not f:
            continue
        c = f.get("propagated")
        if c is None:
            continue
        recs.append((c, {"object_id": rec["object_id"],
                         "lat": rec.get("lat"), "lon": rec.get("lon")}))
    recs.sort(key=lambda x: -x[0])
    return [r for _, r in recs[:top_k]]


def _metrics(hit_vec: List[int], ks=KS) -> Dict:
    hits_at = {k: int(any(hit_vec[:k])) for k in ks}
    fr = next((i + 1 for i, h in enumerate(hit_vec) if h), None)
    return {"hits": hits_at, "rr": (1.0 / fr) if fr else 0.0}


@dataclass
class AttributionResult:
    n_targets: int
    radius_m: float
    methods: List[str]
    mrr: Dict[str, float] = _field(default_factory=dict)
    hit_at: Dict[str, Dict[int, float]] = _field(default_factory=dict)
    boot_delta: Dict[str, dict] = _field(default_factory=dict)  # m1 - control on MRR
    # Number of targets with a non-empty candidate list per method, so an empty
    # control cannot pass as a ranking win.
    n_nonempty: Dict[str, int] = _field(default_factory=dict)
    # Graph-vs-reversed MRR delta on targets where the reversed list is also
    # non-empty. Isolates direction from empty lists near outlets.
    reversed_paired: Dict[str, float] = _field(default_factory=dict)

    def _nonempty_frac(self, m: str) -> float:
        return (self.n_nonempty.get(m, 0) / self.n_targets) if self.n_targets else float("nan")

    def summary(self) -> str:
        lines = [f"attribution over {self.n_targets} exceedance targets "
                 f"(facility within {self.radius_m:.0f} m)\n",
                 f"  {'method':<14}{'MRR':>8}" + "".join(f"   hit@{k}" for k in KS)
                 + f"{'cand!=0':>10}"]
        for m in self.methods:
            row = f"  {m:<14}{self.mrr[m]:>8.3f}"
            row += "".join(f"{self.hit_at[m][k]:>8.1%}" for k in KS)
            row += f"{self._nonempty_frac(m):>10.0%}"
            lines.append(row)
        lines.append("\n  graph-upstream vs controls (MRR delta, target bootstrap 95% CI):")
        for c, d in self.boot_delta.items():
            verdict = "GRAPH WINS" if d["lo"] > 0 else ("control wins" if d["hi"] < 0
                                                        else "inconclusive")
            lines.append(f"    vs {c:<12} {d['mean']:+.3f}  [{d['lo']:+.3f}, {d['hi']:+.3f}]  {verdict}")
        rp = self.reversed_paired
        if rp:
            v = "GRAPH WINS" if rp["lo"] > 0 else ("reversed wins" if rp["hi"] < 0 else "inconclusive")
            lines.append(f"    vs reversed (paired, both non-empty, n={rp['n']}) "
                         f"{rp['mean']:+.3f}  [{rp['lo']:+.3f}, {rp['hi']:+.3f}]  {v}")
        return "\n".join(lines)


def run_attribution(graph: RiverGraph, field_by_month: Mapping[str, Mapping[str, float]],
                    targets_by_month: Mapping[str, Sequence[str]],
                    facilities: FacilityIndex, radius_m: float = 1500.0,
                    top_k: int = 50, min_load: float = 0.4, seed: int = 0,
                    n_boot: int = 2000) -> AttributionResult:
    """``field_by_month[ym]`` = {segment: concentration} for that month;
    ``targets_by_month[ym]`` = exceedance segment ids that month."""
    seg_ids = list(graph.nodes.keys())
    seg_lat = np.array([graph[s]["lat"] for s in seg_ids])
    seg_lon = np.array([graph[s]["lon"] for s in seg_ids])
    rng = np.random.default_rng(seed)
    flows = precompute_flows(graph)          # static, computed once
    methods = ["graph_upstream", "euclidean", "random", "permuted_field", "reversed"]
    per_target = {m: [] for m in methods}   # list of _metrics dicts
    nonempty = {m: 0 for m in methods}      # count of non-empty candidate lists
    rev_ne = []                             # per-target: was the reversed list non-empty?

    for ym, targets in sorted(targets_by_month.items()):
        conc = dict(field_by_month.get(ym, {}))
        if not conc:
            continue
        fld = compute_field(conc, graph, flows=flows)
        # permuted-field null: same graph, concentrations shuffled across segments
        keys = list(conc.keys())
        vals = rng.permutation(np.array([conc[k] for k in keys]))
        fld_perm = compute_field(dict(zip(keys, vals.tolist())), graph, flows=flows)
        for t in targets:
            if t not in graph:
                continue
            m1 = trace_sources(t, fld, graph, top_k=top_k, min_load=min_load)
            if not m1:
                continue
            m2 = _rank_euclidean(t, graph, seg_ids, seg_lat, seg_lon, top_k)
            ridx = rng.choice(len(seg_ids), size=min(top_k, len(seg_ids)), replace=False)
            m3 = [{"object_id": seg_ids[j], "lat": seg_lat[j], "lon": seg_lon[j]} for j in ridx]
            m4 = trace_sources(t, fld_perm, graph, top_k=top_k, min_load=min_load)
            m5 = _rank_reversed(t, graph, fld, top_k)
            for name, reaches in zip(methods, [m1, m2, m3, m4, m5]):
                if reaches:
                    nonempty[name] += 1
                per_target[name].append(_metrics(_hit_vector(facilities, reaches, radius_m)))
            rev_ne.append(bool(m5))

    n = len(per_target["graph_upstream"])
    res = AttributionResult(n_targets=n, radius_m=radius_m, methods=methods,
                            n_nonempty=nonempty)
    for m in methods:
        res.mrr[m] = float(np.mean([d["rr"] for d in per_target[m]])) if n else float("nan")
        res.hit_at[m] = {k: float(np.mean([d["hits"][k] for d in per_target[m]])) if n else float("nan")
                         for k in KS}
    # target bootstrap on MRR(graph) - MRR(control)
    base = np.array([d["rr"] for d in per_target["graph_upstream"]])
    for c in ["euclidean", "permuted_field", "reversed", "random"]:
        ctrl = np.array([d["rr"] for d in per_target[c]])
        diffs = base - ctrl
        boot = np.array([diffs[rng.integers(0, n, n)].mean() for _ in range(n_boot)]) if n else np.array([0.0])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        res.boot_delta[c] = {"mean": float(diffs.mean()), "lo": float(lo), "hi": float(hi)}
    # Paired graph-vs-reversed delta where the reversed list is also non-empty
    # (graph_upstream is non-empty for every included target).
    rev_mask = np.array(rev_ne, dtype=bool)
    if n and rev_mask.any():
        rv = np.array([d["rr"] for d in per_target["reversed"]])
        dp = base[rev_mask] - rv[rev_mask]
        npd = int(rev_mask.sum())
        bp = np.array([dp[rng.integers(0, npd, npd)].mean() for _ in range(n_boot)])
        res.reversed_paired = {"mean": float(dp.mean()),
                               "lo": float(np.percentile(bp, 2.5)),
                               "hi": float(np.percentile(bp, 97.5)), "n": npd}
    return res
