"""Does the satellite-observed optical signal propagate downstream?

Optical signals such as turbidity and sediment plumes travel downstream, so the
observed field may be flow-organised even where dissolved chemistry is not
(see the autocorrelation test). Each sampled segment is paired with its nearest
flow-downstream segment, its nearest Euclidean segment, and a random segment,
and the rank correlations of the pairs are compared. If the downstream
correlation exceeds the Euclidean one, the signal follows flow beyond what
proximity explains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np

from ..graph import RiverGraph
from ..metrics import spearman_r as _spearman
from ..osm import haversine_km_vec


@dataclass
class PropagationResult:
    name: str
    n_pairs: Dict[str, int]
    r_flow: float
    r_euclidean: float
    r_random: float
    boot_lo: float
    boot_hi: float
    r_reversed: float = float("nan")   # nearest flow-upstream partner (wrong direction)
    n_reversed: int = 0

    @property
    def significant(self) -> bool:
        return self.boot_lo > 0

    def summary(self) -> str:
        v = ("FLOW > EUCLIDEAN (CI excludes 0)" if self.boot_lo > 0
             else "EUCLIDEAN > flow (CI excludes 0)" if self.boot_hi < 0
             else "flow ~ Euclidean (CI includes 0)")
        return (f"optical propagation '{self.name}'  "
                f"(flow {self.n_pairs['flow']} / euc {self.n_pairs['euclidean']} pairs)\n"
                f"  Spearman r:  flow-downstream {self.r_flow:+.3f}   "
                f"euclidean {self.r_euclidean:+.3f}   reversed {self.r_reversed:+.3f}   "
                f"random {self.r_random:+.3f}\n"
                f"  r_flow - r_euc {self.r_flow - self.r_euclidean:+.3f}  "
                f"95% CI [{self.boot_lo:+.3f}, {self.boot_hi:+.3f}]  -> {v}")


def propagation_test(graph: RiverGraph, seg_value: Mapping[str, float], name: str = "",
                     max_dist_km: float = 50.0, sample: int = 3000,
                     seed: int = 0, n_boot: int = 2000) -> PropagationResult:
    valued = [s for s, v in seg_value.items() if s in graph and np.isfinite(v)]
    rng = np.random.default_rng(seed)
    if len(valued) > sample:
        valued_sample = list(rng.choice(valued, size=sample, replace=False))
    else:
        valued_sample = valued
    vset = set(seg_value)
    lat = np.array([graph[s]["lat"] for s in valued])
    lon = np.array([graph[s]["lon"] for s in valued])
    val = np.array([seg_value[s] for s in valued])
    vidx = {s: i for i, s in enumerate(valued)}

    flow_a, flow_b, euc_a, euc_b, rnd_a, rnd_b = [], [], [], [], [], []
    rev_a, rev_b = [], []
    for s in valued_sample:
        vs = seg_value[s]
        # nearest flow-downstream segment with a value within max_dist_km
        partner = None
        for rec in graph.walk_downstream(s, 200):
            if rec["object_id"] in vset and rec["flow_distance_m"] / 1000.0 <= max_dist_km:
                partner = rec["object_id"]
                break
        if partner is None:
            continue            # no flow partner: source dropped from all arms (paired)
        flow_a.append(vs); flow_b.append(seg_value[partner])
        # Reversed-flow control: nearest flow-upstream partner.
        up_partner = None
        for rec in graph.walk_upstream(s, 200):
            if rec["object_id"] in vset and rec["flow_distance_m"] / 1000.0 <= max_dist_km:
                up_partner = rec["object_id"]
                break
        if up_partner is not None:
            rev_a.append(vs); rev_b.append(seg_value[up_partner])
        # Nearest valued segment by great-circle distance, for the same source.
        # All arms use the same source set, so r_flow and r_euc are paired.
        # The Euclidean partner is uncapped and always at most as far as the
        # flow partner, so this control favours geography and is conservative.
        i = vidx[s]
        d = haversine_km_vec(lat[i], lon[i], lat, lon)
        d[i] = np.inf
        j = int(np.argmin(d))
        euc_a.append(vs); euc_b.append(val[j])
        # Random partner for the same source, never the source itself.
        k = int(rng.integers(0, len(valued)))
        if k == i and len(valued) > 1:
            k = (k + 1) % len(valued)
        rnd_a.append(vs); rnd_b.append(val[k])

    fa, fb = np.array(flow_a), np.array(flow_b)
    ea, eb = np.array(euc_a), np.array(euc_b)
    r_flow, r_euc = _spearman(fa, fb), _spearman(ea, eb)
    r_rnd = _spearman(np.array(rnd_a), np.array(rnd_b))
    r_rev = (_spearman(np.array(rev_a), np.array(rev_b)) if len(rev_a) >= 3
             else float("nan"))

    # Paired bootstrap: resample the shared source index once per replicate
    # and recompute both correlations on it.
    n = len(fa)
    boot = []
    for _ in range(n_boot if n >= 3 else 0):
        b = rng.integers(0, n, n)
        boot.append(_spearman(fa[b], fb[b]) - _spearman(ea[b], eb[b]))
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (float("nan"), float("nan"))
    return PropagationResult(name=name or "value",
                             n_pairs={"flow": n, "euclidean": n},
                             r_flow=r_flow, r_euclidean=r_euc, r_random=r_rnd,
                             boot_lo=float(lo), boot_hi=float(hi),
                             r_reversed=r_rev, n_reversed=len(rev_a))
