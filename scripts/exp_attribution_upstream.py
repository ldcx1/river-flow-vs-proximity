"""Fair upstream-restricted source ranking (paper Sec. V-B).

Does flow-weighted load ranking find OSM facilities better than proximity when
both rank the same candidates?
  * Targets are point-source-like spikes: reach-months whose local excess
    (observed minus flow-weighted upstream expectation) is above --pct.
  * Both methods rank the same upstream pool of each target. The graph ranks by
    flow-weighted contribution, the control by distance (nearest first).
  * A hit is a ranked reach within R m of a facility, swept over --radii.
  * Lift is hit@1 over the base rate, the share of the target's own upstream
    pool near a facility.

Prints MRR, hit@k and lift per radius with paired target-bootstrap 95% CIs, plus
a split by facility density. Label-free. Published result: tie in Germany,
marginal graph edge in Romania.

Each log holds two runs, --pct 99 (strict targets) then --pct 95 (the
powered set the paper reports). Romania is the default input:

    (python scripts/exp_attribution_upstream.py --pct 99
     python scripts/exp_attribution_upstream.py --pct 95
    ) > results/national/ro_attribution_upstream.txt

    (for p in 99 95; do python scripts/exp_attribution_upstream.py --pct $p \\
        --graph data/de/segment_graph_de.json \\
        --series data/de/turb_series_national.csv \\
        --osm data/de/osm_industrial_de.json; done
    ) > results/national/de_attribution_upstream.txt
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np

import _bootstrap  # noqa: F401
from experiments.graph import RiverGraph
from experiments.osm import FacilityIndex
from experiments.propagation import compute_field, trace_sources

KS = (1, 3, 5)


def _region_name(graph_path: str) -> str:
    stem = os.path.splitext(os.path.basename(graph_path))[0].lower()
    if stem.endswith("_de") or "_de_" in stem:
        return "Germany"
    if stem.endswith("_ro") or "_ro_" in stem:
        return "Romania"
    return stem


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", default=_bootstrap.data_ro("segment_graph_ro.json"))
    ap.add_argument("--series", default=_bootstrap.data_ro("turb_series_ro.csv"))
    ap.add_argument("--osm", default=_bootstrap.data_ro("osm_industrial_ro.json"))
    ap.add_argument("--pct", type=float, default=95.0,
                    help="local-excess percentile above which a reach-month is a point-source target")
    ap.add_argument("--radii", default="500,1000,1500,3000")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--min-up", type=int, default=5,
                    help="require this many upstream candidates (skip near-headwater targets)")
    ap.add_argument("--max-targets", type=int, default=20000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--strat-radius", type=int, default=1500,
                    help="radius at which to stratify targets into base-rate (density) tertiles")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--region", default=None,
                    help="region name for the header (default: inferred from the graph "
                         "file name, _de -> Germany, _ro -> Romania)")
    args = ap.parse_args()
    region = args.region or _region_name(args.graph)
    radii = [int(x) for x in args.radii.split(",")]
    rng = np.random.default_rng(args.seed)

    g = RiverGraph.from_json(args.graph)
    fac = FacilityIndex.from_json(args.osm)
    conc_by_month = defaultdict(dict)
    for r in csv.DictReader(_bootstrap.open_text(args.series)):
        try:
            conc_by_month[r["ym"]][r["object_id"]] = float(r["turb"])
        except (ValueError, KeyError):
            pass

    field_by_month, excess = {}, []
    for ym, conc in conc_by_month.items():
        fld = compute_field(conc, g)
        field_by_month[ym] = fld
        for seg, f in fld.items():
            il = f.get("injected_load")
            if il is not None and il > 0:
                excess.append((il, seg, ym))
    thr = float(np.percentile([e[0] for e in excess], args.pct))
    targets = [(seg, ym) for il, seg, ym in excess if il >= thr]
    if len(targets) > args.max_targets:
        targets = [targets[i] for i in rng.choice(len(targets), args.max_targets, replace=False)]
    print(f"# upstream attribution, {region} | pct=p{args.pct:g} (excess>={thr:.1f}) | "
          f"point-source targets {len(targets)} | facilities {len(fac.facilities)}")

    def rr(hits):
        for i, x in enumerate(hits):
            if x:
                return 1.0 / (i + 1)
        return 0.0

    per = {R: {"g_rr": [], "e_rr": [], "gh": {k: [] for k in KS},
               "eh": {k: [] for k in KS}, "base": []} for R in radii}
    used = 0
    for seg, ym in targets:
        cands = trace_sources(seg, field_by_month[ym], g, min_load=0.0, top_k=10**9)
        if len(cands) < args.min_up:
            continue
        used += 1
        algn = {c["object_id"]: {R: (1 if (c.get("lat") is not None and c.get("lon") is not None
                                and fac.nearby(c["lat"], c["lon"], radius_m=R, limit=1)) else 0)
                                 for R in radii} for c in cands}
        graph_ranked = cands[:args.top_k]
        euc_ranked = sorted(cands, key=lambda r: r["distance_km"])[:args.top_k]
        for R in radii:
            gh = [algn[c["object_id"]][R] for c in graph_ranked]
            eh = [algn[c["object_id"]][R] for c in euc_ranked]
            per[R]["g_rr"].append(rr(gh)); per[R]["e_rr"].append(rr(eh))
            for k in KS:
                per[R]["gh"][k].append(1 if any(gh[:k]) else 0)
                per[R]["eh"][k].append(1 if any(eh[:k]) else 0)
            per[R]["base"].append(float(np.mean([algn[c["object_id"]][R] for c in cands])))
    print(f"# used targets (>= {args.min_up} upstream) {used}\n")

    def bci(vals):
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    def vlabel(lo, hi):
        return "GRAPH WINS" if lo > 0 else ("proximity wins" if hi < 0 else "tie (CI incl 0)")

    for R in radii:
        d = per[R]; n = len(d["g_rr"])
        g_rr, e_rr = np.array(d["g_rr"]), np.array(d["e_rr"])
        gh1, eh1, base = np.array(d["gh"][1], float), np.array(d["eh"][1], float), np.array(d["base"])
        diff = g_rr - e_rr
        idxs = [rng.integers(0, n, n) for _ in range(args.n_boot)]     # shared target resamples
        mlo, mhi = bci(np.array([diff[i].mean() for i in idxs]))
        # lift = hit@1 / local base rate, with bootstrap CIs on each arm and the delta
        g_lift, e_lift = gh1.mean() / base.mean(), eh1.mean() / base.mean()
        gllo, glhi = bci(np.array([gh1[i].mean() / base[i].mean() for i in idxs]))
        ello, elhi = bci(np.array([eh1[i].mean() / base[i].mean() for i in idxs]))
        dllo, dlhi = bci(np.array([gh1[i].mean()/base[i].mean() - eh1[i].mean()/base[i].mean() for i in idxs]))
        print(f"--- radius {R} m | upstream base rate {base.mean():.1%} ---")
        print(f"  graph_upstream   MRR {g_rr.mean():.3f}  hit@1 {gh1.mean():.1%} hit@3 {np.mean(d['gh'][3]):.1%} "
              f"hit@5 {np.mean(d['gh'][5]):.1%}  lift@1 {g_lift:.2f}x [{gllo:.2f},{glhi:.2f}]")
        print(f"  euclid_upstream  MRR {e_rr.mean():.3f}  hit@1 {eh1.mean():.1%} hit@3 {np.mean(d['eh'][3]):.1%} "
              f"hit@5 {np.mean(d['eh'][5]):.1%}  lift@1 {e_lift:.2f}x [{ello:.2f},{elhi:.2f}]")
        print(f"  graph - euclid   MRR   {diff.mean():+.3f} [{mlo:+.3f}, {mhi:+.3f}]  {vlabel(mlo, mhi)}")
        print(f"  graph - euclid   lift@1 {g_lift - e_lift:+.2f}x [{dllo:+.2f}, {dlhi:+.2f}]  {vlabel(dllo, dlhi)}\n")

    # Split targets into base-rate tertiles: is any graph edge carried only by
    # facility-dense targets?
    R = args.strat_radius
    d = per[R]
    base = np.array(d["base"]); diff = np.array(d["g_rr"]) - np.array(d["e_rr"])
    order = np.argsort(base)
    print(f"density stratification @ {R} m (graph-euclid MRR within base-rate tertiles):")
    for lab, i3 in zip(("low ", "med ", "high"), np.array_split(order, 3)):
        m = len(i3); dd = diff[i3]
        lo, hi = bci(np.array([dd[rng.integers(0, m, m)].mean() for _ in range(args.n_boot)]))
        br = base[i3]
        print(f"  {lab} density (base {br.min():.1%}-{br.max():.1%}, n={m}): "
              f"{dd.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  {vlabel(lo, hi)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
