"""Local vs inherited excess at facility-adjacent exceedances (paper Sec. V-B:
inherited dominates in about 85% of cases in Romania and 90% in Germany).

Each acquisition date is one snapshot of the turbidity field. The upstream-influence
prior (propagation.compute_field) splits every reach's value into an expected
upstream blend (inherited) and a local excess (injected_load). For each per-pass
exceedance above the national 95th percentile that has an upstream baseline, the
local fraction is injected_load / observed. Inherited dominates when it is below 0.5.

Also printed: facility-adjacent vs facility-free local fractions (reach-cluster
bootstrap), and how often the split agrees with an observed step from upstream.

    python scripts/exp_inherited.py --region ro > results/national/inherited_ro.txt   # ~1 min
    python scripts/exp_inherited.py --region de > results/national/inherited_de.txt   # ~10 min
"""
import argparse
import csv
from collections import defaultdict

import numpy as np

import _bootstrap
from experiments.graph import RiverGraph
from experiments.osm import FacilityIndex
from experiments.propagation import compute_field, precompute_flows

REGIONS = {
    "de": (_bootstrap.data_de, "segment_graph_de.json",
           "turb_perdate_national_full.csv", "osm_industrial_de.json"),
    "ro": (_bootstrap.data_ro, "segment_graph_ro.json",
           "turb_perdate_ro_full.csv", "osm_industrial_ro.json"),
}


def cluster_boot(mask_a, mask_b, vals, reaches, n_boot, seed=0):
    """Reach-cluster bootstrap of mean(vals[a]) - mean(vals[b])."""
    uniq = np.array(sorted(set(reaches)))
    rows_of = {u: np.where(reaches == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([rows_of[u] for u in rng.choice(uniq, len(uniq), replace=True)])
        a, b = vals[idx][mask_a[idx]], vals[idx][mask_b[idx]]
        if len(a) and len(b):
            out.append(a.mean() - b.mean())
    return float(np.mean(out)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", choices=sorted(REGIONS), default="ro")
    ap.add_argument("--pct", type=float, default=95.0)
    ap.add_argument("--radius", type=float, default=1500.0, help="facility radius (m)")
    ap.add_argument("--local-cut", type=float, default=0.5)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    data, gname, pname, oname = REGIONS[args.region]
    g = RiverGraph.from_json(data(gname))
    flows = precompute_flows(g)
    fac = FacilityIndex.from_json(data(oname))
    near_of = {o: bool(n.get("lat") is not None and
                       fac.nearby(n["lat"], n["lon"], radius_m=args.radius, limit=1))
               for o, n in g.nodes.items()}

    snap = defaultdict(dict)  # date -> {reach: turbidity}
    with _bootstrap.open_text(data(pname)) as fh:
        for row in csv.DictReader(fh):
            if row["object_id"] not in g.nodes:
                continue
            try:
                v = float(row["turb"])
            except (ValueError, KeyError):
                continue
            snap[row["acquired_at"][:10]][row["object_id"]] = v
    allvals = [v for conc in snap.values() for v in conc.values()]
    thr = float(np.percentile(allvals, args.pct))
    print(f"region={args.region} passes={len(allvals)} dates={len(snap)} "
          f"thr(p{args.pct:g})={thr:.0f}")

    rows = []  # (reach, local fraction, facility-adjacent, step from upstream)
    for conc in snap.values():
        field = compute_field(conc, g, flows=flows)
        for oid, obs in conc.items():
            if obs < thr:
                continue
            f = field.get(oid, {})
            exp = f.get("expected")
            if exp is None or exp <= 0 or obs <= 0:
                continue
            up_obs = [conc[u] for u in g[oid].get("up", []) if u in conc]
            step = not up_obs or max(up_obs) < thr
            rows.append((oid, f["injected_load"] / obs, near_of.get(oid, False), step))

    reach = np.array([r[0] for r in rows])
    lf = np.array([r[1] for r in rows])
    near = np.array([r[2] for r in rows])
    step = np.array([r[3] for r in rows])
    cut = args.local_cut
    print(f"exceedances with an upstream baseline: {len(rows)}, "
          f"facility-adjacent: {near.sum()} ({near.mean():.0%})\n")

    fa = lf[near]
    print("(a) facility-adjacent exceedances")
    print(f"  inherited dominates (local fraction < {cut}): {(fa < cut).mean():.1%}")
    print(f"  local dominates (>= {cut}):                  {(fa >= cut).mean():.1%}")
    print(f"  mean local fraction {fa.mean():.3f}")

    d, lo, hi = cluster_boot(near, ~near, lf, reach, args.n_boot)
    print("\n(b) local fraction, facility-adjacent minus facility-free")
    print(f"  {lf[near].mean():.3f} vs {lf[~near].mean():.3f}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    loc = lf >= cut
    print("\n(c) agreement with an observed step")
    print(f"  local-dominated with upstream below threshold: {step[loc].mean():.0%}")
    print(f"  inherited-dominated with upstream also above:  {(~step[~loc]).mean():.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
