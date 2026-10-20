"""Time-aware source attribution (paper Sec. V-B, Fig. 4).

Each per-pass turbidity exceedance (above the national 95th percentile) is
looked at against its three-hop upstream neighbourhood over a 30-day window.
Passes within +/-6 days (about one usable revisit) count as simultaneous.

(a) Provenance classes. Each exceedance with observed upstream and own history
    is classified as:
      concurrent   upstream also exceeds now
      lagged       upstream exceeded earlier in the window, not now
      build-up     no upstream signal, the reach was already rising (>= 0.7 x threshold)
      fresh spike  no upstream signal and no prior rise
    Shares are printed for all exceedances and for facility-adjacent ones
    (reach within 1.5 km of an OSM facility).

(b) The control. Is the flow-upstream set co-elevated more often than an
    equally sized set of the nearest observed reaches that are not flow-connected
    (none within 6 hops up or down)? Paired per exceedance, both sets observed,
    reach-cluster bootstrap on the difference.

    python scripts/exp_provenance.py --region ro
    python scripts/exp_provenance.py --region de   # reads the split .gz parts directly
"""
import argparse
import bisect
import csv
from collections import Counter, defaultdict
from datetime import date

import numpy as np
from scipy.spatial import cKDTree

import _bootstrap
from experiments.graph import RiverGraph
from experiments.osm import FacilityIndex

REGIONS = {
    "de": (_bootstrap.data_de, "segment_graph_de.json",
           "turb_perdate_national_full.csv", "osm_industrial_de.json"),
    "ro": (_bootstrap.data_ro, "segment_graph_ro.json",
           "turb_perdate_ro_full.csv", "osm_industrial_ro.json"),
}
CLASSES = ("concurrent", "lagged", "build-up", "fresh spike")


def load_passes(path, in_graph):
    by_reach, allvals = defaultdict(list), []
    with _bootstrap.open_text(path) as fh:
        for row in csv.DictReader(fh):
            oid = row["object_id"]
            if oid not in in_graph:
                continue
            try:
                v = float(row["turb"])
            except (ValueError, KeyError):
                continue
            by_reach[oid].append((date.fromisoformat(row["acquired_at"][:10]).toordinal(), v))
            allvals.append(v)
    ords, vals = {}, {}
    for oid, s in by_reach.items():
        s.sort()
        ords[oid] = [t for t, _ in s]
        vals[oid] = [v for _, v in s]
    return ords, vals, allvals


def unit_xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def print_control(pairs, label, rng, n_boot):
    """Flow vs control co-elevation rates and a reach-cluster bootstrap CI."""
    reach = np.array([p[0] for p in pairs])
    flow = np.array([p[1] for p in pairs], bool)
    ctl = np.array([p[2] for p in pairs], bool)
    rows_of = defaultdict(list)
    for i, r in enumerate(reach):
        rows_of[r].append(i)
    uniq = sorted(rows_of)
    boot = []
    for _ in range(n_boot):
        idx = np.concatenate([rows_of[u] for u in rng.choice(uniq, len(uniq), replace=True)])
        boot.append(flow[idx].mean() - ctl[idx].mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {label}: n={len(pairs)} reaches={len(uniq)}  flow {flow.mean():.1%}  "
          f"control {ctl.mean():.1%}  delta {100 * (flow.mean() - ctl.mean()):+.1f} pp "
          f"[{100 * lo:+.1f}, {100 * hi:+.1f}]")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", choices=sorted(REGIONS), default="ro")
    ap.add_argument("--pct", type=float, default=95.0)
    ap.add_argument("--hops", type=int, default=3, help="upstream neighbourhood depth")
    ap.add_argument("--excl", type=int, default=6,
                    help="control reaches must not be flow-connected within this many hops")
    ap.add_argument("--tol", type=int, default=6, help="days counted as simultaneous")
    ap.add_argument("--lookback", type=int, default=30, help="history window (days)")
    ap.add_argument("--rise", type=float, default=0.7)
    ap.add_argument("--radius", type=float, default=1500.0, help="facility radius (m)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data, gname, pname, oname = REGIONS[args.region]
    g = RiverGraph.from_json(data(gname))
    fac = FacilityIndex.from_json(data(oname))
    ords, vals, allvals = load_passes(data(pname), set(g.nodes))
    thr = float(np.percentile(allvals, args.pct))
    print(f"region={args.region} reaches={len(ords)} passes={len(allvals)} "
          f"thr(p{args.pct:g})={thr:.0f} hops={args.hops} tol=+/-{args.tol}d "
          f"lookback={args.lookback}d")

    def window(oid, lo, hi):
        """(exceeds threshold, n passes) for reach oid with lo <= day <= hi."""
        O = ords.get(oid)
        if not O:
            return False, 0
        a, b = bisect.bisect_left(O, lo), bisect.bisect_right(O, hi)
        return (b > a and max(vals[oid][a:b]) >= thr), b - a

    def any_window(ids, lo, hi):
        hit, n = False, 0
        for u in ids:
            h, k = window(u, lo, hi)
            hit |= h
            n += k
        return hit, n

    upstream = {r: [x["object_id"] for x in g.walk_upstream(r, max_depth=args.hops)
                    if x["object_id"] in ords] for r in ords}
    near_fac = {r: bool(fac.nearby(g[r]["lat"], g[r]["lon"], radius_m=args.radius, limit=1))
                for r in ords}

    # Nearest observed reaches that are not flow-connected (the control set).
    obs = list(ords)
    tree = cKDTree(unit_xyz([g[o]["lat"] for o in obs], [g[o]["lon"] for o in obs]))

    def control_set(r, k):
        excl = {r}
        excl.update(x["object_id"] for x in g.walk_upstream(r, max_depth=args.excl))
        excl.update(x["object_id"] for x in g.walk_downstream(r, max_depth=args.excl))
        _, nbr = tree.query(unit_xyz([g[r]["lat"]], [g[r]["lon"]])[0],
                            k=min(k + len(excl) + 5, len(obs)))
        return [obs[j] for j in np.atleast_1d(nbr) if obs[j] not in excl][:k]

    cls_all, cls_fac = Counter(), Counter()
    headwater = insufficient = 0
    prior_passes = []  # own passes in the window before each exceedance
    pairs = []  # (reach, flow co-elevated, control co-elevated)
    for r, O in ords.items():
        V, U = vals[r], upstream[r]
        P = control_set(r, len(U)) if U else []
        for k, o in enumerate(O):
            if V[k] < thr:
                continue
            _, own_n = window(r, o - args.lookback, o - 1)
            prior_passes.append(own_n)
            if not U:
                headwater += 1
                continue
            # (a) provenance class
            own_max = max(V[bisect.bisect_left(O, o - args.lookback):k], default=0.0)
            now_hit, now_n = any_window(U, o - args.tol, o + args.tol)
            prev_hit, prev_n = any_window(U, o - args.lookback, o - args.tol - 1)
            if now_n + prev_n == 0 or own_n == 0:
                insufficient += 1
            else:
                c = ("concurrent" if now_hit else "lagged" if prev_hit
                     else "build-up" if own_max >= args.rise * thr else "fresh spike")
                cls_all[c] += 1
                if near_fac[r]:
                    cls_fac[c] += 1
            # (b) flow vs control co-elevation, both sets observed
            if P:
                ctl_hit, ctl_n = any_window(P, o - args.tol, o + args.tol)
                if now_n and ctl_n:
                    pairs.append((r, now_hit, ctl_hit))

    n_all, n_fac = sum(cls_all.values()), sum(cls_fac.values())
    print(f"\nown passes in the {args.lookback} days before an exceedance: "
          f"median {np.median(prior_passes):.0f}")
    print(f"\n(a) provenance: classified={n_all} (facility-adjacent {n_fac}) "
          f"headwater={headwater} insufficient={insufficient}")
    print(f"  {'class':12s} {'all':>6s} {'facility-adj':>13s}")
    for c in CLASSES:
        print(f"  {c:12s} {cls_all[c] / max(n_all, 1):6.1%} {cls_fac[c] / max(n_fac, 1):13.1%}")

    rng = np.random.default_rng(args.seed)
    print("\n(b) co-elevation control (paired, both sets observed)")
    print_control([p for p in pairs], "all", rng, args.n_boot)
    print_control([p for p in pairs if near_fac[p[0]]], "facility-adjacent", rng, args.n_boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
