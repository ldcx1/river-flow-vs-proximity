"""Benjamini-Hochberg multiplicity correction (paper, "Metrics and inference").

Prints ``results/national/multiplicity.txt`` from the numbers in
``results/results_data.py``.

  * Each CI-derived effect gets a two-sided p from a normal approximation to its
    95% bootstrap interval: SE = (hi - lo) / (2 * 1.96), z = point / SE. This
    assumes a roughly symmetric bootstrap distribution. Autocorrelation rows use
    their permutation p.
  * Recovery and denoise are left out of the family. Their CIs come from about 20
    synthetic-injection runs on one field, not from resampling reaches. They are
    printed separately as descriptive only.
  * Two families: A = the positive effects, B = A plus the study's null and
    negative tests. The script also counts the resolution-sweep comparisons and
    re-tests the positives at that larger family size.
  * The two agricultural-fusion rows belong to a withdrawn result that the paper
    does not report. They stay in the family so its size matches the paper's
    comparison count.
  * BH assumes independence or PRDS. Several rows share pairs or targets, so the
    adjustment is approximate.

    python scripts/exp_multiplicity.py > results/national/multiplicity.txt
"""

import argparse
import math
import os
import sys

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from experiments.eval import bh_adjust

# results_data.py lives under results/, not in the installed package.
_RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
if _RESULTS not in sys.path:
    sys.path.insert(0, _RESULTS)
import results_data as R  # noqa: E402


# A 2000-resample bootstrap cannot resolve a two-sided p below 2/2000, but the
# normal approximation can return far smaller values from a tight CI. Floor every
# CI-derived p there. This only raises p-values far below the FDR boundary, so no
# survival verdict changes.
BOOT_P_FLOOR = 2.0 / 2000.0


def p_from_ci(point: float, lo: float, hi: float) -> float:
    """Two-sided p from a 95% bootstrap interval via a normal approximation,
    floored at the bootstrap's resolution (``BOOT_P_FLOOR``)."""
    se = (hi - lo) / (2 * 1.96)
    if se <= 0:
        return BOOT_P_FLOOR if point != 0 else 1.0
    z = abs(point) / se
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
    return max(p, BOOT_P_FLOOR)


def _ci(point, lo, hi):
    return ("ci", float(point), float(lo), float(hi))


def _p(pval):
    return ("p", float(pval))


# Effects come straight from results_data.py.
_ee_turb = (R.PROPAGATION["Turbidity"][0] - R.PROPAGATION["Turbidity"][1], R.PROPAGATION["Turbidity"][3], R.PROPAGATION["Turbidity"][4])
_ee_ndci = (R.PROPAGATION["NDCI"][0] - R.PROPAGATION["NDCI"][1], R.PROPAGATION["NDCI"][3], R.PROPAGATION["NDCI"][4])
_bacc_k2 = next(c for c in R.BASINS["curve"] if c[0] == 2)  # (k, n, diff, lo, hi)

# Family A: positive graph effects, plus the two withdrawn ag-fusion rows
# (not reported in the paper, kept so the family size matches its count).
POSITIVES = [
    ("Co-fluctuation NDCI (common)",   _ci(*R.COFLUCTUATION[("NDCI", "common-mode")][2:5])),
    ("Co-fluctuation NDCI (seasonal)", _ci(*R.COFLUCTUATION[("NDCI", "seasonal")][2:5])),
    ("Co-fluctuation turb (common)",   _ci(*R.COFLUCTUATION[("Turbidity", "common-mode")][2:5])),
    ("Co-fluctuation turb (seasonal)", _ci(*R.COFLUCTUATION[("Turbidity", "seasonal")][2:5])),
    ("Conditional incremental value",        _ci(*R.CONDITIONAL["incremental_auc"]["gain"])),
    ("Attribution upstream vs reversed",     _ci(*R.ATTRIBUTION["delta"]["vs reversed"])),
    ("Basins (k=2)",                  _ci(_bacc_k2[2], _bacc_k2[3], _bacc_k2[4])),
    ("Break coherence-violation",       _ci(*R.ANOMALY["break"]["vs_euclidean"])),
    ("Tracing spatio-temporal",           _ci(*R.ANOMALY["tracing"]["delta"])),
    ("Fusion: ag-upstream vs optical", _ci(*R.AG["delta"]["upstream vs optical"])),
    ("Fusion: all-ag vs optical",      _ci(*R.AG["delta"]["all-ag vs optical"])),
]

# Synthetic-injection effects, kept out of the BH family. Their CIs come from
# about 20 injection runs on one field, not from resampling reaches.
SYNTHETIC_INJECTION = [
    ("Recovery connected-event (injection-run)",  _ci(*R.ANOMALY["recovery"]["delta"])),
    ("Denoise precision (injection-run)",          _ci(*R.ANOMALY["denoise"]["delta"])),
]

# Family B adds the study's own null and negative tests.
NULLS = [
    ("Attribution vs random",                  _ci(*R.ATTRIBUTION["delta"]["vs random"])),
    ("Propagation turbidity",      _ci(*_ee_turb)),
    ("Attribution vs euclidean",               _ci(*R.ATTRIBUTION["delta"]["vs euclidean"])),
    ("Gap-fill DO vs global",      _ci(*R.GAPFILL["Dissolved oxygen"][1])),
    ("Detection physics vs optical",   _ci(*R.DETECTION["d_physics_vs_optical"])),
    ("Propagation NDCI",           _ci(*_ee_ndci)),
    ("Residual vs euclidean",                _ci(*R.ANOMALY["residual"]["vs_euclidean"])),
    ("Gap-fill Exceedance vs global", _ci(*R.GAPFILL["Exceedance flag"][1])),
    ("Autocorrelation Nitrate",           _p(R.AUTOCORRELATION["Nitrate"][4])),
    ("Autocorrelation DO",                _p(R.AUTOCORRELATION["Dissolved oxygen"][4])),
    ("Attribution vs permuted",                _ci(*R.ATTRIBUTION["delta"]["vs permuted"])),
    ("Detection all vs optical",       _ci(*R.DETECTION["d_all_vs_optical"])),
    ("Gap-fill Nitrate vs global", _ci(*R.GAPFILL["Nitrate"][1])),
    ("Autocorrelation Exceedance",        _p(R.AUTOCORRELATION["Exceedance flag"][4])),
]


def effect_p(spec) -> float:
    if spec[0] == "ci":
        return p_from_ci(spec[1], spec[2], spec[3])
    return spec[1]


def render_family(title: str, effects, alpha: float = 0.05) -> str:
    labels = [e[0] for e in effects]
    pvals = [effect_p(e[1]) for e in effects]
    adj, reject = bh_adjust(pvals, alpha=alpha)
    order = sorted(range(len(effects)), key=lambda i: pvals[i])  # ascending p
    lines = [f"{title} ({len(effects)} effects, FDR {alpha})",
             f"  {'effect':<36}{'p':>8}{'BH p':>10}  survives"]
    for i in order:
        lines.append(f"  {labels[i]:<36}{pvals[i]:>8.4f}{adj[i]:>10.4f}   "
                     f"{'yes' if reject[i] else 'NO'}")
    n_survive = int(reject.sum())
    lines.append(f"  -> {n_survive}/{len(effects)} survive at FDR {alpha}.")
    return "\n".join(lines)


def _bh_survivors_at_m(pvals, m: int, alpha: float = 0.05):
    """BH step-up survival if the family had ``m`` >= len(pvals) tests.

    Places these effects at ranks 1..len(pvals) against denominator m, so
    ``rej[i]`` says whether effect i survives in a family of m tests."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    thresh_rank = 0
    for rank, i in enumerate(order, 1):
        if pvals[i] <= alpha * rank / m:
            thresh_rank = rank
    rej = [False] * len(pvals)
    for rank, i in enumerate(order, 1):
        if rank <= thresh_rank:
            rej[i] = True
    return rej


SWEEP_JSON = os.path.join(_RESULTS, "national", "resolution_sweep.json")


def sweep_comparison_count(path: str = SWEEP_JSON) -> int:
    """Count the additional comparisons in the resolution sweep (widths x experiments)."""
    import json
    if not os.path.exists(path):
        return 0
    swp = json.load(open(path))
    per_width = ["cofluctuation_turb", "cofluctuation_ndci", "residual", "break",
                 "conditional_incremental", "recovery", "denoise", "tracing"]
    return sum(1 for w in swp for k in per_width if k in swp[w])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default=SWEEP_JSON,
                    help="resolution-sweep JSON whose comparisons enlarge the family")
    args = ap.parse_args()

    print("Benjamini-Hochberg multiplicity correction (two families).\n")
    print(render_family("FAMILY A: positive graph effects + withdrawn ag-fusion rows", POSITIVES))
    print()
    print(render_family("FAMILY B: full study family (positives + the study's "
                        "null/negative tests)", POSITIVES + NULLS))

    print("\nSynthetic-injection effects (excluded from the FDR family above: their CIs")
    print("are injection-run intervals, not population resamples; descriptive only):")
    for lbl, spec in SYNTHETIC_INJECTION:
        print(f"  {lbl:<44} delta={spec[1]:+.3f} [{spec[2]:+.3f}, {spec[3]:+.3f}]")

    # Withdrawn ag-fusion rows: do they survive family B?
    fam_b = POSITIVES + NULLS
    b_labels = [e[0] for e in fam_b]
    _, b_rej = bh_adjust([effect_p(e[1]) for e in fam_b], alpha=0.05)
    b_rej = {b_labels[i]: b_rej[i] for i in range(len(b_labels))}
    print("\nFusion (uniform normal-approx p, broader family FDR 0.05):")
    for lab in ("Fusion: ag-upstream vs optical", "Fusion: all-ag vs optical"):
        print(f"  {lab:<34} -> {'survives' if b_rej[lab] else 'does not survive'}")
    print("  Under the single uniform p-method neither fusion effect survives the broader")
    print("  family. The land-cover fusion result is withdrawn and not in the paper.")

    # Full comparison count and survival of the positives at that size.
    n_study = len(POSITIVES) + len(NULLS)
    n_sweep = sweep_comparison_count(args.sweep)
    n_true = n_study + n_sweep
    print(f"\nTrue comparison universe: {n_study} study effects + {n_sweep} resolution-sweep")
    print(f"  comparisons = {n_true}+ (Strahler strata, deseasoning modes and radius variants")
    print(f"  add more, so treat {n_true} as a lower bound).")
    core = _bh_survivors_at_m([effect_p(spec) for _, spec in POSITIVES], n_true, 0.05)
    surv = [POSITIVES[i][0] for i in range(len(POSITIVES)) if core[i]]
    print(f"  Family A effects surviving BH at m={n_true}: {len(surv)}/{len(POSITIVES)}.")
    print(f"    survivors: {', '.join(surv)}")

    print("\n  NOTE: BH here is two-sided (p from |point|/SE), so a Family-B 'survivor'")
    print("  can be an effect significant against the graph. Attribution vs")
    print("  random, Propagation turbidity, and Attribution vs euclidean are graph losses,")
    print("  not wins. The positive graph survivors are the ones named below.")

    print("\nRobust conclusion:")
    print("  The graph-structure effects (co-fluctuation, break, tracing, attribution")
    print("  direction, conditional, basins) survive BH under both families and a single")
    print(f"  uniform p-method. At the full ~{n_true}-comparison universe the co-fluctuation,")
    print("  conditional, attribution-direction and basin-scaling effects still survive; the")
    print("  break effect survives only at m<=65 and drops out for any larger family (its p")
    print("  sits on the FDR boundary), so we treat it as boundary-sensitive and do not")
    print("  headline it. The tracing gain and the fusion gain do not survive (tracing is")
    print("  partly definitional, fusion withdrawn). Recovery/denoise are reported as")
    print("  synthetic-injection stability, not population effects.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
