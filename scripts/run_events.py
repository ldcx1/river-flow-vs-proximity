"""Per-pass match of Sentinel-2 to in-situ readings (paper: NDTI vs ammonium).

For each in-situ reading (2020-2023), takes the nearest cloud-free pass on the
same segment within +/-7 days. Reports match rates by window, Spearman
correlation of optical indices with each determinand at +/-3 days, and ROC-AUC
of risk_score for exceedance at +/-1, 3 and 7 days. Needs scipy. The matched
events table is written only with --out-csv and the figure only with --out-fig
(the figure needs matplotlib, the ``figures`` extra).

    python scripts/run_events.py > results/national/events.txt
    python scripts/run_events.py --out-csv events_matched_de.csv \\
        --out-fig fig_events.pdf
"""
import argparse
import csv
from collections import defaultdict
from datetime import date

import numpy as np
from scipy.stats import spearmanr

import _bootstrap
from experiments.metrics import roc_auc  # noqa: E402  (tie-aware; risk_score is heavily tied)

INDICES = ["NDWI", "MNDWI", "NDVI", "NDTI", "NDCI", "BSI", "TURBIDITY", "risk_score"]
RULES = {"Dissolved oxygen": ("lt", 6.0, "mg/L"), "BOD5": ("gt", 6.0, "mg{O2}/L"),
         "Ammonium": ("gt", 0.4, "mg{NH4}/L"), "Nitrite": ("gt", 0.1, "mg{NO2}/L"),
         "Nitrate": ("gt", 25.0, "mg{NO3}/L"), "Chlorophyll a": ("gt", 25.0, "ug/L")}
WINDOWS = [1, 3, 7]


def d_iso(s):
    return date(int(s[:4]), int(s[5:7]), int(s[8:10]))


def d_compact(s):
    return date(int(s[:4]), int(s[4:6]), int(s[6:8])) if "-" not in s else d_iso(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pass-csv", default=_bootstrap.data_de("pass_obs_national.csv"),
                    help="per-pass Sentinel-2 indices CSV")
    ap.add_argument("--insitu", default=_bootstrap.data_de("insitu_readings_national_de.csv"),
                    help="in-situ readings CSV")
    ap.add_argument("--out-csv", default=None,
                    help="write the matched events table to this CSV")
    ap.add_argument("--out-fig", default=None,
                    help="write the two-panel figure to this path (PDF/PNG)")
    args = ap.parse_args()

    passes = defaultdict(list)
    for r in csv.DictReader(_bootstrap.open_text(args.pass_csv)):
        rec = {k: (float(r[k]) if r.get(k) not in (None, "") else None) for k in INDICES}
        passes[r["object_id"]].append((d_iso(r["date"]), rec))
    for s in passes:
        passes[s].sort()
    npass = sum(len(v) for v in passes.values())
    print(f"pass observations: {npass} over {len(passes)} segments")

    def nearest(seg, day, win):
        best, blag = None, win + 1
        for pd_, rec in passes.get(seg, ()):
            lag = abs((pd_ - day).days)
            if lag <= win and lag < blag:
                best, blag = rec, lag
        return best, blag

    agg = defaultdict(list)
    n_raw = 0
    for r in csv.DictReader(_bootstrap.open_text(args.insitu)):
        if r["determinand"] not in RULES or not ("2020" <= r["date"][:4] <= "2023"):
            continue
        try:
            agg[(r["segment"], r["date"], r["determinand"])].append(float(r["value"]))
        except ValueError:
            pass
        else:
            n_raw += 1
    insitu = [(seg, d_compact(dt), det, float(np.mean(vs)))
              for (seg, dt, det), vs in agg.items()]
    print(f"in-situ readings (rule determinands, 2020-2023): {len(insitu)} "
          f"({n_raw - len(insitu)} co-located same-day duplicates collapsed)")

    matched = []
    for seg, day, det, val in insitu:
        rec, lag = nearest(seg, day, max(WINDOWS))
        if rec is None:
            continue
        direction, thr, _ = RULES[det]
        exc = int(val < thr) if direction == "lt" else int(val > thr)
        matched.append({"segment": seg, "insitu_date": day.isoformat(), "lag": lag,
                        "determinand": det, "value": val, "exceed": exc, **rec})
    print(f"matched events (<= {max(WINDOWS)}d): {len(matched)} "
          f"({len(matched)/max(1,len(insitu)):.0%} of readings)")

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(matched[0].keys()))
            w.writeheader(); w.writerows(matched)

    print("\nmatch rate by window:")
    rates = []
    for win in WINDOWS:
        sub = [m for m in matched if m["lag"] <= win]
        lags = sorted(m["lag"] for m in sub)
        rates.append(len(sub))
        print(f"  +/-{win}d: {len(sub)} events ({len(sub)/max(1,len(insitu)):.0%}), "
              f"median lag {lags[len(lags)//2] if lags else '-'}d")

    print("\nSpearman (in-situ value vs satellite index, +/-3d):")
    print(f"  {'determinand':<16} {'n':>4}  " + "  ".join(f"{i:>9}" for i in ["NDTI", "NDCI", "TURBIDITY", "risk_score"]))
    for det in RULES:
        sub = [m for m in matched if m["determinand"] == det and m["lag"] <= 3]
        if len(sub) < 10:
            print(f"  {det:<16} {len(sub):>4}  (too few)"); continue
        vals = np.array([m["value"] for m in sub])
        cells = []
        for idx in ["NDTI", "NDCI", "TURBIDITY", "risk_score"]:
            iv = np.array([m[idx] if m[idx] is not None else np.nan for m in sub])
            ok = ~np.isnan(iv)
            rho = spearmanr(vals[ok], iv[ok]).statistic if ok.sum() >= 10 else float("nan")
            cells.append(f"{rho:>9.2f}")
        print(f"  {det:<16} {len(sub):>4}  " + "  ".join(cells))

    print("\nROC-AUC: satellite risk_score predicts in-situ exceedance:")
    aucs = []
    for win in WINDOWS:
        sub = [m for m in matched if m["lag"] <= win and m["risk_score"] is not None]
        y = [m["exceed"] for m in sub]; p = [m["risk_score"] for m in sub]
        a = roc_auc(y, p); aucs.append(a)
        print(f"  +/-{win}d: n={len(sub)}, positives={sum(y)} ({np.mean(y):.0%}), AUC={a:.3f}")

    if args.out_fig:
        _figs(matched, rates, aucs, args.out_fig)


def _figs(matched, rates, aucs, path):
    # Imported here so the table run needs no plotting stack.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "figure.dpi": 200, "axes.grid": True, "grid.alpha": 0.3})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    a1.bar([f"+/-{w}d" for w in WINDOWS], rates, color="#4c78a8")
    a1.set_ylabel("matched events"); a1.set_title("(a) Pass-to-sample matches by window")
    for i, r in enumerate(rates):
        a1.text(i, r, str(r), ha="center", va="bottom", fontsize=7)
    sub = [m for m in matched if m["lag"] <= 3 and m["risk_score"] is not None]
    pos = [m["risk_score"] for m in sub if m["exceed"] == 1]
    neg = [m["risk_score"] for m in sub if m["exceed"] == 0]
    a2.boxplot([neg, pos], tick_labels=["no exceedance", "exceedance"], showfliers=False)
    a2.set_ylabel("satellite risk score (0-7)")
    a2.set_title(f"(b) Pass risk vs in-situ event (+/-3d, AUC={aucs[1]:.2f})")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
