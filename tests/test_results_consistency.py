"""Consistency checks for results/results_data.py.

The chain paper <-> results_data <-> results/national/*.txt is maintained by hand.
These tests catch a number edited in one place but not another, and pin the
headline verdicts.
"""
import os
import sys

_RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
sys.path.insert(0, _RESULTS)
import results_data as R  # noqa: E402


def test_strong_cell_panel_internally_consistent():
    # 8,100 labelled reach-months with ~994 positives. The GNN uses the same cells.
    assert abs(R.DATA["strong_cells"] * R.DATA["strong_pos_frac"] - 994) < 5
    assert R.GNN["n_cells"] == R.DATA["strong_cells"]


def test_detection_prevalence_matches_rows():
    # 978 positives / 7739 rows = 12.6% (a subset of the 8,100 labelled cells)
    assert abs(R.DETECTION["prevalence"] - 978 / R.DETECTION["n_rows"]) < 0.002


def test_gnn_gat_does_not_beat_no_edge_mlp():
    # Table II: the GAT does not beat the no-edge MLP on ROC or PR
    assert R.GNN["gat_directed"][0] <= R.GNN["mlp_no_edges"][0]
    assert R.GNN["gat_directed"][1] <= R.GNN["mlp_no_edges"][1]


def test_event_riskscore_near_chance():
    # the per-pass index risk score is near chance at every day window
    for auc in R.EVENTS["riskscore_auc"].values():
        assert 0.50 <= auc <= 0.53


def test_fusion_graph_specific_gain_includes_zero():
    # Withdrawn ag-fusion result. Its graph-specific contrast (upstream vs local)
    # must keep a CI that includes 0, so it cannot return as a finding.
    pt, lo, hi = R.AG["delta"]["upstream vs local"]
    assert lo < 0 < hi


def test_ci_triples_well_formed():
    triples = (list(R.AG["delta"].values()) + list(R.ATTRIBUTION["delta"].values()) + [
        R.ANOMALY["residual"]["vs_euclidean"], R.ANOMALY["break"]["vs_euclidean"],
        R.ANOMALY["recovery"]["delta"], R.ANOMALY["denoise"]["delta"], R.ANOMALY["tracing"]["delta"],
        R.CONDITIONAL["incremental_auc"]["gain"],
        R.DETECTION["d_physics_vs_optical"], R.DETECTION["d_all_vs_optical"],
    ])
    for pt, lo, hi in triples:
        assert lo <= hi, (pt, lo, hi)
        assert lo - 1e-9 <= pt <= hi + 1e-9, (pt, lo, hi)


# ---- Headline numbers (catch drift in results_data.py) -------------------------
# Hard-coded values quoted in the paper or its logs. Editing one in results_data.py
# without updating the paper fails a test.

def test_detection_optical_headline_pinned():
    roc, _, pr, _ = R.DETECTION["optical"]
    assert roc == 0.605 and pr == 0.183          # Sec. V-A and Table II, optical only


def test_cofluctuation_diffs_pinned():
    # Table III: NDCI +0.091, turbidity +0.029 (common-mode)
    assert R.COFLUCTUATION[("NDCI", "common-mode")][2] == 0.091
    assert R.COFLUCTUATION[("Turbidity", "common-mode")][2] == 0.029


def test_proxy_ammonium_turbidity_pinned():
    r, lo, hi, n = R.PROXY["Ammonium~turbidity"]
    assert r == 0.204 and (lo, hi) == (0.157, 0.248) and n == 6083   # Sec. V-A


def test_romania_replication_matches_germany():
    # Label-free suite re-run on Romania (Table III). Co-fluctuation is significant
    # under both controls, the static turbidity field ties, and the synthetic
    # recovery/denoise gains and the basin-accretion result replicate.
    r = R.ROMANIA_REPL
    # co-fluctuation: every CI above 0, distance- and bearing-matched
    for d in (r["cofluctuation"], r["cofluctuation_orient"]):
        for c_flow, c_euc, diff, lo, hi, p, n in d.values():
            assert diff > 0 and lo > 0
    # NDCI common-mode effect at least as large as Germany's
    assert r["cofluctuation"][("NDCI", "common-mode")][2] >= R.COFLUCTUATION[("NDCI", "common-mode")][2]
    # static optical field: turbidity a tie (CI includes 0), NDCI only a small flow edge
    assert r["propagation"]["Turbidity"][4] <= 0 <= r["propagation"]["Turbidity"][5]
    assert r["propagation"]["NDCI"][4] > 0
    # synthetic recovery and denoise gains exclude 0
    assert r["anomaly"]["recovery"]["delta"][1] > 0
    assert r["anomaly"]["denoise"]["delta"][1] > 0
    # basin accretion: significant at every basin count
    for k, nseg, diff, lo, hi in r["basins"]:
        assert lo > 0


def test_fusion_upstream_gain_excludes_zero():
    # Withdrawn ag-fusion result (not in the paper). Pins the logged delta that
    # the multiplicity family in exp_multiplicity.py counts.
    pt, lo, hi = R.AG["delta"]["upstream vs optical"]
    assert pt == 0.065 and lo > 0


def test_static_signs_hold():
    # in-situ Moran's I: flow >= Euclidean for every determinand
    assert all(v[3] >= 0 for v in R.AUTOCORRELATION.values())
    # static optical field: Euclidean beats flow for turbidity (diff CI below 0)
    assert R.PROPAGATION["Turbidity"][3] < 0 and R.PROPAGATION["Turbidity"][4] < 0


def test_ensemble_calibration_claim_pinned():
    # Isotonic calibration cuts ECE from ~0.19 to ~0.03 (ensemble.txt).
    ece_single = R.ENSEMBLE["single"][3]
    ece_isotonic = R.ENSEMBLE["ensemble+isotonic"][3]
    assert round(ece_single, 2) == 0.19 and round(ece_isotonic, 2) == 0.03


def test_data_inventory_headline_numbers_pinned():
    # Table I and abstract: 14.6M per-pass observations, 66,261 reaches, 113,619
    # facilities. 14.6M has no log in the package, so pin it here.
    assert R.DATA["s2_pass_obs_M"] == 14.6
    assert R.DATA["graph_segments"] == 66261 and R.DATA["facilities"] == 113619


def test_cofluctuation_bearing_control_survives():
    # The co-fluctuation effect survives the bearing-matched control: every CI is
    # above 0 and the common-mode effects keep most of their size.
    for c_flow, c_euc, diff, lo, hi, p, n in R.COFLUCTUATION_ORIENT.values():
        assert diff > 0 and lo > 0
    assert R.COFLUCTUATION_ORIENT[("NDCI", "common-mode")][2] >= 0.08      # distance-matched: 0.091
    assert R.COFLUCTUATION_ORIENT[("Turbidity", "common-mode")][2] >= 0.02  # distance-matched: 0.029


def test_basin_scaling_range_matches_paper():
    # Table III scaling row: min-max of the basin-accretion curve per country.
    de = [c[2] for c in R.BASINS["curve"]]
    ro = [c[2] for c in R.ROMANIA_REPL["basins"]]
    assert (round(min(de), 3), round(max(de), 3)) == (0.012, 0.035)
    assert (round(min(ro), 3), round(max(ro), 3)) == (0.029, 0.068)


# ---- Directional claims and point estimates ------------------------------------
# A CI well-formedness check alone would let a sign flip pass. These pin the claims.

def test_anomaly_suite_directional_claims_pinned():
    # Break, recovery, denoise and tracing CIs exclude 0 (the first three are graph
    # wins in Table III). The residual score ties Euclidean (CI includes 0).
    assert R.ANOMALY["break"]["vs_euclidean"][1] > 0          # +0.007 [+0.002,+0.012]
    assert R.ANOMALY["recovery"]["delta"][1] > 0                 # +0.019 [+0.013,+0.025]
    assert R.ANOMALY["denoise"]["delta"][1] > 0                 # +0.007 [+0.005,+0.009]
    assert R.ANOMALY["tracing"]["delta"][1] > 0                 # +0.050 [+0.017,+0.092]
    assert R.ANOMALY["residual"]["vs_euclidean"][1] < 0 < R.ANOMALY["residual"]["vs_euclidean"][2]
    # point estimates (recovery F1 levels are in Table III's footnote)
    assert R.ANOMALY["recovery"]["flow"] == 0.059 and R.ANOMALY["recovery"]["euc"] == 0.040
    assert R.ANOMALY["tracing"]["static_top1"] == 0.317 and R.ANOMALY["tracing"]["st_top1"] == 0.367
    assert round(R.ANOMALY["tracing"]["st_top1"] - R.ANOMALY["tracing"]["static_top1"], 3) == 0.050
    assert R.ANOMALY["tracing"]["mrr_static"] == 0.449 and R.ANOMALY["tracing"]["mrr_euc"] == 0.112


def test_eb_attribution_headline_pinned():
    # Unrestricted source ranking (ATTRIBUTION.txt). Sec. IV quotes the two deltas.
    assert R.ATTRIBUTION["mrr"]["graph"] == 0.585 and R.ATTRIBUTION["mrr"]["random"] == 0.723
    assert R.ATTRIBUTION["hit1"]["graph"] == 0.512
    assert R.ATTRIBUTION["delta"]["vs reversed"][1] > 0       # beats reversed flow, +0.056
    assert R.ATTRIBUTION["delta"]["vs euclidean"][2] < 0      # loses to proximity, -0.041


def test_cofluctuation_orient_exact_pinned():
    # bearing-matched control, common-mode (COFLUCTUATION_orient.txt)
    assert R.COFLUCTUATION_ORIENT[("NDCI", "common-mode")][2:5] == (0.102, 0.083, 0.120)
    assert R.COFLUCTUATION_ORIENT[("Turbidity", "common-mode")][2:5] == (0.026, 0.015, 0.038)
    assert R.COFLUCTUATION_ORIENT[("NDCI", "common-mode")][6] == 1098     # n drops 1908 -> 1098


def test_events_headline_pinned():
    # 40/67/87% of readings matched to a pass within +-1/3/7 days. The per-pass
    # NDTI vs ammonium Spearman 0.34 is quoted in Sec. V-A.
    assert R.EVENTS["match_rate"] == {1: 0.40, 3: 0.67, 7: 0.87}
    assert R.EVENTS["spearman_pm3d"]["Ammonium~NDTI"] == 0.34


def test_robust_decoupled_decay_excludes_zero():
    # The neighbour-break win survives a decoupled injection decay and a
    # decay-weighted Euclidean control (CIs exclude 0).
    for key in ("break_matched", "break_decoupled", "break_euc_decay"):
        assert R.ROBUST[key][1] > 0, key


def test_ensemble_secondary_numbers_pinned():
    # Pin the isotonic and selective ROC too (xgboost 3.3.0, scikit-learn 1.9.0)
    assert R.ENSEMBLE["ensemble+isotonic"][0] == 0.576
    assert R.ENSEMBLE["selective_roc_50pct"] == 0.588


def test_ensemble_log_matches_pins():
    # ensemble.txt is the captured stdout of scripts/run_ensemble.py
    text = _log_text("ensemble.txt")
    for key, label in (("single", "single XGBoost"), ("ensemble", "ensemble (mean)"),
                       ("ensemble+isotonic", "ensemble + isotonic cal.")):
        row = next(l for l in text.splitlines() if l.startswith(label))
        vals = tuple(float(v) for v in row[len(label):].split())
        assert vals == R.ENSEMBLE[key], (key, vals)
    row50 = next(l for l in text.splitlines() if l.strip().startswith("50%"))
    assert float(row50.split()[3]) == R.ENSEMBLE["selective_roc_50pct"]


# ---- Multiplicity (Benjamini-Hochberg FDR, Sec. IV) -----------------------------
# scripts/exp_multiplicity.py writes multiplicity.txt. The committed log must equal
# its output, and the FDR verdicts must not flip.

def _load_multiplicity_module():
    import importlib.util
    scripts = os.path.join(os.path.dirname(__file__), "..", "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    path = os.path.join(scripts, "exp_multiplicity.py")
    spec = importlib.util.spec_from_file_location("exp_multiplicity", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_multiplicity_family_excludes_injection_run_effects():
    # Recovery and denoise CIs resample ~20 injection runs on one field, not the
    # river population. They must stay out of the FDR family.
    m = _load_multiplicity_module()
    fam_labels = [e[0] for e in (m.POSITIVES + m.NULLS)]
    assert not any(("Recovery" in l or "Denoise" in l) for l in fam_labels), fam_labels
    synth = [e[0] for e in m.SYNTHETIC_INJECTION]
    assert any("Recovery" in l for l in synth) and any("Denoise" in l for l in synth)


def test_multiplicity_fusion_suggestive_structure_robust():
    # One p-value method for every CI effect. Under it the two withdrawn ag-fusion
    # effects fail BH, while the graph-structure effects survive.
    from experiments.eval import bh_adjust
    m = _load_multiplicity_module()
    fam = m.POSITIVES + m.NULLS
    labels = [e[0] for e in fam]
    _, reject = bh_adjust([m.effect_p(e[1]) for e in fam], alpha=0.05)
    survive = dict(zip(labels, reject))
    assert not survive["Fusion: ag-upstream vs optical"]   # BH ~0.062, fails
    assert not survive["Fusion: all-ag vs optical"]         # BH ~0.071, fails
    for lab in ("Co-fluctuation NDCI (common)", "Break coherence-violation",
                "Tracing spatio-temporal", "Attribution upstream vs reversed",
                "Conditional incremental value", "Basins (k=2)"):
        assert survive[lab], lab
    # the structure effects also survive at the full sweep-inflated family size
    core = m._bh_survivors_at_m([m.effect_p(s) for _, s in m.POSITIVES],
                                m.sweep_comparison_count() + len(fam), 0.05)
    assert sum(core) >= 6


def test_multiplicity_log_matches_generator():
    import subprocess
    scripts = os.path.join(os.path.dirname(__file__), "..", "scripts")
    out = subprocess.run([sys.executable, os.path.join(scripts, "exp_multiplicity.py")],
                         capture_output=True, text=True, check=True).stdout
    with open(os.path.join(_RESULTS, "national", "multiplicity.txt"), encoding="utf-8") as fh:
        log = fh.read()
    assert out.strip() == log.strip()


def _log_text(name):
    with open(os.path.join(_RESULTS, "national", name), encoding="utf-8") as fh:
        return fh.read()


def test_results_data_numbers_appear_in_national_logs():
    # Each number below must appear literally (3 decimals, unsigned) in its source
    # log. Signed log tokens such as "+0.091" contain the unsigned form, so a
    # substring check is enough.
    checks = {
        "detection.txt": [R.DETECTION["optical"][0], R.DETECTION["optical"][2]],
        "ag_ablation.txt": [R.AG["opt+upstream"], R.AG["delta"]["upstream vs optical"][0]],
        "COFLUCTUATION.txt": [R.COFLUCTUATION[("NDCI", "common-mode")][2], R.COFLUCTUATION[("Turbidity", "common-mode")][2]],
        "PROPAGATION.txt": [R.PROPAGATION["Turbidity"][0]],
        "gnn.txt": [R.GNN["gat_directed"][0], R.GNN["mlp_no_edges"][0]],
        "proxy.txt": [R.PROXY["Ammonium~turbidity"][0]],
        "ro_cofluctuation.txt": [R.ROMANIA_REPL["cofluctuation"][("NDCI", "common-mode")][2]],
        "ro_anomaly.txt": [R.ROMANIA_REPL["anomaly"]["recovery"]["delta"][0]],
        "AUTOCORRELATION.txt": [R.AUTOCORRELATION["Nitrate"][3]],
        "ndci_strahler.txt": [R.NDCI_STRAHLER["Headwater (1-2)"][0],
                              R.NDCI_STRAHLER["Large (5+)"][0]],
        "CONDITIONAL.txt": [R.CONDITIONAL["cofluct_strahler"]["Large (5+)"][1],
                            R.CONDITIONAL["cofluct_strahler"]["Mid (3-4)"][2]],
    }
    for log, values in checks.items():
        text = _log_text(log)
        for v in values:
            assert f"{abs(v):.3f}" in text, f"{v} (results_data) absent from {log}"


# ---- More drift guards ----------------------------------------------------------

def test_gapfill_table_row_and_mixed_sign_pinned():
    # Gap-filling vs the global mean: all three CIs include 0 (Sec. V-B quotes the
    # exceedance-flag value). Two of three favour the global mean, so this cannot be
    # reframed as a graph win.
    assert R.GAPFILL["Nitrate"][1][0] == 0.032               # vs global mean
    assert R.GAPFILL["Dissolved oxygen"][1][0] < 0           # graph worse than global mean
    assert R.GAPFILL["Exceedance flag"][1][0] < 0            # graph worse than global mean


def test_conditional_cc_numbers_pinned():
    # Conditional analyses (CONDITIONAL.txt). Only the Strahler bins are in the paper.
    assert R.CONDITIONAL["incremental_auc"]["gain"] == (0.008, 0.004, 0.012)
    assert R.CONDITIONAL["partial_moran"]["Turbidity"][2] < 0     # Euclidean wins on partial Moran
    assert all(r < 1.0 for r in R.CONDITIONAL["variogram_ratio"].values())     # flow pairs more similar
    assert R.CONDITIONAL["cofluct_strahler"]["Headwater (1-2)"][1] > 0              # headwater CI excl 0


def test_resolution_sweep_supports_paper():
    import json
    with open(os.path.join(_RESULTS, "national", "resolution_sweep.json"),
              encoding="utf-8") as fh:
        d = json.load(fh)
    # Fig. 5 and Sec. V-C: ~85% of one-day cells are empty vs ~5% at 28 days. Also
    # pins the one-day turbidity effect and NDCI staying near +0.085 at every width.
    assert round(d["1"]["cofluctuation_turb"]["diff"], 3) == 0.037
    assert 0.83 <= d["1"]["coverage"]["nan_rate"] <= 0.86
    assert d["28"]["coverage"]["nan_rate"] <= 0.10
    assert all(0.08 <= d[w]["cofluctuation_ndci"]["diff"] <= 0.092 for w in d)   # NDCI holds ~0.085

def test_ndci_strahler_advantage_grows_with_river_size():
    """The NDCI co-fluctuation advantage must grow with Strahler order and every
    bin's CI must exclude zero. Sec. V-B uses this to argue the effect is not a
    narrow-reach mixed-pixel artefact.
    """
    bins = ["Headwater (1-2)", "Mid (3-4)", "Large (5+)"]
    diffs = [R.NDCI_STRAHLER[b][0] for b in bins]
    assert diffs == sorted(diffs), "NDCI advantage must grow with river size: %s" % diffs
    for b in bins:
        lo, hi = R.NDCI_STRAHLER[b][1], R.NDCI_STRAHLER[b][2]
        assert lo > 0, "%s CI must exclude zero, got [%s, %s]" % (b, lo, hi)


def test_romania_wfd_ladder_and_transfer_reversal_pinned():
    """Romanian WFD detector (Sec. V-B): the prior lifts locally calibrated optics
    from 0.610 to 0.653 (+0.043) but hurts zero-shot transfer (-0.065).
    """
    text = _log_text("romania.txt")
    for token in ["0.610", "0.653", "+0.043", "-0.065"]:
        assert token in text, "romania.txt no longer contains %s" % token


def _provenance_log(country):
    import os
    import re
    path = os.path.join(os.path.dirname(__file__), "..", "results", "national",
                        f"provenance_{country}.txt")
    text = open(path, encoding="utf-8").read()
    shares = dict(re.findall(r"^  (concurrent|lagged|build-up|fresh spike)\s+([\d.]+)%",
                             text, re.M))
    flow, ctl, delta = re.search(
        r"^  all: .*flow ([\d.]+)%  control ([\d.]+)%  delta ([+-][\d.]+) pp", text,
        re.M).groups()
    window = int(re.search(r"before an exceedance: median (\d+)", text).group(1))
    return shares, (float(flow), float(ctl), float(delta)), window


def test_provenance_matches_logs():
    # Fig. 4: provenance classes (all classified exceedances) and the co-elevation
    # control, regenerated by scripts/exp_provenance.py.
    names = {"concurrent": 0, "fresh spike": 1, "lagged": 2, "build-up": 3}
    for country, col, key in (("de", 1, "Germany"), ("ro", 2, "Romania")):
        shares, co, window = _provenance_log(country)
        for name, row in names.items():
            assert round(float(shares[name])) == R.PROVENANCE["classes"][row][col], (country, name)
        assert window == R.PROVENANCE["passes_per_window"][key]
        if country == "de":
            assert co == R.PROVENANCE["co_elevated"]["Germany"]
        else:
            # Known drift: the paper prints 79.8 / 78.0 / +1.8 for Romania.
            assert co == R.PROVENANCE["co_elevated_regenerated"]["Romania"]


def test_basins_headline_full_set_equals_cofluctuation():
    # Same settings, same full set: the last accretion point is the headline cell.
    h = R.BASINS_HEADLINE
    assert h["de"][-1][2:] == R.COFLUCTUATION[("Turbidity", "common-mode")][2:5]
    assert h["ro"][-1][2:] == R.ROMANIA_REPL["cofluctuation"][("Turbidity", "common-mode")][2:5]


def test_facility_shares_match_log():
    import os
    import re
    path = os.path.join(os.path.dirname(__file__), "..", "results", "national",
                        "facility_shares.txt")
    text = open(path, encoding="utf-8").read()
    got = {c: float(v) / 100 for c, v in
           re.findall(r"^(Germany|Romania):.* ([\d.]+)% within", text, re.M)}
    for country, share in R.FACILITY_PROXIMITY["frac_reaches_within_1500m"].items():
        assert abs(got[country] - share) < 0.001, country
    m = re.search(r"exceedance months on such reaches: \d+/\d+ = ([\d.]+)%", text)
    assert round(float(m.group(1)) / 100, 3) == R.FACILITY_PROXIMITY["frac_exceedance_months_within_1500m_de"]


def test_inherited_share_matches_logs():
    # Sec. V-B: inherited dominates in ~90% (DE) and ~85% (RO) of facility-adjacent
    # exceedances (scripts/exp_inherited.py).
    import os
    import re
    for country, key in (("de", "Germany"), ("ro", "Romania")):
        path = os.path.join(os.path.dirname(__file__), "..", "results", "national",
                            f"inherited_{country}.txt")
        text = open(path, encoding="utf-8").read()
        share = float(re.search(r"inherited dominates .*: ([\d.]+)%", text).group(1)) / 100
        assert round(share, 2) == R.PROVENANCE["inherited_share"][key], country
