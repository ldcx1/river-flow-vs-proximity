"""Validate the shipped data bundle: schema, value ranges, and references to the
river graph. Also checks that indices.risk_score reproduces the shipped
`risk_score` column on real per-pass rows. Runs offline. A missing file skips
its test rather than failing.
"""
import csv
import json
from pathlib import Path

import pytest

from experiments.ingest import indices

DATA = Path(__file__).resolve().parents[2] / "data" / "de"
DATA_RO = Path(__file__).resolve().parents[2] / "data" / "ro"


def _rows(path):
    if not path.exists():
        pytest.skip(f"missing bundle file: {path}")
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_graph(p):
    if not p.exists():
        pytest.skip(f"{p.name} missing")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _graph():
    return _load_graph(DATA / "segment_graph_de.json")


def _graph_ro():
    return _load_graph(DATA_RO / "segment_graph_ro.json")


def test_segment_graph_schema_and_ranges():
    g = _graph()
    assert len(g) > 60_000
    need = {"next_down", "up", "strahler", "length_m", "lat", "lon", "river_name"}
    for oid, n in list(g.items())[:3000]:
        assert need <= set(n)
        assert 47.0 <= n["lat"] <= 56.0 and 5.0 <= n["lon"] <= 16.0
        assert n["strahler"] >= 1
        assert n["next_down"] is None or n["next_down"] in g   # no dangling edges


def test_strong_labels_reference_graph_and_are_binary():
    g = _graph()
    rows = _rows(DATA / "strong_labels_national_de.csv")
    reaches = {r["object_id"] for r in rows}
    assert reaches <= set(g)                        # every labeled reach exists
    assert {r["label"] for r in rows} <= {"0", "1"}


def test_pass_obs_risk_score_matches_ported_formula():
    rows = _rows(DATA / "pass_obs_national.csv")
    assert rows and set(rows[0]) >= set(indices.METRIC_FIELDS) | {"risk_score"}
    ok = bad = 0
    for r in rows:
        try:
            shipped = float(r["risk_score"])
        except (TypeError, ValueError):
            continue
        if shipped != int(shipped):     # tile-seam averaged rows: not a single scene
            continue
        means = {k: float(r[k]) for k in indices.METRIC_FIELDS if r.get(k) not in (None, "")}
        got = indices.risk_score(means)["risk_score"]
        ok += (got == int(shipped)); bad += (got != int(shipped))
    assert ok > 100                                  # plenty of single-scene rows
    assert bad / (ok + bad) < 0.02                   # formula reproduces the shipped column


def test_features_panel_has_expected_columns():
    p = DATA / "features_national_de.csv"
    if not p.exists():
        pytest.skip("features_national_de.csv missing")
    cols = next(csv.reader(open(p, encoding="utf-8")))
    for c in ["object_id", "label", "o_TURBIDITY", "o_NDCI",
              "p_expected", "t_strahler", "f_industrial"]:
        assert c in cols
    assert "o_POLLUTION" not in cols   # composite index not shipped


def test_no_panel_carries_the_composite_index():
    """No shipped feature panel may carry a composite index column. The optics are
    the seven public Sentinel-2 indices and the flow prior propagates TURBIDITY.
    """
    panels = sorted((DATA.parent).glob("*/features_*.csv"))
    if not panels:
        pytest.skip("no feature panels present")
    offenders = []
    for path in panels:
        cols = next(csv.reader(open(path, encoding="utf-8")))
        if any(c.upper().endswith("POLLUTION") for c in cols):
            offenders.append(path.name)
    assert not offenders, "composite index present in: %s" % ", ".join(offenders)


# -- Romania graph-structure bundle (results_data.ROMANIA_REPL) -------------

def test_ro_segment_graph_schema_and_ranges():
    g = _graph_ro()
    assert len(g) > 18_000
    need = {"next_down", "up", "strahler", "length_m", "lat", "lon", "river_name"}
    for oid, n in list(g.items())[:3000]:
        assert need <= set(n)
        assert 43.0 <= n["lat"] <= 49.0 and 20.0 <= n["lon"] <= 30.0   # Romania
        assert n["strahler"] >= 1
        assert n["next_down"] is None or n["next_down"] in g           # no dangling edges


def test_ro_optical_files_reference_graph_and_are_4dp():
    g = _graph_ro()
    for fname, extra in [("turb_series_ro.csv", {"ym"}),
                         ("turbidity_field_ro.csv", set())]:
        rows = _rows(DATA_RO / fname)
        assert rows and {"object_id", "turb", "ndci"} | extra <= set(rows[0])
        reaches = {r["object_id"] for r in rows}
        assert reaches <= set(g)                       # every optical reach exists in the DAG
        for r in rows[:5000]:                          # optical values rounded to 4 decimals
            for col in ("turb", "ndci"):
                v = r[col]
                if v not in ("", None):
                    frac = v.partition(".")[2]
                    assert len(frac) <= 4, f"{fname}:{col}={v} not rounded to 4dp"
