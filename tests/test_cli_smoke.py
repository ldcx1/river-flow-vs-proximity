"""Smoke tests for the experiment CLIs. A tiny synthetic graph and optical series
run end to end through the command-line entry points and must exit 0. This
catches import and argument-wiring errors that library-level tests miss."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]


def _tiny_inputs(tmp_path):
    # two parallel 12-node chains (flow != geography), 18 months of optical series
    nodes = {}
    for tag, lon in (("a", 0.0), ("b", 0.015)):
        for k in range(12):
            nodes[f"{tag}{k}"] = {
                "up": [f"{tag}{k-1}"] if k > 0 else [],
                "next_down": f"{tag}{k+1}" if k < 11 else None,
                "strahler": 1, "length_m": 2000.0,
                "lat": k * 0.02, "lon": lon, "river_id": tag, "river_name": tag,
            }
    graph = tmp_path / "g.json"
    graph.write_text(json.dumps(nodes), encoding="utf-8")
    rng = np.random.default_rng(0)
    lines = ["object_id,ym,turb,ndci"]
    for s in nodes:
        for mi in range(18):
            ym = f"20{20 + mi // 12:02d}-{mi % 12 + 1:02d}"
            lines.append(f"{s},{ym},{abs(rng.standard_normal()) * 0.1:.4f},"
                         f"{abs(rng.standard_normal()) * 0.1:.4f}")
    series = tmp_path / "series.csv"
    series.write_text("\n".join(lines), encoding="utf-8")
    return str(graph), str(series)


def _run(script, *args):
    r = subprocess.run([sys.executable, str(PKG / "scripts" / script), *args],
                       cwd=str(PKG), capture_output=True, text=True)
    assert r.returncode == 0, f"{script} exited {r.returncode}:\n{r.stdout}\n{r.stderr}"


def test_exp_anomaly_cli_smoke(tmp_path):
    graph, series = _tiny_inputs(tmp_path)
    _run("exp_anomaly.py", "--graph", graph, "--series", series, "--exp", "recovery")


def test_exp_basins_cli_smoke(tmp_path):
    graph, series = _tiny_inputs(tmp_path)
    _run("exp_basins.py", "--graph", graph, "--series", series)


def test_new_clis_parse_help():
    for script in ("exp_provenance.py", "build_optical_series.py"):
        _run(script, "--help")


def test_open_text_reads_split_gzip_parts(tmp_path):
    # The German per-pass archive ships as gzip members split across .part files.
    import gzip
    sys.path.insert(0, str(PKG / "scripts"))
    import _bootstrap
    (tmp_path / "x.csv.gz.part00").write_bytes(gzip.compress(b"a,b\n1,2\n"))
    (tmp_path / "x.csv.gz.part01").write_bytes(gzip.compress(b"3,4\n"))
    with _bootstrap.open_text(str(tmp_path / "x.csv")) as fh:
        assert fh.read() == "a,b\n1,2\n3,4\n"


# Every CLI must answer --help without running anything. A script that starts an
# experiment instead (loads data, queries a service) blows the timeout and fails.
_HELP_TIMEOUT_S = 30
_ALL_SCRIPTS = sorted(
    p for p in list((PKG / "scripts").glob("*.py")) + list((PKG / "scripts" / "ingest").glob("*.py"))
    if not p.name.startswith("_"))


@pytest.mark.parametrize("script", _ALL_SCRIPTS, ids=lambda p: p.relative_to(PKG / "scripts").as_posix())
def test_every_script_help_exits_quickly(script, tmp_path):
    r = subprocess.run([sys.executable, str(script), "--help"], cwd=str(tmp_path),
                       capture_output=True, text=True, timeout=_HELP_TIMEOUT_S)
    assert r.returncode == 0, f"{script.name} --help exited {r.returncode}:\n{r.stderr}"
    assert r.stdout.startswith("usage:"), r.stdout[:200]
    assert list(tmp_path.iterdir()) == [], f"{script.name} --help wrote files"
