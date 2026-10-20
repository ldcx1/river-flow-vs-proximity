"""Strong-label exceedance logic: directions, unit-mismatch guarding, month
grouping from compact dates, and any-determinand-positive aggregation."""

from experiments.data import DEFAULT_RULES, build_strong_labels, is_exceedance


def test_is_exceedance_directions():
    do = DEFAULT_RULES["Dissolved oxygen"]      # lt 6.0
    assert is_exceedance(do, 5.9) is True
    assert is_exceedance(do, 6.1) is False
    bod = DEFAULT_RULES["BOD5"]                  # gt 6.0
    assert is_exceedance(bod, 6.1) is True
    assert is_exceedance(bod, 5.9) is False


def _r(seg, det, date, value, uom):
    return {"segment": seg, "determinand": det, "date": date, "value": value, "uom": uom}


def test_any_determinand_makes_cell_positive():
    readings = [
        _r("s1", "Nitrate", "20210315", "30", "mg{NO3}/L"),       # 30 > 25 -> exceed
        _r("s1", "Dissolved oxygen", "20210320", "8.0", "mg/L"),  # 8 > 6 -> ok
        _r("s1", "Nitrate", "20210610", "10", "mg{NO3}/L"),       # ok month
    ]
    rows, stats = build_strong_labels(readings, {"s1": "B"})
    by_month = {r["year_month"]: r for r in rows}
    assert by_month["2021-03"]["label"] == 1
    assert by_month["2021-03"]["drivers"] == "Nitrate"
    assert by_month["2021-06"]["label"] == 0
    assert stats.n_pos == 1


def test_unit_mismatch_is_skipped_not_compared():
    # Nitrate as N (mg{N}/L) at 30 would falsely exceed the mg{NO3}/L rule if
    # compared blindly. It must be skipped and tallied instead.
    readings = [_r("s1", "Nitrate", "20210315", "30", "mg{N}/L")]
    rows, stats = build_strong_labels(readings, {"s1": "B"})
    assert rows == []                       # no valid reading -> no cell
    assert stats.uom_mismatch[("Nitrate", "mg{N}/L")] == 1


def test_year_filter():
    readings = [
        _r("s1", "Nitrate", "20190315", "30", "mg{NO3}/L"),   # before window
        _r("s1", "Nitrate", "20240315", "30", "mg{NO3}/L"),   # after window
        _r("s1", "Nitrate", "20210315", "30", "mg{NO3}/L"),   # in window
    ]
    rows, _ = build_strong_labels(readings, {"s1": "B"}, years=("2020", "2023"))
    assert [r["year_month"] for r in rows] == ["2021-03"]


def test_unknown_determinand_ignored():
    readings = [_r("s1", "Temperature", "20210315", "99", "C")]
    rows, stats = build_strong_labels(readings, {"s1": "B"})
    assert rows == []
    assert stats.det_seen.get("Temperature", 0) == 0


def test_month_parsing_from_compact_date():
    readings = [_r("s1", "Nitrate", "20210820", "30", "mg{NO3}/L")]
    rows, _ = build_strong_labels(readings, {"s1": "B"})
    assert rows[0]["year_month"] == "2021-08"


def test_basin_lookup_falls_back():
    readings = [_r("sX", "Nitrate", "20210315", "30", "mg{NO3}/L")]
    rows, _ = build_strong_labels(readings, {})    # sX not in site_basin
    assert rows[0]["basin"] == "?"
