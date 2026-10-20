"""Waterbase surface-water row gate + snapping."""
from experiments.ingest import waterbase as wb


def _row(site="S1", cat="RW", det="Nitrate", matrix="Water",
         val="1.0", date="2021-05-06T00:00:00"):
    r = [""] * 11
    r[wb.C_SITE] = site; r[wb.C_CAT] = cat; r[wb.C_DET] = det
    r[wb.C_MATRIX] = matrix; r[wb.C_UOM] = "mg/L"; r[wb.C_DATE] = date; r[wb.C_VAL] = val
    return r


SITE_SEG = {"S1": ("RL123", "Rhine")}


def test_accept_row_keeps_surface_nitrate():
    assert wb.accept_row(_row(), SITE_SEG) == \
        ["S1", "RL123", "Rhine", "Nitrate", "2021-05-06", "1.0", "mg/L"]


def test_accept_row_drops_groundwater_category():
    assert wb.accept_row(_row(cat="GW"), SITE_SEG) is None


def test_accept_row_drops_unsnapped_site():
    assert wb.accept_row(_row(site="SX"), SITE_SEG) is None


def test_accept_row_drops_non_optical_determinand():
    assert wb.accept_row(_row(det="pH"), SITE_SEG) is None


def test_accept_row_drops_non_water_matrix():
    assert wb.accept_row(_row(matrix="Sediment"), SITE_SEG) is None


def test_haversine_zero_and_known_distance():
    assert wb.haversine(50, 8, 50, 8) == 0.0
    d = wb.haversine(50.0, 8.0, 50.0, 8.01)     # ~0.01 deg lon at 50N ≈ 0.7 km
    assert 600 < d < 800


def test_grid_index_nearest():
    g = {"A": {"lat": 50.0, "lon": 8.0}, "B": {"lat": 51.0, "lon": 9.0}}
    oid, d = wb.GridIndex(g).nearest(50.001, 8.001)
    assert oid == "A" and d < 200
