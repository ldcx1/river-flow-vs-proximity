"""Sentinel-2 index and risk-score math (mirrors the Earth Engine reducers)."""
from experiments.ingest import indices


def test_normalized_difference():
    assert indices.normalized_difference(3, 1) == 0.5
    assert indices.normalized_difference(0, 0) == 0.0
    assert indices.normalized_difference(1, 3) == -0.5


def test_compute_indices_turbidity_is_red_and_ndci_formula():
    idx = indices.compute_indices(
        {"B2": 0.1, "B3": 0.2, "B4": 0.3, "B5": 0.4, "B8": 0.5, "B11": 0.6})
    assert idx["TURBIDITY"] == 0.3                       # raw red band B4
    assert idx["NDCI"] == indices.normalized_difference(0.4, 0.3)   # (B5-B4)/(B5+B4)
    assert idx["NDWI"] == indices.normalized_difference(0.2, 0.5)   # (B3-B8)/(B3+B8)


def test_risk_clean_water_is_low():
    r = indices.risk_score({"MNDWI": 0.5, "NDTI": 0.0, "TURBIDITY": 0.0,
                            "NDVI": 0.5, "NDCI": 0.0, "BSI": 0.0})
    assert r["is_water"] == 1 and r["risk_score"] == 0 and r["risk_level"] == "LOW"


def test_risk_high_turbid_water_is_high():
    # water: 2(ndti) +1(turb) +1(ndci) +1(ndti&ndci) = 5, land: 1(ndvi<.3)+1(bsi>.3)=2 -> 7
    r = indices.risk_score({"MNDWI": 0.5, "NDTI": 0.2, "TURBIDITY": 0.2,
                            "NDVI": 0.2, "NDCI": 0.2, "BSI": 0.4})
    assert r["risk_score"] == 7 and r["risk_level"] == "HIGH"


def test_risk_non_water_zeroes_water_component():
    r = indices.risk_score({"MNDWI": 0.0, "NDTI": 0.5, "TURBIDITY": 0.5,
                            "NDVI": 0.5, "NDCI": 0.5, "BSI": 0.0})
    assert r["is_water"] == 0 and r["water_risk"] == 0 and r["risk_score"] == 0


def test_risk_missing_keys_default_to_zero():
    # all missing -> 0: not water, land = 1(ndvi 0<0.3) + 0(bsi 0 not>0.3) = 1
    assert indices.risk_score({})["risk_score"] == 1
