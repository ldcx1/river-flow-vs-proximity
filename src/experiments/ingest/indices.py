"""Pure-Python Sentinel-2 spectral index formulas.

They mirror the Earth Engine code in :mod:`experiments.ingest.gee`, so they can
be tested without an Earth Engine session and used to check shipped values.

Bands are Sentinel-2 surface reflectance: B2 blue, B3 green, B4 red, B5 red-edge,
B8 NIR, B11 SWIR. Indices are normalized differences ``(a-b)/(a+b)`` except
TURBIDITY, which is the raw red band (B4).
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

METRIC_FIELDS = ["NDWI", "MNDWI", "NDVI", "NDTI", "NDCI", "BSI", "TURBIDITY"]


def normalized_difference(a: float, b: float) -> float:
    """Earth Engine ``normalizedDifference`` semantics: (a-b)/(a+b), 0 if a+b==0."""
    s = a + b
    return (a - b) / s if s else 0.0


def compute_indices(bands: Mapping[str, float]) -> Dict[str, float]:
    """Per-pixel/segment indices from a band dict (B2,B3,B4,B5,B8,B11)."""
    b = {k: float(v) for k, v in bands.items()}
    swir, red, nir, blue = b["B11"], b["B4"], b["B8"], b["B2"]
    denom = (swir + red) + (nir + blue)
    return {
        "NDWI": normalized_difference(b["B3"], b["B8"]),
        "MNDWI": normalized_difference(b["B3"], b["B11"]),
        "NDVI": normalized_difference(b["B8"], b["B4"]),
        "NDTI": normalized_difference(b["B4"], b["B3"]),
        "NDCI": normalized_difference(b["B5"], b["B4"]),
        "BSI": (((swir + red) - (nir + blue)) / denom) if denom else 0.0,
        "TURBIDITY": red,
    }


def _num(means: Mapping[str, Optional[float]], key: str) -> float:
    v = means.get(key)
    return float(v) if v is not None else 0.0


def risk_score(means: Mapping[str, Optional[float]]) -> Dict[str, object]:
    """The shipped ``risk_score`` column, computed from per-segment index means.

    Water risk (only where MNDWI>0.2 marks water): 2 if NDTI>0.1, +1 if
    TURBIDITY>0.15, +1 if NDCI>0.1, +1 if NDTI>0.1 and NDCI>0.1. Land risk:
    +1 if NDVI<0.3, +1 if BSI>0.3. Level: HIGH>=5, MEDIUM>=3, else LOW.
    """
    mndwi = _num(means, "MNDWI"); ndti = _num(means, "NDTI")
    turb = _num(means, "TURBIDITY"); ndvi = _num(means, "NDVI")
    ndci = _num(means, "NDCI"); bsi = _num(means, "BSI")
    is_water = 1 if mndwi > 0.2 else 0
    water_risk = (
        (2 if ndti > 0.1 else 0)
        + (1 if turb > 0.15 else 0)
        + (1 if ndci > 0.1 else 0)
        + (1 if (ndti > 0.1 and ndci > 0.1) else 0)
    ) * is_water
    land_risk = (1 if ndvi < 0.3 else 0) + (1 if bsi > 0.3 else 0)
    score = water_risk + land_risk
    level = "HIGH" if score >= 5 else ("MEDIUM" if score >= 3 else "LOW")
    return {"risk_score": score, "water_risk": water_risk, "land_risk": land_risk,
            "is_water": is_water, "risk_level": level}
