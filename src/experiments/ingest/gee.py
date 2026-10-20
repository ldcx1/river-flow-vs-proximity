"""Google Earth Engine access: authentication from the environment and the
Sentinel-2 index image used by the ingest pipeline.

``earthengine-api`` is optional, so ``ee`` is imported inside each function and
this module imports without it. Credentials come from the environment:

  GEE_SERVICE_ACCOUNT_KEY   path to a service-account JSON key (recommended)
  GEE_PROJECT               the Earth Engine / Google Cloud project id

Neither needs billing.
"""
from __future__ import annotations

import json
import os

from .indices import METRIC_FIELDS  # noqa: F401  (re-export for callers)

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
_initialized = False


def init_ee() -> str:
    """Initialise Earth Engine once from environment credentials.

    Returns a human-readable identity string. Raises if neither a service-account
    key (``GEE_SERVICE_ACCOUNT_KEY``) nor persisted ``earthengine authenticate``
    credentials are available.
    """
    global _initialized
    import ee
    if _initialized:
        return "earth-engine (already initialised)"
    project = os.environ.get("GEE_PROJECT")
    key = os.environ.get("GEE_SERVICE_ACCOUNT_KEY")
    if key and os.path.exists(key):
        with open(key) as fh:
            email = json.load(fh).get("client_email", "service-account")
        ee.Initialize(ee.ServiceAccountCredentials(email, key), project=project)
        _initialized = True
        return f"{email} (service account, project={project})"
    # Fallback: credentials from a prior `earthengine authenticate`.
    ee.Initialize(project=project)
    _initialized = True
    return f"persisted user credentials (project={project})"


def _mask_clouds(img):
    """Drop SCL classes 3 (shadow), 8/9 (cloud med/high), 10 (cirrus)."""
    import ee  # noqa: F401
    scl = img.select("SCL")
    return img.updateMask(scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)))


def indices_image(start: str, end: str, geom):
    """Median cloud-masked Sentinel-2 index image over ``[start, end)`` clipped to
    ``geom`` (an ``ee.Geometry``). Bands: the seven :data:`METRIC_FIELDS`."""
    import ee
    col = (ee.ImageCollection(S2_COLLECTION)
           .filterBounds(geom).filterDate(start, end).map(_mask_clouds))
    img = col.median().clip(geom)
    ndwi = img.normalizedDifference(["B3", "B8"]).rename("NDWI")
    mndwi = img.normalizedDifference(["B3", "B11"]).rename("MNDWI")
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndti = img.normalizedDifference(["B4", "B3"]).rename("NDTI")
    ndci = img.normalizedDifference(["B5", "B4"]).rename("NDCI")
    bsi = img.expression(
        "((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))",
        {"SWIR": img.select("B11"), "RED": img.select("B4"),
         "NIR": img.select("B8"), "BLUE": img.select("B2")}).rename("BSI")
    turbidity = img.select("B4").rename("TURBIDITY")
    return ee.Image.cat([ndwi, mndwi, ndvi, ndti, ndci, bsi, turbidity]).select(METRIC_FIELDS)


def reduce_means(index_img, geom, scale: int = 20) -> dict:
    """Mean of each index band over ``geom`` (synchronous ``getInfo``)."""
    import ee
    d = index_img.reduceRegion(reducer=ee.Reducer.mean(), geometry=geom,
                               scale=scale, maxPixels=int(1e9)).getInfo()
    return {k: (d.get(k) if d else None) for k in METRIC_FIELDS}
