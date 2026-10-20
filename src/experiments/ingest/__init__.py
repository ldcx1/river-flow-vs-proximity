"""Data acquisition and parsing for the reproduction bundle.

This code shows how the files under ``data/`` are produced from public raw
sources. The experiments read the committed bundle and do not need it. It is
kept so the pipeline is reproducible and its pure logic is testable.

Sources and the module that handles each:

  * Sentinel-2 optical indices (Google Earth Engine)  -> ``gee`` + ``indices``
  * EU-Hydro river network (geodatabase)              -> ``euhydro``
  * EEA Waterbase in-situ chemistry (CSV)             -> ``waterbase``
  * OpenStreetMap industrial facilities (Overpass)    -> ``osm``

Credentials come from the environment (see ``gee.init_ee``). The heavy Earth
Engine export steps are documented in ``scripts/ingest/README.md``.
"""

from . import indices  # noqa: F401  (pure, always importable)
