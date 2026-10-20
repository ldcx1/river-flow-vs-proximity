"""EEA Waterbase (WISE-6) in-situ chemistry: surface-water filtering and snapping.

  * :func:`accept_row` decides whether one raw disaggregated row is kept. It
    keeps river, lake, transitional and coastal water (groundwater is about 38%
    of German rows and is dropped), water matrices only (no sediment or biota),
    and optically relevant determinands.
  * :func:`haversine` and :class:`GridIndex` snap a monitoring site to the
    nearest river reach.

Used by ``scripts/ingest/extract_insitu.py`` and ``snap_sites.py``.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

# --- surface-water filtering -------------------------------------------------

# Determinands kept (optically relevant or used by the graph experiments).
OPTICAL_DETERMINANDS = {
    "Total suspended solids", "Nitrate", "Nitrite", "Ammonium",
    "Dissolved oxygen", "BOD5", "Chlorophyll a", "Water transparency",
}
# Surface-water body categories (groundwater "GW" and unknown are dropped).
SURFACE_CATEGORIES = {"RW", "LW", "TW", "CW"}

# Fixed column indices in Waterbase_v2024_1_T_WISE6_DisaggregatedData.csv:
# countryCode, monitoringSiteIdentifier, ...Scheme, parameterWaterBodyCategory,
# ...Code, ...Label(determinand), procedureAnalysedMatrix, resultUom,
# phenomenonTimeSamplingDate, sampleIdentifier, resultObservedValue, ...
C_SITE, C_CAT, C_DET, C_MATRIX, C_UOM, C_DATE, C_VAL = 1, 3, 5, 6, 7, 8, 10


def accept_row(row: Sequence[str], site_seg: Dict[str, Tuple[str, str]]
               ) -> Optional[list]:
    """Return ``[site, segment, basin, determinand, date, value, uom]`` for a kept
    disaggregated row, else ``None``.

    ``site_seg`` maps a monitoring-site id to ``(segment, basin)`` from snapping.
    A row is kept only if its site snapped, its determinand is optically relevant,
    its matrix is water, and its water-body category is surface water.
    """
    if len(row) <= C_VAL:
        return None
    seg = site_seg.get(row[C_SITE])
    if seg is None:
        return None
    if row[C_DET] not in OPTICAL_DETERMINANDS:
        return None
    if not row[C_MATRIX].startswith("W"):        # water matrix only
        return None
    if row[C_CAT] not in SURFACE_CATEGORIES:     # surface water only (drop GW)
        return None
    date, val = row[C_DATE][:10], row[C_VAL]
    if not date or not val:
        return None
    return [row[C_SITE], seg[0], seg[1], row[C_DET], date, val, row[C_UOM]]


# --- snapping ----------------------------------------------------------------

EARTH_R_M = 6_371_000.0


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R_M * math.asin(min(1.0, math.sqrt(h)))


class GridIndex:
    """Bucket graph nodes into a lat/lon grid for fast nearest-node lookup.

    Cells are about 4-5 km at German latitudes, larger than the snap radius, so
    the 3x3 window around a query contains the nearest node within radius.
    """

    def __init__(self, graph: Dict[str, dict], cell: float = 0.05):
        self.g, self.cell = graph, cell
        self.idx: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        for oid, n in graph.items():
            self.idx[(int(n["lat"] // cell), int(n["lon"] // cell))].append(oid)

    def nearest(self, lat: float, lon: float) -> Tuple[Optional[str], float]:
        """Nearest node id and its distance (m) to (lat, lon)."""
        c = (int(lat // self.cell), int(lon // self.cell))
        best, bd = None, float("inf")
        for a in (-1, 0, 1):
            for o in (-1, 0, 1):
                for oid in self.idx.get((c[0] + a, c[1] + o), ()):
                    nn = self.g[oid]
                    d = haversine(lat, lon, nn["lat"], nn["lon"])
                    if d < bd:
                        best, bd = oid, d
        return best, bd
