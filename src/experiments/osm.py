"""Spatial index of facilities (industrial sites, wastewater plants, power
stations, quarries, mines, landfills) used to find pressure points near ranked
source reaches.

A grid of 0.05 degree cells (about 5 km) gives constant-time proximity lookups
without a spatial database.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

_CELL = 0.05  # degrees


def haversine_m(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    """Great-circle distance in metres between two lat/lon points."""
    R = 6371000.0
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp = math.radians(b_lat - a_lat)
    dl = math.radians(b_lon - a_lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(h)))


def haversine_km_vec(lat0: float, lon0: float, lats, lons):
    """Great-circle distance (km) from one point to arrays of lat/lon.

    Used for nearest-neighbour selection, so geographic baselines use true
    distances rather than planar lat/lon. numpy is imported lazily."""
    import numpy as np
    R = 6371.0
    p1 = math.radians(lat0)
    p2 = np.radians(lats)
    dp = np.radians(np.asarray(lats) - lat0)
    dl = np.radians(np.asarray(lons) - lon0)
    h = np.sin(dp / 2) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.minimum(1.0, np.sqrt(h)))


class FacilityIndex:
    """Grid-indexed facility lookup. Construct from a list of facility dicts
    (each with at least ``lat``/``lon``, optionally ``kind``) or from a JSON file
    via :meth:`from_json`. An empty index returns ``[]`` for every lookup.
    """

    def __init__(self, facilities: Optional[List[dict]] = None, cell: float = _CELL):
        self.cell = cell
        self.facilities: List[dict] = facilities or []
        self._index: Dict[Tuple[int, int], list] = defaultdict(list)
        for fac in self.facilities:
            self._index[self._cell(fac["lat"], fac["lon"])].append(fac)

    @classmethod
    def from_json(cls, path: str, cell: float = _CELL) -> "FacilityIndex":
        try:
            with open(path, encoding="utf-8") as f:
                facs = json.load(f)
        except FileNotFoundError:
            facs = []
        return cls(facs, cell=cell)

    def _cell(self, lat: float, lon: float) -> Tuple[int, int]:
        return (int(math.floor(lat / self.cell)), int(math.floor(lon / self.cell)))

    def available(self) -> bool:
        return bool(self.facilities)

    def nearby(self, lat: float, lon: float, radius_m: float = 2000.0,
               limit: int = 5) -> List[dict]:
        """Facilities within ``radius_m`` of (lat, lon), nearest first. Each
        result carries the original facility fields plus ``distance_m``.

        Searches the home cell and its 8 neighbours. ``radius_m`` is well under
        one cell (about 5 km), so the 3x3 window covers it.
        """
        if not self._index:
            return []
        clat, clon = self._cell(lat, lon)
        hits: List[dict] = []
        for dlat in (-1, 0, 1):
            for dlon in (-1, 0, 1):
                for fac in self._index.get((clat + dlat, clon + dlon), ()):
                    d = haversine_m(lat, lon, fac["lat"], fac["lon"])
                    if d <= radius_m:
                        hits.append({**fac, "distance_m": round(d)})
        hits.sort(key=lambda f: f["distance_m"])
        return hits[:limit]

    def annotate_sources(self, sources: List[dict], radius_m: float = 2000.0,
                         limit: int = 5) -> List[dict]:
        """Attach a ``facilities`` list to each ranked source reach (in place)
        and return it. Lists are empty if the index is empty."""
        for s in sources:
            if s.get("lat") is None or s.get("lon") is None:
                s["facilities"] = []
                continue
            s["facilities"] = self.nearby(s["lat"], s["lon"], radius_m=radius_m, limit=limit)
        return sources
