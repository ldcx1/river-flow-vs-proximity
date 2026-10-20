"""OpenStreetMap industrial / wastewater facilities via the Overpass API.

Produces ``data/de/osm_industrial_de.json``: a list of
``{id, kind, name, lat, lon, tags}``. :func:`fetch_facilities` does the live
download. The bbox is split into tiles and each selector is queried per tile to
keep Overpass requests small.
"""
from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

OVERPASS_URLS = ["https://overpass-api.de/api/interpreter",
                 "https://overpass.kumi.systems/api/interpreter"]

# (Overpass tag selector, facility kind). Order sets the priority in kind_for.
SELECTORS: List[Tuple[str, str]] = [
    ('["man_made"="wastewater_plant"]', "wastewater"),
    ('["man_made"="works"]', "works"),
    ('["landuse"="industrial"]', "industrial"),
    ('["power"="plant"]', "power_plant"),
    ('["landuse"="quarry"]', "quarry"),
    ('["man_made"="mineshaft"]', "mine"),
    ('["landuse"="landfill"]', "landfill"),
    ('["man_made"="storage_tank"]', "storage"),
]
KEEP_TAGS = {"name", "name:en", "operator", "industrial", "landuse",
             "man_made", "power", "plant:source", "product", "substance"}

# Germany test-basin bounding box (W, S, E, N).
DE_BBOX = (8.19, 47.28, 15.03, 53.88)


def tiles(bbox: Tuple[float, float, float, float], cols: int = 5, rows: int = 5
          ) -> Iterator[Tuple[float, float, float, float]]:
    """Split a (W,S,E,N) bbox into cols*rows (S,W,N,E) tiles for Overpass."""
    w, s, e, n = bbox
    dw, dn = (e - w) / cols, (n - s) / rows
    for c in range(cols):
        for r in range(rows):
            yield (s + r * dn, w + c * dw, s + (r + 1) * dn, w + (c + 1) * dw)


def build_query(selector: str, tile: Tuple[float, float, float, float]) -> str:
    """Overpass QL for one (S,W,N,E) tile and one tag selector."""
    s, w, n, e = tile
    return (f"[out:json][timeout:90];\n(\n  way{selector}({s},{w},{n},{e});\n"
            f"  node{selector}({s},{w},{n},{e});\n);\nout center tags;\n")


def kind_for(tags: Dict[str, str]) -> str:
    """Map an element's tags to a facility kind (first matching selector wins)."""
    for sel, kind in SELECTORS:
        k, v = sel.strip("[]").split("=")
        if tags.get(k.strip('"')) == v.strip('"'):
            return kind
    return "industrial"


def parse_element(el: dict, kind_hint: Optional[str] = None) -> Optional[dict]:
    """Turn one Overpass element into a facility record, or None if unlocated."""
    tags = el.get("tags", {})
    if "center" in el:
        lat, lon = el["center"]["lat"], el["center"]["lon"]
    elif "lat" in el:
        lat, lon = el["lat"], el["lon"]
    else:
        return None
    return {"id": f"{el['type']}/{el['id']}", "kind": kind_hint or kind_for(tags),
            "name": tags.get("name") or tags.get("operator") or "",
            "lat": round(lat, 5), "lon": round(lon, 5),
            "tags": {k: v for k, v in tags.items() if k in KEEP_TAGS}}


def fetch_facilities(bbox: Tuple[float, float, float, float] = DE_BBOX,
                     cols: int = 5, rows: int = 5, pause: float = 1.0) -> List[dict]:
    """Live tiled Overpass download -> deduplicated facility list."""
    import time

    import requests
    headers = {"User-Agent": "river-wq-research/1.0 (water-quality research)"}

    def post(query: str) -> Optional[dict]:
        last = None
        for url in OVERPASS_URLS:
            try:
                r = requests.post(url, data={"data": query}, headers=headers, timeout=120)
                r.raise_for_status()
                return r.json()
            except Exception as exc:  # noqa: BLE001  (mirror + retry)
                last = exc
                time.sleep(3)
        print(f"[osm] all mirrors failed: {last}")
        return None

    by_id: Dict[str, dict] = {}
    tile_list = list(tiles(bbox, cols, rows))
    for selector, kind in SELECTORS:
        for tile in tile_list:
            data = post(build_query(selector, tile))
            for el in (data or {}).get("elements", []):
                rec = parse_element(el, kind_hint=kind_for(el.get("tags", {})))
                if rec and rec["id"] not in by_id:
                    by_id[rec["id"]] = rec
            time.sleep(pause)
        print(f"[osm] {kind}: total {len(by_id)}")
    return list(by_id.values())
