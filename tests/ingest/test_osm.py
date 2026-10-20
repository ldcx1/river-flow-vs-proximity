"""OSM/Overpass query builder + element parser."""
from experiments.ingest import osm


def test_tiles_count_and_coverage():
    ts = list(osm.tiles((0, 0, 10, 10), cols=5, rows=5))
    assert len(ts) == 25
    assert min(t[1] for t in ts) == 0 and max(t[3] for t in ts) == 10   # W..N span


def test_build_query_has_selector_bbox_and_output():
    q = osm.build_query('["man_made"="works"]', (1, 2, 3, 4))
    assert 'man_made' in q and '(1,2,3,4)' in q and 'out center tags' in q


def test_kind_for_priority_and_fallback():
    assert osm.kind_for({"man_made": "wastewater_plant"}) == "wastewater"
    assert osm.kind_for({"landuse": "industrial"}) == "industrial"
    assert osm.kind_for({"amenity": "cafe"}) == "industrial"   # fallback


def test_parse_element_node_way_and_missing():
    node = osm.parse_element({"type": "node", "id": 1, "lat": 50.0, "lon": 8.0,
                              "tags": {"name": "X", "man_made": "works", "foo": "bar"}})
    assert node["id"] == "node/1" and node["kind"] == "works" and node["name"] == "X"
    assert "foo" not in node["tags"]                     # KEEP_TAGS filter
    way = osm.parse_element({"type": "way", "id": 2,
                             "center": {"lat": 50.0, "lon": 8.0}, "tags": {}})
    assert way["id"] == "way/2" and way["lat"] == 50.0
    assert osm.parse_element({"type": "way", "id": 3, "tags": {}}) is None   # unlocated
