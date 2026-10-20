"""River-segment water-quality modelling from Sentinel-2 optical indicators,
comparing river-graph topology with a connectivity-matched proximity baseline.

Companion code for the IEEE ICCP 2026 paper. Public surface:

    RiverGraph               river-segment DAG (loading, walks, topological order)
    DischargeModel, ...      per-segment flow proxies
    compute_field,           flow-weighted upstream-influence field, source
    trace_sources,           tracing and downstream impact
    downstream_impact
    FacilityIndex            nearby industrial and wastewater facilities
    metrics, cv              shared metrics and cross-validation splits
    load_panel, Panel        feature-panel loading
"""

from . import cv, metrics
from .discharge import DischargeModel, SurfaceAreaDischarge, default_discharge
from .graph import RiverGraph
from .io.panels import Panel, load_panel
from .osm import FacilityIndex, haversine_m
from .propagation import compute_field, downstream_impact, trace_sources

__version__ = "0.1.0"

__all__ = [
    "RiverGraph",
    "DischargeModel", "SurfaceAreaDischarge", "default_discharge",
    "compute_field", "trace_sources", "downstream_impact",
    "FacilityIndex", "haversine_m",
    "metrics", "cv",
    "Panel", "load_panel",
    "__version__",
]
