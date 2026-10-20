"""Per-segment discharge proxy.

There is no measured per-segment flow, so the default model estimates it from
channel width, which is derived from the EU-Hydro Strahler order. Wider reaches
carry and dilute more water.

``DischargeModel.flow(node)`` returns an m3/s-like proxy. Only ratios of flow
between confluent reaches are used by the mixing model, so the absolute scale
does not matter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

# Representative bankfull width (m) per Strahler order. Order-of-magnitude
# hydraulic-geometry values: each step up roughly doubles width, from a 2 m
# headwater brook to the ~800 m Danube (Strahler 9).
_STRAHLER_WIDTH_M = {
    1: 2.0, 2: 4.0, 3: 8.0, 4: 16.0, 5: 32.0,
    6: 70.0, 7: 150.0, 8: 350.0, 9: 800.0,
}
_DEFAULT_WIDTH_M = 5.0

# Q ~ width * depth * velocity, and depth and velocity grow slowly with width,
# so a single power law on width gives a relative proxy: Q = k * width^EXP.
_WIDTH_EXP = 1.6
_K = 0.05


class DischargeModel(ABC):
    """``flow(node)`` returns an m3/s-like flow proxy for a graph node dict
    (must carry at least ``strahler``)."""

    name = "abstract"

    @abstractmethod
    def flow(self, node: dict) -> float:  # pragma: no cover - interface
        ...


class SurfaceAreaDischarge(DischargeModel):
    """Width from Strahler order, flow from a power law on width. The default."""

    name = "surface"

    def width_m(self, node: dict) -> float:
        return _STRAHLER_WIDTH_M.get(int(node.get("strahler", 0) or 0), _DEFAULT_WIDTH_M)

    def flow(self, node: dict) -> float:
        w = self.width_m(node)
        return round(_K * (w ** _WIDTH_EXP), 4)



def default_discharge() -> DischargeModel:
    """The default surface-area discharge model (the one the paper uses)."""
    return SurfaceAreaDischarge()
