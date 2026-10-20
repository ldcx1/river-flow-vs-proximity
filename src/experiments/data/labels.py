"""Strong (in-situ) labels: per-(segment, month) water-quality exceedance.

WFD status labels are about 90% positive and nearly constant, so in-situ
readings give the target instead. A reading exceeds when it breaches a LAWA/WFD
river good/moderate class boundary. A (segment, month) cell is positive if any
determinand exceeds that month, matching the monthly Sentinel-2 composites.

The model features are proxies for these exceedances. They are not direct
optical detection of the determinands, which are often dissolved.

Pure core of ``scripts/build_strong_labels.py``, which handles the file IO.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple


@dataclass(frozen=True)
class ExceedanceRule:
    """``direction='lt'``: a value below ``threshold`` exceeds (low DO).
    ``'gt'``: a value above exceeds. ``uom`` is the unit the threshold assumes.
    Readings in a different unit are skipped rather than compared."""

    direction: str
    threshold: float
    uom: str


# Thresholds in the units reported by EEA Waterbase. The paper lists the LAWA/WFD
# source of each boundary.
DEFAULT_RULES: Dict[str, ExceedanceRule] = {
    "Dissolved oxygen": ExceedanceRule("lt", 6.0, "mg/L"),
    "BOD5":             ExceedanceRule("gt", 6.0, "mg{O2}/L"),
    "Ammonium":         ExceedanceRule("gt", 0.4, "mg{NH4}/L"),
    "Nitrite":          ExceedanceRule("gt", 0.1, "mg{NO2}/L"),
    "Nitrate":          ExceedanceRule("gt", 25.0, "mg{NO3}/L"),
    "Chlorophyll a":    ExceedanceRule("gt", 25.0, "ug/L"),
}


def is_exceedance(rule: ExceedanceRule, value: float) -> bool:
    """True if ``value`` breaches ``rule``'s boundary in the bad direction."""
    return value < rule.threshold if rule.direction == "lt" else value > rule.threshold


@dataclass
class LabelStats:
    n_cells: int
    n_segments: int
    n_pos: int
    per_basin: Dict[str, Counter]
    det_seen: Counter
    det_exceed: Counter
    uom_mismatch: Counter

    @property
    def positive_rate(self) -> float:
        return self.n_pos / self.n_cells if self.n_cells else 0.0


def _year_month(date: str) -> Tuple[str, str]:
    """Split a compact ``YYYYMMDD`` date into ``(year, 'YYYY-MM')``.

    Waterbase sampling dates are compact (e.g. ``20210315``). Dashed ISO dates
    must be normalised first (the in-situ extractor does this).
    """
    return date[:4], f"{date[:4]}-{date[4:6]}"


def build_strong_labels(
    readings: Iterable[Mapping[str, str]],
    site_basin: Mapping[str, str],
    rules: Mapping[str, ExceedanceRule] = DEFAULT_RULES,
    years: Tuple[str, str] = ("2020", "2023"),
) -> Tuple[List[dict], LabelStats]:
    """Aggregate dated in-situ ``readings`` into (segment, month) strong labels.

    Each reading is a mapping with ``segment``, ``determinand``, ``date``
    (compact ``YYYYMMDD``), ``value``, ``uom``. ``site_basin`` maps segment ->
    basin name. Returns ``(rows, stats)`` where each row is
    ``{object_id, basin, year_month, n_readings, n_exceed, drivers, label}``.

    A reading counts only if its determinand has a rule and its unit matches the
    rule's ``uom``. Unit mismatches are tallied in ``stats.uom_mismatch``.
    """
    cell: Dict[Tuple[str, str], dict] = defaultdict(
        lambda: {"n": 0, "exc": 0, "drivers": set()})
    det_seen, det_exc, uom_mismatch = Counter(), Counter(), Counter()

    for r in readings:
        det = r["determinand"]
        rule = rules.get(det)
        if rule is None:
            continue
        date = r["date"]
        if not (years[0] <= date[:4] <= years[1]):
            continue
        if r["uom"] != rule.uom:
            uom_mismatch[(det, r["uom"])] += 1
            continue
        try:
            val = float(r["value"])
        except (ValueError, TypeError):
            continue
        if not math.isfinite(val):
            # "NaN"/"inf" parse as floats but are not readings. Counting them
            # would dilute the label.
            continue
        _, ym = _year_month(date)
        c = cell[(r["segment"], ym)]
        c["n"] += 1
        det_seen[det] += 1
        if is_exceedance(rule, val):
            c["exc"] += 1
            c["drivers"].add(det)
            det_exc[det] += 1

    rows: List[dict] = []
    per_basin: Dict[str, Counter] = defaultdict(Counter)
    n_pos = 0
    for (seg, ym), c in cell.items():
        label = 1 if c["exc"] > 0 else 0
        basin = site_basin.get(seg, "?")
        rows.append({
            "object_id": seg, "basin": basin, "year_month": ym,
            "n_readings": c["n"], "n_exceed": c["exc"],
            "drivers": "|".join(sorted(c["drivers"])), "label": label,
        })
        n_pos += label
        per_basin[basin][label] += 1

    rows.sort(key=lambda r: (r["object_id"], r["year_month"]))
    stats = LabelStats(
        n_cells=len(rows), n_segments=len({r["object_id"] for r in rows}),
        n_pos=n_pos, per_basin=per_basin, det_seen=det_seen, det_exceed=det_exc,
        uom_mismatch=uom_mismatch,
    )
    return rows, stats
