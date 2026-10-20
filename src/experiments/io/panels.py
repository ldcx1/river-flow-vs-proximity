"""Load a (segment x month) feature-panel CSV into model-ready arrays.

A panel row is one labelled (segment, month) cell. Feature columns are named with
a one-letter group prefix so experiments can ablate whole groups:

    o_*  optical    Sentinel-2 indices
    p_*  physics    flow-weighted upstream-influence prior (compute_field)
    t_*  topology   graph context (strahler, length, in-degree, confluence)
    f_*  pressure   facility and agricultural pressure (incl. f_ag_*)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

GROUP_PREFIXES: Dict[str, str] = {
    "optical": "o_", "physics": "p_", "topo": "t_", "pressure": "f_",
}
FEATURE_PREFIXES = tuple(GROUP_PREFIXES.values())


@dataclass
class Panel:
    """A loaded feature panel. ``X`` may contain NaN for missing features. XGBoost
    handles NaN natively and the logistic model mean-imputes per fold."""

    X: np.ndarray            # (n_rows, n_features) float, NaN = missing
    y: np.ndarray            # (n_rows,) float 0/1
    groups: np.ndarray       # (n_rows,) group label per row (segment object_id)
    feature_cols: List[str]  # column name per X column
    rows: List[dict]         # raw CSV rows (for extra fields, e.g. weak_label)

    def __len__(self) -> int:
        return len(self.y)

    def group_of(self, col: str) -> Optional[str]:
        for group, prefix in GROUP_PREFIXES.items():
            if col.startswith(prefix):
                return group
        return None

    def select(self, keep: Optional[Iterable[str]] = None,
               drop: Optional[Iterable[str]] = None) -> List[int]:
        """Column indices for the requested feature *groups*. ``keep`` whitelists
        groups, ``drop`` blacklists them, and both may be combined."""
        keep_set = set(keep) if keep is not None else None
        drop_set = set(drop) if drop is not None else set()
        idx = []
        for j, c in enumerate(self.feature_cols):
            grp = self.group_of(c)
            if keep_set is not None and grp not in keep_set:
                continue
            if grp in drop_set:
                continue
            idx.append(j)
        return idx

    def select_prefix(self, prefixes: Sequence[str]) -> List[int]:
        """Column indices for columns starting with any of ``prefixes``, for
        subsets finer than whole groups (e.g. ``o_`` plus ``f_ag_``)."""
        return [j for j, c in enumerate(self.feature_cols)
                if any(c.startswith(p) for p in prefixes)]

    def present_groups(self) -> set:
        return {g for g, p in GROUP_PREFIXES.items()
                if any(c.startswith(p) for c in self.feature_cols)}


def load_panel(path: str, label_col: str = "label",
               group_col: str = "object_id",
               prefixes: Sequence[str] = FEATURE_PREFIXES) -> Panel:
    """Read a feature-panel CSV. Feature columns are those whose name starts with
    one of ``prefixes``. ``label_col`` is the binary target and ``group_col`` is
    the CV grouping key."""
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"empty panel: {path}")
    feat_cols = [c for c in rows[0] if any(c.startswith(p) for p in prefixes)]
    X = np.full((len(rows), len(feat_cols)), np.nan)
    y = np.array([int(r[label_col]) for r in rows], dtype=float)
    groups = np.array([r[group_col] for r in rows])
    for i, r in enumerate(rows):
        for j, c in enumerate(feat_cols):
            v = r[c]
            if v not in ("", None):
                try:
                    X[i, j] = float(v)
                except ValueError:
                    pass
    return Panel(X=X, y=y, groups=groups, feature_cols=feat_cols, rows=rows)
