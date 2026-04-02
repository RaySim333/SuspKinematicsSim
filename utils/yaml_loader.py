from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a Python dictionary."""
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML content at {path} must be a mapping")
    return data


def to_point_dict(raw_points: dict[str, list[float]]) -> dict[str, np.ndarray]:
    """Convert a mapping of point names to [x, y, z] arrays."""
    points: dict[str, np.ndarray] = {}
    for name, values in raw_points.items():
        arr = np.asarray(values, dtype=float)
        if arr.shape != (3,):
            raise ValueError(f"Hardpoint '{name}' must have exactly 3 numeric values")
        points[name] = arr
    return points
