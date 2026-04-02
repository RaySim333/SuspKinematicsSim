from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from corner.core.hardpoints import SuspensionModel
    from corner.core.kinematics_calcs import compute_kinematic_channels
    from corner.core.solver import KinematicSolver
    from corner.utils.yaml_loader import load_yaml
except ModuleNotFoundError:
    from core.hardpoints import SuspensionModel
    from core.kinematics_calcs import compute_kinematic_channels
    from core.solver import KinematicSolver
    from utils.yaml_loader import load_yaml


@dataclass
class SingleCornerResult:
    travel_mm: np.ndarray
    channels: dict[str, np.ndarray]
    positions: list[dict[str, np.ndarray] | None]
    valid_mask: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        data = {"travel_mm": self.travel_mm, "valid": self.valid_mask.astype(int)}
        data.update(self.channels)
        return pd.DataFrame(data)


def load_model_and_options(model_path: str | Path) -> tuple[SuspensionModel, dict[str, Any]]:
    config = load_yaml(model_path)
    model = SuspensionModel.from_yaml(model_path)
    optional_channels = config.get("optional_channels", {})
    if not isinstance(optional_channels, dict):
        optional_channels = {}
    return model, optional_channels


def _extract_damper_length(
    points: dict[str, np.ndarray],
    damper_points: tuple[str, str] | None,
) -> float:
    if damper_points is None:
        return float("nan")

    moving, fixed = damper_points
    if moving not in points or fixed not in points:
        return float("nan")

    return float(np.linalg.norm(points[moving] - points[fixed]))


def _compute_motion_ratio(
    wheel_z_mm: np.ndarray,
    damper_length_mm: np.ndarray,
) -> np.ndarray:
    if np.isnan(damper_length_mm).all() or len(wheel_z_mm) < 3:
        return np.full_like(wheel_z_mm, np.nan, dtype=float)

    with np.errstate(invalid="ignore", divide="ignore"):
        d_damper = np.gradient(damper_length_mm)
        d_wheel = np.gradient(wheel_z_mm)
        ratio = -d_damper / d_wheel
    return ratio


def run_single_corner_sweep(
    model: SuspensionModel,
    travel_min_mm: float,
    travel_max_mm: float,
    step_mm: float,
    ride_height_travel_mm: float = 0.0,
    right_side: bool = False,
    damper_points: tuple[str, str] | None = None,
) -> SingleCornerResult:
    if step_mm <= 0.0:
        raise ValueError("step_mm must be positive")
    if travel_max_mm <= travel_min_mm:
        raise ValueError("travel_max_mm must be greater than travel_min_mm")

    solver = KinematicSolver(model=model, right_side=right_side)
    input_point_name, input_axis_idx = model.input_dof_target
    nominal_input = float(solver.initial_points[input_point_name][input_axis_idx])

    travel = np.arange(travel_min_mm, travel_max_mm + 0.5 * step_mm, step_mm, dtype=float)

    channel_names = [
        "wheel_center_x_mm",
        "wheel_center_y_mm",
        "wheel_center_z_mm",
        "camber_deg",
        "toe_deg",
        "caster_deg",
        "kingpin_inclination_deg",
        "scrub_radius_mm",
        "mechanical_trail_mm",
        "instant_center_y_mm",
        "instant_center_z_mm",
        "roll_center_z_mm",
        "track_change_mm",
        "lateral_scrub_mm",
        "wheel_longitudinal_shift_mm",
        "damper_length_mm",
        "motion_ratio",
    ]

    acc: dict[str, list[float]] = {name: [] for name in channel_names}
    positions: list[dict[str, np.ndarray] | None] = []
    valid_mask: list[bool] = []

    current_guess = solver.constraint_system.pack_variables(solver.initial_points)

    for travel_step in travel:
        target_value = nominal_input + float(travel_step)
        try:
            points = solver.solve_position(target_input_value=target_value, initial_guess=current_guess)
            current_guess = solver.constraint_system.pack_variables(points)
            base = compute_kinematic_channels(model, points)
            valid_mask.append(True)
            positions.append(points)

            for key in (
                "wheel_center_x_mm",
                "wheel_center_y_mm",
                "wheel_center_z_mm",
                "camber_deg",
                "toe_deg",
                "caster_deg",
                "kingpin_inclination_deg",
                "scrub_radius_mm",
                "mechanical_trail_mm",
                "instant_center_y_mm",
                "instant_center_z_mm",
                "roll_center_z_mm",
            ):
                acc[key].append(float(base[key]))

            acc["damper_length_mm"].append(_extract_damper_length(points, damper_points))
        except Exception:
            valid_mask.append(False)
            positions.append(None)
            for key in (
                "wheel_center_x_mm",
                "wheel_center_y_mm",
                "wheel_center_z_mm",
                "camber_deg",
                "toe_deg",
                "caster_deg",
                "kingpin_inclination_deg",
                "scrub_radius_mm",
                "mechanical_trail_mm",
                "instant_center_y_mm",
                "instant_center_z_mm",
                "roll_center_z_mm",
                "damper_length_mm",
            ):
                acc[key].append(float("nan"))

    wheel_x = np.asarray(acc["wheel_center_x_mm"], dtype=float)
    wheel_y = np.asarray(acc["wheel_center_y_mm"], dtype=float)
    wheel_z = np.asarray(acc["wheel_center_z_mm"], dtype=float)
    damper_len = np.asarray(acc["damper_length_mm"], dtype=float)

    ride_idx = int(np.argmin(np.abs(travel - float(ride_height_travel_mm))))
    x_ref = wheel_x[ride_idx]
    y_ref = wheel_y[ride_idx]

    acc["wheel_longitudinal_shift_mm"] = list(wheel_x - x_ref)
    acc["track_change_mm"] = list(wheel_y - y_ref)
    acc["lateral_scrub_mm"] = list(wheel_y - y_ref)
    acc["motion_ratio"] = list(_compute_motion_ratio(wheel_z, damper_len))

    channels = {k: np.asarray(v, dtype=float) for k, v in acc.items()}

    return SingleCornerResult(
        travel_mm=travel,
        channels=channels,
        positions=positions,
        valid_mask=np.asarray(valid_mask, dtype=bool),
    )


def resolve_damper_points(optional_channels: dict[str, Any]) -> tuple[str, str] | None:
    raw = optional_channels.get("damper_points")
    if not isinstance(raw, list) or len(raw) != 2:
        return None
    return str(raw[0]), str(raw[1])
