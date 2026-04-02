from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from corner.utils.yaml_loader import load_yaml, to_point_dict


@dataclass
class AxleSpinJoint:
    """Constraint definition for a point that can only spin around a fixed axle."""

    point_name: str
    axis_direction: np.ndarray
    axis_point_name: str | None = None
    axis_point: np.ndarray | None = None

    @classmethod
    def from_yaml(cls, data: dict[str, Any]) -> "AxleSpinJoint":
        point_name = str(data["point"])
        axis_direction = np.asarray(data["axis_direction"], dtype=float)

        axis_point_name: str | None = None
        axis_point: np.ndarray | None = None
        raw_axis_point = data.get("axis_point")
        if isinstance(raw_axis_point, str):
            axis_point_name = raw_axis_point
        elif raw_axis_point is not None:
            axis_point = np.asarray(raw_axis_point, dtype=float)

        return cls(
            point_name=point_name,
            axis_direction=axis_direction,
            axis_point_name=axis_point_name,
            axis_point=axis_point,
        )

    def mirrored_axis_direction(self, right_side: bool = False) -> np.ndarray:
        axis = self.axis_direction.astype(float).copy()
        if right_side:
            axis[1] *= -1.0
        norm = float(np.linalg.norm(axis))
        if norm < 1e-12:
            raise ValueError(f"Invalid axis_direction for point '{self.point_name}'")
        return axis / norm

    def resolve_axis_point(
        self,
        points: dict[str, np.ndarray],
        right_side: bool = False,
    ) -> np.ndarray:
        if self.axis_point_name is not None:
            return points[self.axis_point_name].copy()

        if self.axis_point is None:
            raise ValueError(f"Missing axis_point for axle-spin joint at '{self.point_name}'")

        point = self.axis_point.astype(float).copy()
        if right_side:
            point[1] *= -1.0
        return point


@dataclass
class SuspensionModel:
    """Container for model topology and hardpoint metadata."""

    model_type: str
    units: str
    hardpoints: dict[str, np.ndarray]
    rigid_links: list[tuple[str, str]]
    fixed_points: set[str]
    upright_points: tuple[str, str, str]
    wheel_plane_points: tuple[str, str, str]
    input_dof: str
    wheel_center_name: str
    ground_z: float
    axle_spin_joints: list[AxleSpinJoint] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._validate_names()
        self.nominal_lengths = self._compute_nominal_lengths(self.hardpoints)

    @property
    def input_dof_target(self) -> tuple[str, int]:
        """Parse input DOF string into (point_name, axis_index)."""
        axis_map = {"x": 0, "y": 1, "z": 2}
        if "_" not in self.input_dof:
            raise ValueError(
                f"Invalid input_dof '{self.input_dof}'. Expected format '<point>_<axis>'"
            )
        point_name, axis_name = self.input_dof.rsplit("_", 1)
        axis_name = axis_name.lower()
        if axis_name not in axis_map:
            raise ValueError(
                f"Invalid input_dof axis '{axis_name}'. Must be one of: x, y, z"
            )
        if point_name not in self.hardpoints:
            raise ValueError(f"Invalid input_dof point '{point_name}' not in hardpoints")
        return point_name, axis_map[axis_name]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SuspensionModel":
        data = load_yaml(path)
        hardpoints = to_point_dict(data["hardpoints"])
        rigid_links = [tuple(link) for link in data["rigid_links"]]
        axle_spin_joints = [
            AxleSpinJoint.from_yaml(entry) for entry in data.get("axle_spin_joints", [])
        ]

        return cls(
            model_type=str(data.get("type", "custom")),
            units=str(data.get("units", "mm")),
            hardpoints=hardpoints,
            rigid_links=rigid_links,
            fixed_points=set(data["fixed_points"]),
            upright_points=tuple(data["upright_points"]),
            wheel_plane_points=tuple(data.get("wheel_plane_points", data["upright_points"])),
            input_dof=str(data["input_dof"]),
            wheel_center_name=str(data.get("wheel_center_name", "wheel_center")),
            ground_z=float(data.get("ground_z", 0.0)),
            axle_spin_joints=axle_spin_joints,
        )

    @property
    def all_point_names(self) -> list[str]:
        return list(self.hardpoints.keys())

    @property
    def movable_points(self) -> list[str]:
        return [name for name in self.all_point_names if name not in self.fixed_points]

    def mirrored_points(self, right_side: bool = False) -> dict[str, np.ndarray]:
        """Return points for left side, or mirrored to right side about Y=0."""
        if not right_side:
            return {k: v.copy() for k, v in self.hardpoints.items()}

        mirrored: dict[str, np.ndarray] = {}
        for name, point in self.hardpoints.items():
            p = point.copy()
            p[1] *= -1.0
            mirrored[name] = p
        return mirrored

    def translated(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> "SuspensionModel":
        """Return a new model translated in global coordinates."""
        delta = np.array([dx, dy, dz], dtype=float)
        new_points = {name: point + delta for name, point in self.hardpoints.items()}

        translated_joints: list[AxleSpinJoint] = []
        for joint in self.axle_spin_joints:
            axis_point = None
            if joint.axis_point is not None:
                axis_point = joint.axis_point + delta

            translated_joints.append(
                AxleSpinJoint(
                    point_name=joint.point_name,
                    axis_direction=joint.axis_direction.copy(),
                    axis_point_name=joint.axis_point_name,
                    axis_point=axis_point,
                )
            )

        return SuspensionModel(
            model_type=self.model_type,
            units=self.units,
            hardpoints=new_points,
            rigid_links=list(self.rigid_links),
            fixed_points=set(self.fixed_points),
            upright_points=tuple(self.upright_points),
            wheel_plane_points=tuple(self.wheel_plane_points),
            input_dof=self.input_dof,
            wheel_center_name=self.wheel_center_name,
            ground_z=self.ground_z,
            axle_spin_joints=translated_joints,
        )

    def _compute_nominal_lengths(self, points: dict[str, np.ndarray]) -> dict[tuple[str, str], float]:
        lengths: dict[tuple[str, str], float] = {}
        for i_name, j_name in self.rigid_links:
            p_i = points[i_name]
            p_j = points[j_name]
            lengths[(i_name, j_name)] = float(np.linalg.norm(p_i - p_j))
        return lengths

    def _validate_names(self) -> None:
        all_names = set(self.hardpoints.keys())

        for fixed_name in self.fixed_points:
            if fixed_name not in all_names:
                raise ValueError(f"Unknown fixed point '{fixed_name}'")

        for i_name, j_name in self.rigid_links:
            if i_name not in all_names or j_name not in all_names:
                raise ValueError(f"Unknown rigid link endpoints: ({i_name}, {j_name})")

        for name in self.upright_points:
            if name not in all_names:
                raise ValueError(f"Unknown upright point '{name}'")

        for name in self.wheel_plane_points:
            if name not in all_names:
                raise ValueError(f"Unknown wheel plane point '{name}'")

        if self.wheel_center_name not in all_names:
            raise ValueError(f"Unknown wheel center '{self.wheel_center_name}'")

        for joint in self.axle_spin_joints:
            if joint.point_name not in all_names:
                raise ValueError(f"Unknown axle-spin joint point '{joint.point_name}'")

            if joint.axis_point_name is not None:
                if joint.axis_point_name not in all_names:
                    raise ValueError(
                        f"Unknown axle-spin axis point '{joint.axis_point_name}'"
                    )
                if joint.axis_point_name not in self.fixed_points:
                    raise ValueError(
                        f"Axle-spin axis point '{joint.axis_point_name}' must be fixed"
                    )

            if joint.axis_point is not None and joint.axis_point.shape != (3,):
                raise ValueError(
                    f"axle-spin axis_point for '{joint.point_name}' must have 3 coordinates"
                )

            if joint.axis_direction.shape != (3,):
                raise ValueError(
                    f"axle-spin axis_direction for '{joint.point_name}' must have 3 coordinates"
                )

            if float(np.linalg.norm(joint.axis_direction)) < 1e-12:
                raise ValueError(
                    f"axle-spin axis_direction for '{joint.point_name}' must be non-zero"
                )


@dataclass
class SweepResult:
    """Computed positions and derived channels for a bump/rebound sweep."""

    travel_values: np.ndarray
    positions: list[dict[str, np.ndarray]]
    channels: dict[str, np.ndarray]

    def to_records(self) -> list[dict[str, float]]:
        records: list[dict[str, float]] = []
        for idx, travel in enumerate(self.travel_values):
            row: dict[str, float] = {"travel_mm": float(travel)}
            for key, values in self.channels.items():
                row[key] = float(values[idx])
            records.append(row)
        return records
