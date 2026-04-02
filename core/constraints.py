from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from corner.core.hardpoints import SuspensionModel


@dataclass
class ConstraintSystem:
    """Build and evaluate residuals for rigid-link constraints."""

    model: SuspensionModel
    right_side: bool = False

    def __post_init__(self) -> None:
        self.variable_names = self.model.movable_points
        self.fixed_names = sorted(self.model.fixed_points)

    def pack_variables(self, points: dict[str, np.ndarray]) -> np.ndarray:
        return np.hstack([points[name] for name in self.variable_names])

    def unpack_variables(self, vector: np.ndarray, base_points: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        points = {name: value.copy() for name, value in base_points.items()}
        for idx, name in enumerate(self.variable_names):
            points[name] = vector[idx * 3 : (idx + 1) * 3]
        return points

    def make_residual_function(
        self,
        target_input_value: float,
        base_points: dict[str, np.ndarray],
    ) -> Callable[[np.ndarray], np.ndarray]:
        input_point_name, input_axis_idx = self.model.input_dof_target
        axle_spin_targets: list[tuple[str, np.ndarray, np.ndarray, float, float]] = []
        for joint in self.model.axle_spin_joints:
            axis_point = joint.resolve_axis_point(base_points, right_side=self.right_side)
            axis_dir = joint.mirrored_axis_direction(right_side=self.right_side)

            rel_nominal = base_points[joint.point_name] - axis_point
            nominal_axial = float(np.dot(rel_nominal, axis_dir))
            radial_vec = rel_nominal - nominal_axial * axis_dir
            nominal_radial = float(np.linalg.norm(radial_vec))

            axle_spin_targets.append(
                (joint.point_name, axis_point, axis_dir, nominal_axial, nominal_radial)
            )

        def residual_fn(variable_vector: np.ndarray) -> np.ndarray:
            points = self.unpack_variables(variable_vector, base_points)
            residuals: list[float] = []

            for i_name, j_name in self.model.rigid_links:
                length = np.linalg.norm(points[i_name] - points[j_name])
                nominal = self.model.nominal_lengths[(i_name, j_name)]
                residuals.append(length - nominal)

            for point_name, axis_point, axis_dir, nominal_axial, nominal_radial in axle_spin_targets:
                rel = points[point_name] - axis_point
                axial = float(np.dot(rel, axis_dir))
                radial = float(np.linalg.norm(rel - axial * axis_dir))
                residuals.append(axial - nominal_axial)
                residuals.append(radial - nominal_radial)

            residuals.append(points[input_point_name][input_axis_idx] - target_input_value)
            return np.asarray(residuals, dtype=float)

        return residual_fn
