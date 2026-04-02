from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

try:
    from corner.core.constraints import ConstraintSystem
    from corner.core.hardpoints import SuspensionModel, SweepResult
    from corner.core.kinematics_calcs import compute_kinematic_channels
except ModuleNotFoundError:
    from core.constraints import ConstraintSystem
    from core.hardpoints import SuspensionModel, SweepResult
    from core.kinematics_calcs import compute_kinematic_channels


@dataclass
class KinematicSolver:
    """Nonlinear rigid-body kinematics solver using least squares."""

    model: SuspensionModel
    right_side: bool = False

    def __post_init__(self) -> None:
        self.initial_points = self.model.mirrored_points(right_side=self.right_side)
        self.constraint_system = ConstraintSystem(self.model, right_side=self.right_side)

    def solve_position(
        self,
        target_input_value: float,
        initial_guess: np.ndarray | None = None,
        max_nfev: int = 400,
    ) -> dict[str, np.ndarray]:
        """Solve one static position for the selected input DOF value."""
        if initial_guess is None:
            initial_guess = self.constraint_system.pack_variables(self.initial_points)

        residual_fn = self.constraint_system.make_residual_function(
            target_input_value=target_input_value,
            base_points=self.initial_points,
        )

        n_vars = int(initial_guess.size)
        n_residuals = int(residual_fn(initial_guess).size)
        method = "lm" if n_residuals >= n_vars else "trf"

        result = least_squares(
            residual_fn,
            x0=initial_guess,
            method=method,
            max_nfev=max_nfev,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        if not result.success:
            raise RuntimeError(f"Solver failed: {result.message}")

        solved = self.constraint_system.unpack_variables(result.x, self.initial_points)
        return solved

    def sweep_wheel_travel(
        self,
        travel_min_mm: float,
        travel_max_mm: float,
        step_mm: float,
    ) -> SweepResult:
        """Run a bump/rebound sweep and evaluate derived channels at each step."""
        if step_mm <= 0.0:
            raise ValueError("step_mm must be positive")

        input_point_name, input_axis_idx = self.model.input_dof_target
        nominal_input = self.initial_points[input_point_name][input_axis_idx]
        travel_values = np.arange(travel_min_mm, travel_max_mm + 0.5 * step_mm, step_mm)

        solved_positions: list[dict[str, np.ndarray]] = []
        channel_accumulator: dict[str, list[float]] = {}

        current_guess = self.constraint_system.pack_variables(self.initial_points)

        for travel in travel_values:
            target_value = nominal_input + travel
            points = self.solve_position(target_input_value=target_value, initial_guess=current_guess)
            current_guess = self.constraint_system.pack_variables(points)

            solved_positions.append(points)
            channels = compute_kinematic_channels(self.model, points)
            for key, value in channels.items():
                channel_accumulator.setdefault(key, []).append(float(value))

        stacked_channels = {
            name: np.asarray(values, dtype=float) for name, values in channel_accumulator.items()
        }

        return SweepResult(
            travel_values=np.asarray(travel_values, dtype=float),
            positions=solved_positions,
            channels=stacked_channels,
        )
