from __future__ import annotations

import numpy as np

try:
    from corner.core.hardpoints import SuspensionModel
except ModuleNotFoundError:
    from core.hardpoints import SuspensionModel


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("Cannot normalize near-zero vector")
    return vector / norm


def wheel_plane_frame(model: SuspensionModel, points: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local wheel frame unit vectors (forward, normal, up)."""
    center_name, top_name, front_name = model.wheel_plane_points
    c = points[center_name]
    top = points[top_name]
    front = points[front_name]

    up_dir = _normalize(top - c)
    forward_raw = front - c
    # Orthogonalize forward against up to improve numerical stability.
    forward = _normalize(forward_raw - np.dot(forward_raw, up_dir) * up_dir)
    normal = _normalize(np.cross(up_dir, forward))
    return forward, normal, up_dir


def camber_deg(model: SuspensionModel, points: dict[str, np.ndarray]) -> float:
    """Camber from wheel up-direction in vehicle YZ plane (deg)."""
    _, _, up_dir = wheel_plane_frame(model, points)
    return float(np.degrees(np.arctan2(up_dir[1], up_dir[2])))


def toe_deg(model: SuspensionModel, points: dict[str, np.ndarray]) -> float:
    """Toe from wheel forward-direction in vehicle XY plane (deg)."""
    forward, _, _ = wheel_plane_frame(model, points)
    return float(np.degrees(np.arctan2(forward[1], forward[0])))


def kingpin_axis(model: SuspensionModel, points: dict[str, np.ndarray]) -> np.ndarray:
    lower_name, upper_name, _ = model.upright_points
    return _normalize(points[upper_name] - points[lower_name])


def caster_deg(model: SuspensionModel, points: dict[str, np.ndarray]) -> float:
    """Caster from kingpin axis in side view (deg)."""
    kp = kingpin_axis(model, points)
    return float(np.degrees(np.arctan2(kp[0], kp[2])))


def kingpin_inclination_deg(model: SuspensionModel, points: dict[str, np.ndarray]) -> float:
    """KPI from kingpin axis in front view (deg)."""
    kp = kingpin_axis(model, points)
    return float(np.degrees(np.arctan2(kp[1], kp[2])))


def steering_axis_ground_intersection(
    model: SuspensionModel,
    points: dict[str, np.ndarray],
) -> np.ndarray:
    lower_name, upper_name, _ = model.upright_points
    p0 = points[lower_name]
    axis = points[upper_name] - points[lower_name]
    if abs(axis[2]) < 1e-9:
        return np.array([np.nan, np.nan, model.ground_z], dtype=float)
    t = (model.ground_z - p0[2]) / axis[2]
    return p0 + t * axis


def contact_patch_estimate(model: SuspensionModel, points: dict[str, np.ndarray]) -> np.ndarray:
    wc = points[model.wheel_center_name]
    return np.array([wc[0], wc[1], model.ground_z], dtype=float)


def scrub_radius_mm(model: SuspensionModel, points: dict[str, np.ndarray]) -> float:
    sag = steering_axis_ground_intersection(model, points)
    cp = contact_patch_estimate(model, points)
    return float(cp[1] - sag[1])


def mechanical_trail_mm(model: SuspensionModel, points: dict[str, np.ndarray]) -> float:
    sag = steering_axis_ground_intersection(model, points)
    cp = contact_patch_estimate(model, points)
    return float(cp[0] - sag[0])


def _line_intersection_yz(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
) -> np.ndarray:
    """Intersection in YZ plane for two lines; returns [y, z]."""
    a = np.array([[p2[0] - p1[0], -(p4[0] - p3[0])], [p2[1] - p1[1], -(p4[1] - p3[1])]], dtype=float)
    b = np.array([p3[0] - p1[0], p3[1] - p1[1]], dtype=float)
    det = np.linalg.det(a)
    if abs(det) < 1e-10:
        return np.array([np.nan, np.nan], dtype=float)
    t, _u = np.linalg.solve(a, b)
    y = p1[0] + t * (p2[0] - p1[0])
    z = p1[1] + t * (p2[1] - p1[1])
    return np.array([y, z], dtype=float)


def instant_center_front_view(model: SuspensionModel, points: dict[str, np.ndarray]) -> np.ndarray:
    """Front-view instant center (Y, Z) from upper/lower arm centerlines."""
    required = {
        "chassis_lower_front",
        "chassis_lower_rear",
        "chassis_upper_front",
        "chassis_upper_rear",
        "lower_ball_joint",
        "upper_ball_joint",
    }
    if not required.issubset(points):
        return np.array([np.nan, np.nan], dtype=float)

    lower_mid = 0.5 * (points["chassis_lower_front"] + points["chassis_lower_rear"])
    upper_mid = 0.5 * (points["chassis_upper_front"] + points["chassis_upper_rear"])

    # Project to YZ plane.
    l1 = np.array([lower_mid[1], lower_mid[2]], dtype=float)
    l2 = np.array([points["lower_ball_joint"][1], points["lower_ball_joint"][2]], dtype=float)
    u1 = np.array([upper_mid[1], upper_mid[2]], dtype=float)
    u2 = np.array([points["upper_ball_joint"][1], points["upper_ball_joint"][2]], dtype=float)

    return _line_intersection_yz(l1, l2, u1, u2)


def roll_center_height_mm(model: SuspensionModel, points: dict[str, np.ndarray]) -> float:
    """Approximate front-view roll center height at vehicle centerline Y=0."""
    ic = instant_center_front_view(model, points)
    cp = contact_patch_estimate(model, points)
    if np.isnan(ic).any():
        return float("nan")

    y_ic, z_ic = ic
    y_cp, z_cp = cp[1], cp[2]
    if abs(y_cp - y_ic) < 1e-10:
        return float("nan")

    slope = (z_cp - z_ic) / (y_cp - y_ic)
    z_centerline = z_ic + slope * (0.0 - y_ic)
    return float(z_centerline)


def compute_kinematic_channels(model: SuspensionModel, points: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute standard kinematic output channels for one solved position."""
    wc = points[model.wheel_center_name]
    ic = instant_center_front_view(model, points)

    return {
        "wheel_center_x_mm": float(wc[0]),
        "wheel_center_y_mm": float(wc[1]),
        "wheel_center_z_mm": float(wc[2]),
        "camber_deg": camber_deg(model, points),
        "toe_deg": toe_deg(model, points),
        "caster_deg": caster_deg(model, points),
        "kingpin_inclination_deg": kingpin_inclination_deg(model, points),
        "scrub_radius_mm": scrub_radius_mm(model, points),
        "mechanical_trail_mm": mechanical_trail_mm(model, points),
        "instant_center_y_mm": float(ic[0]),
        "instant_center_z_mm": float(ic[1]),
        "roll_center_z_mm": roll_center_height_mm(model, points),
    }
