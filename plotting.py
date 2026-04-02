from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from corner.core.hardpoints import SuspensionModel


def plot_angles_vs_travel(df: pd.DataFrame, output_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["travel_mm"], df["camber_deg"], label="Camber [deg]", lw=2)
    ax.plot(df["travel_mm"], df["toe_deg"], label="Toe [deg]", lw=2)
    ax.plot(df["travel_mm"], df["caster_deg"], label="Caster [deg]", lw=2)
    ax.plot(df["travel_mm"], df["kingpin_inclination_deg"], label="KPI [deg]", lw=2)
    ax.set_title("Angles vs Wheel Travel")
    ax.set_xlabel("Wheel travel [mm]")
    ax.set_ylabel("Angle [deg]")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_track_and_scrub(df: pd.DataFrame, output_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["travel_mm"], df["track_change_mm"], label="Track change [mm]", lw=2)
    ax.plot(df["travel_mm"], df["lateral_scrub_mm"], label="Lateral scrub [mm]", lw=2)
    ax.plot(df["travel_mm"], df["wheel_longitudinal_shift_mm"], label="Wheel longitudinal shift [mm]", lw=2)
    ax.set_title("Track / Scrub / Longitudinal Shift vs Travel")
    ax.set_xlabel("Wheel travel [mm]")
    ax.set_ylabel("Displacement [mm]")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_wireframe_positions(
    model: SuspensionModel,
    positions: dict[str, np.ndarray] | None,
    output_path: str | Path,
    title: str,
) -> Path:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    if positions is not None:
        xyz = np.array([positions[name] for name in model.all_point_names], dtype=float)
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c="k", s=25)

        for i_name, j_name in model.rigid_links:
            p_i = positions[i_name]
            p_j = positions[j_name]
            ax.plot([p_i[0], p_j[0]], [p_i[1], p_j[1]], [p_i[2], p_j[2]], color="tab:blue", lw=1.5)

        ax.set_xlim(float(np.min(xyz[:, 0])) - 20.0, float(np.max(xyz[:, 0])) + 20.0)
        ax.set_ylim(float(np.min(xyz[:, 1])) - 20.0, float(np.max(xyz[:, 1])) + 20.0)
        ax.set_zlim(float(np.min(xyz[:, 2])) - 20.0, float(np.max(xyz[:, 2])) + 20.0)
    else:
        ax.text2D(0.05, 0.95, "Invalid mechanism at this travel", transform=ax.transAxes)

    ax.set_title(title)
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
