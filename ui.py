from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from corner.simulator import load_model_and_options, resolve_damper_points, run_single_corner_sweep
    from corner.core.hardpoints import SuspensionModel
except ModuleNotFoundError:
    from simulator import load_model_and_options, resolve_damper_points, run_single_corner_sweep
    from core.hardpoints import SuspensionModel


def _build_angles_figure(df: pd.DataFrame) -> plt.Figure:
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
    return fig


def _build_scrub_figure(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["travel_mm"], df["track_change_mm"], label="Track change [mm]", lw=2)
    ax.plot(df["travel_mm"], df["lateral_scrub_mm"], label="Lateral scrub [mm]", lw=2)
    ax.plot(df["travel_mm"], df["wheel_longitudinal_shift_mm"], label="Longitudinal shift [mm]", lw=2)
    ax.set_title("Track / Scrub / Longitudinal Shift vs Travel")
    ax.set_xlabel("Wheel travel [mm]")
    ax.set_ylabel("Displacement [mm]")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def _build_geometry_constraints_figure(
    model: SuspensionModel,
    points: dict[str, np.ndarray] | None,
    damper_points: tuple[str, str] | None,
    show_labels: bool,
) -> go.Figure:
    fig = go.Figure()

    if points is None:
        fig.update_layout(title="Geometry + Constraints (invalid mechanism at this travel)")
        return fig

    all_names = model.all_point_names
    fixed_names = [name for name in all_names if name in model.fixed_points]
    movable_names = [name for name in all_names if name not in model.fixed_points]

    # Rigid-distance constraints are visualized as linkage lines.
    for i_name, j_name in model.rigid_links:
        p_i = points[i_name]
        p_j = points[j_name]
        fig.add_trace(
            go.Scatter3d(
                x=[float(p_i[0]), float(p_j[0])],
                y=[float(p_i[1]), float(p_j[1])],
                z=[float(p_i[2]), float(p_j[2])],
                mode="lines",
                line={"color": "#1f77b4", "width": 4},
                showlegend=False,
                hoverinfo="skip",
            )
        )

    if fixed_names:
        xyz_fixed = np.array([points[name] for name in fixed_names], dtype=float)
        fig.add_trace(
            go.Scatter3d(
                x=xyz_fixed[:, 0],
                y=xyz_fixed[:, 1],
                z=xyz_fixed[:, 2],
                mode="markers+text" if show_labels else "markers",
                marker={"size": 5, "color": "#d62728", "symbol": "square"},
                text=fixed_names if show_labels else None,
                textposition="top center",
                name="Fixed points",
            )
        )

    if movable_names:
        xyz_movable = np.array([points[name] for name in movable_names], dtype=float)
        fig.add_trace(
            go.Scatter3d(
                x=xyz_movable[:, 0],
                y=xyz_movable[:, 1],
                z=xyz_movable[:, 2],
                mode="markers+text" if show_labels else "markers",
                marker={"size": 4, "color": "#111111", "symbol": "circle"},
                text=movable_names if show_labels else None,
                textposition="top center",
                name="Movable points",
            )
        )

    # Input DOF constraint direction as a vector.
    dof_name, dof_axis = model.input_dof_target
    p0 = points[dof_name]
    axis_vec = np.zeros(3, dtype=float)
    axis_vec[dof_axis] = 1.0
    span = np.ptp(np.array([points[name] for name in all_names], dtype=float), axis=0)
    arrow_len = max(float(np.max(span)) * 0.12, 40.0)
    axis_vec *= arrow_len
    p1 = p0 + axis_vec
    fig.add_trace(
        go.Scatter3d(
            x=[float(p0[0]), float(p1[0])],
            y=[float(p0[1]), float(p1[1])],
            z=[float(p0[2]), float(p1[2])],
            mode="lines+markers",
            line={"color": "#2ca02c", "width": 7},
            marker={"size": [2, 5], "color": "#2ca02c"},
            name="Input DOF direction",
        )
    )

    # Optional damper measurement points and connection.
    if damper_points is not None:
        p_move_name, p_fix_name = damper_points
        if p_move_name in points and p_fix_name in points:
            p_m = points[p_move_name]
            p_f = points[p_fix_name]
            fig.add_trace(
                go.Scatter3d(
                    x=[float(p_m[0]), float(p_f[0])],
                    y=[float(p_m[1]), float(p_f[1])],
                    z=[float(p_m[2]), float(p_f[2])],
                    mode="lines",
                    line={"color": "#9467bd", "width": 6, "dash": "dash"},
                    name="Damper measurement",
                )
            )

    # Optional axle-spin constraints if present in YAML model.
    for idx, joint in enumerate(model.axle_spin_joints):
        if joint.point_name not in points:
            continue
        center = joint.resolve_axis_point(points)
        direction = joint.mirrored_axis_direction(right_side=False)
        axis_len = max(float(np.max(span)) * 0.10, 30.0)
        a0 = center - direction * axis_len
        a1 = center + direction * axis_len
        fig.add_trace(
            go.Scatter3d(
                x=[float(a0[0]), float(a1[0])],
                y=[float(a0[1]), float(a1[1])],
                z=[float(a0[2]), float(a1[2])],
                mode="lines",
                line={"color": "#ff7f0e", "width": 6, "dash": "dot"},
                name="Axle-spin axis" if idx == 0 else "Axle-spin axis (extra)",
                showlegend=idx == 0,
            )
        )

    fig.update_layout(
        title="Interactive Geometry + Constraints",
        height=760,
        legend={"orientation": "h", "y": 1.02, "x": 0.0},
        scene={
            "xaxis_title": "X [mm]",
            "yaxis_title": "Y [mm]",
            "zaxis_title": "Z [mm]",
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
    )
    return fig


def _constraints_summary_table(
    model: SuspensionModel,
    damper_points: tuple[str, str] | None,
) -> pd.DataFrame:
    dof_name, dof_axis = model.input_dof_target
    axis_label = ["x", "y", "z"][dof_axis]
    rows: list[tuple[str, str]] = [
        ("Total points", str(len(model.all_point_names))),
        ("Fixed points", str(len(model.fixed_points))),
        ("Movable points", str(len(model.movable_points))),
        ("Rigid links", str(len(model.rigid_links))),
        ("Input DOF", f"{dof_name}_{axis_label}"),
        ("Upright points", ", ".join(model.upright_points)),
        ("Wheel plane points", ", ".join(model.wheel_plane_points)),
        ("Axle-spin constraints", str(len(model.axle_spin_joints))),
        (
            "Damper points",
            "None" if damper_points is None else f"{damper_points[0]} -> {damper_points[1]}",
        ),
    ]
    return pd.DataFrame(rows, columns=["constraint", "value"])


def _load_uploaded_yaml(uploaded_file) -> tuple[SuspensionModel, dict]:
    suffix = Path(uploaded_file.name).suffix or ".yaml"
    with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False) as handle:
        handle.write(uploaded_file.getvalue())
        path = Path(handle.name)
    try:
        return load_model_and_options(path)
    finally:
        path.unlink(missing_ok=True)


def _summary_table(df: pd.DataFrame, ride_height_mm: float) -> pd.DataFrame:
    idx = int(np.argmin(np.abs(df["travel_mm"].to_numpy(dtype=float) - ride_height_mm)))
    row = df.iloc[idx]
    metrics = [
        ("camber_deg", "deg"),
        ("toe_deg", "deg"),
        ("caster_deg", "deg"),
        ("kingpin_inclination_deg", "deg"),
        ("scrub_radius_mm", "mm"),
        ("mechanical_trail_mm", "mm"),
        ("roll_center_z_mm", "mm"),
        ("track_change_mm", "mm"),
        ("wheel_longitudinal_shift_mm", "mm"),
        ("motion_ratio", "-"),
    ]
    return pd.DataFrame(
        {
            "metric": [m for m, _u in metrics],
            "value": [float(row[m]) for m, _u in metrics],
            "units": [u for _m, u in metrics],
        }
    )


def main() -> None:
    st.set_page_config(page_title="Corner Kinematics UI", layout="wide")
    st.title("Single-Corner Double Wishbone Kinematics")
    st.caption("Cross-platform browser UI (Windows/macOS/Linux) using Streamlit")

    default_model = Path(__file__).resolve().parent / "models" / "single_corner_double_wishbone.yaml"

    with st.sidebar:
        st.header("Inputs")
        model_source = st.radio("Model source", ["Default model", "Custom file upload"], index=0)
        uploaded_file = st.file_uploader("Upload YAML", type=["yaml", "yml"])

        travel_min = st.number_input("Travel min [mm]", value=-50.0, step=1.0)
        travel_max = st.number_input("Travel max [mm]", value=50.0, step=1.0)
        step = st.number_input("Step [mm]", value=5.0, min_value=0.1, step=0.5)
        ride_height = st.number_input("Ride height reference [mm]", value=0.0, step=1.0)
        right_side = st.checkbox("Mirror to right side", value=False)

    if travel_max <= travel_min:
        st.error("Travel max must be greater than travel min.")
        st.stop()

    selected_travel = [-50.0, 0.0, 50.0]

    try:
        if model_source == "Custom file upload" and uploaded_file is not None:
            model, optional_channels = _load_uploaded_yaml(uploaded_file)
            model_name = uploaded_file.name
        else:
            model, optional_channels = load_model_and_options(default_model)
            model_name = str(default_model)
    except Exception as exc:
        st.error(f"Model load failed: {exc}")
        st.stop()

    damper_points = resolve_damper_points(optional_channels)

    with st.spinner("Running kinematic sweep..."):
        try:
            result = run_single_corner_sweep(
                model=model,
                travel_min_mm=float(travel_min),
                travel_max_mm=float(travel_max),
                step_mm=float(step),
                ride_height_travel_mm=float(ride_height),
                right_side=bool(right_side),
                damper_points=damper_points,
            )
        except Exception as exc:
            st.error(f"Solve failed: {exc}")
            st.stop()

    df = result.to_dataframe()
    valid_ratio = 100.0 * float(result.valid_mask.mean()) if len(result.valid_mask) > 0 else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model", model_name)
    m2.metric("Steps", str(len(df)))
    m3.metric("Valid steps", f"{int(result.valid_mask.sum())}/{len(result.valid_mask)}")
    m4.metric("Validity", f"{valid_ratio:.1f}%")

    tabs = st.tabs(["Plots", "Ride Height Summary", "Selected Travel", "Geometry + Constraints", "Raw Data"])

    with tabs[0]:
        st.pyplot(_build_angles_figure(df), use_container_width=True)
        st.pyplot(_build_scrub_figure(df), use_container_width=True)

    with tabs[1]:
        summary = _summary_table(df, float(ride_height))
        st.dataframe(summary, use_container_width=True)

    with tabs[2]:
        travel_array = df["travel_mm"].to_numpy(dtype=float)
        row_indices = [int(np.argmin(np.abs(travel_array - t))) for t in selected_travel]
        selected_df = df.iloc[row_indices].copy()
        st.dataframe(selected_df, use_container_width=True)

    with tabs[3]:
        travel_array = df["travel_mm"].to_numpy(dtype=float)
        default_target = 0.0
        geometry_travel = st.slider(
            "Geometry snapshot [mm]",
            min_value=float(travel_array.min()),
            max_value=float(travel_array.max()),
            value=float(np.clip(default_target, travel_array.min(), travel_array.max())),
            key="geometry_snapshot",
        )
        show_labels = st.checkbox("Show point labels", value=True)

        idx = int(np.argmin(np.abs(travel_array - geometry_travel)))
        fig_geometry = _build_geometry_constraints_figure(
            model=model,
            points=result.positions[idx],
            damper_points=damper_points,
            show_labels=show_labels,
        )
        st.plotly_chart(fig_geometry, use_container_width=True)

        st.subheader("Constraint Summary")
        st.dataframe(_constraints_summary_table(model, damper_points), use_container_width=True)

    with tabs[4]:
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="single_corner_results.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download JSON",
            data=df.to_json(orient="records", indent=2).encode("utf-8"),
            file_name="single_corner_results.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
