from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from corner.plotting import plot_angles_vs_travel, plot_track_and_scrub, plot_wireframe_positions
from corner.simulator import load_model_and_options, resolve_damper_points, run_single_corner_sweep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="corner.run_corner")
    parser.add_argument(
        "--model",
        default="corner/models/single_corner_double_wishbone.yaml",
        help="Path to model YAML file",
    )
    parser.add_argument("--travel-min", type=float, default=-100.0, help="Minimum travel [mm]")
    parser.add_argument("--travel-max", type=float, default=100.0, help="Maximum travel [mm]")
    parser.add_argument("--step", type=float, default=5.0, help="Travel step [mm]")
    parser.add_argument("--ride-height", type=float, default=0.0, help="Ride-height travel reference [mm]")
    parser.add_argument("--right-side", action="store_true", help="Mirror model to right side")
    parser.add_argument(
        "--selected-travel",
        nargs="*",
        type=float,
        default=[-50.0, 0.0, 50.0],
        help="Travel values for selected-points table [mm]",
    )
    parser.add_argument(
        "--output-dir",
        default="corner/outputs",
        help="Directory for CSV/JSON/plots",
    )
    return parser


def _nearest_rows(df: pd.DataFrame, targets_mm: list[float]) -> pd.DataFrame:
    idxs = [int(np.argmin(np.abs(df["travel_mm"].to_numpy(dtype=float) - t))) for t in targets_mm]
    return df.iloc[idxs].copy()


def _geometry_summary(df: pd.DataFrame, ride_height_mm: float) -> pd.DataFrame:
    idx = int(np.argmin(np.abs(df["travel_mm"].to_numpy(dtype=float) - ride_height_mm)))
    row = df.iloc[idx]
    keys = [
        "camber_deg",
        "toe_deg",
        "caster_deg",
        "kingpin_inclination_deg",
        "scrub_radius_mm",
        "mechanical_trail_mm",
        "roll_center_z_mm",
        "instant_center_y_mm",
        "instant_center_z_mm",
        "track_change_mm",
        "wheel_longitudinal_shift_mm",
        "motion_ratio",
    ]
    return pd.DataFrame(
        {
            "metric": keys,
            "value": [float(row[k]) if k in row else float("nan") for k in keys],
            "units": [
                "deg",
                "deg",
                "deg",
                "deg",
                "mm",
                "mm",
                "mm",
                "mm",
                "mm",
                "mm",
                "mm",
                "-",
            ],
        }
    )


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, optional_channels = load_model_and_options(args.model)
    damper_points = resolve_damper_points(optional_channels)

    result = run_single_corner_sweep(
        model=model,
        travel_min_mm=float(args.travel_min),
        travel_max_mm=float(args.travel_max),
        step_mm=float(args.step),
        ride_height_travel_mm=float(args.ride_height),
        right_side=bool(args.right_side),
        damper_points=damper_points,
    )

    df = result.to_dataframe()
    summary = _geometry_summary(df, ride_height_mm=float(args.ride_height))
    selected = _nearest_rows(df, [float(v) for v in args.selected_travel])

    csv_path = output_dir / "single_corner_results.csv"
    json_path = output_dir / "single_corner_results.json"
    summary_csv = output_dir / "ride_height_summary.csv"
    selected_csv = output_dir / "selected_travel_points.csv"

    df.to_csv(csv_path, index=False)
    summary.to_csv(summary_csv, index=False)
    selected.to_csv(selected_csv, index=False)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(df.to_dict(orient="records"), handle, indent=2)

    plot_angles = plot_angles_vs_travel(df, output_dir / "angles_vs_travel.png")
    plot_scrub = plot_track_and_scrub(df, output_dir / "track_and_scrub_vs_travel.png")

    # Generate 3D wireframes around requested key travel values.
    travel_array = df["travel_mm"].to_numpy(dtype=float)
    for t in [float(v) for v in args.selected_travel]:
        idx = int(np.argmin(np.abs(travel_array - t)))
        position = result.positions[idx]
        plot_wireframe_positions(
            model,
            position,
            output_dir / f"wireframe_{travel_array[idx]:+.0f}mm.png",
            title=f"Single-Corner Wireframe @ {travel_array[idx]:+.1f} mm",
        )

    invalid_steps = int((~result.valid_mask).sum())
    print(f"Model: {args.model}")
    print(f"Travel range: {args.travel_min:.1f} .. {args.travel_max:.1f} mm (step {args.step:.1f} mm)")
    print(f"Invalid steps: {invalid_steps}")
    print(f"Results CSV: {csv_path}")
    print(f"Results JSON: {json_path}")
    print(f"Ride-height summary: {summary_csv}")
    print(f"Selected travel table: {selected_csv}")
    print(f"Plot: {plot_angles}")
    print(f"Plot: {plot_scrub}")


if __name__ == "__main__":
    main()
