# Corner: Single-Corner Double Wishbone Kinematics

This folder contains a lightweight simulation project for a single-wheel double wishbone suspension.
It includes its own standalone solver core and can be deployed independently via Streamlit.

## Features

- 3D hardpoint input via YAML
- Wheel travel sweep (bump/rebound)
- Core kinematic outputs per travel step:
  - camber
  - toe (bump steer)
  - caster
  - KPI
  - track change / lateral scrub
  - wheel center longitudinal shift
  - scrub radius / kingpin offset
  - mechanical trail
  - roll center height and front-view instant center
  - motion ratio (if damper points are defined)
- Mechanism validity tracking for each step
- Outputs:
  - CSV and JSON full results
  - ride-height summary table
  - selected-travel values table
  - matplotlib plots
  - interactive geometry + constraints view (single 3D panel)

## Install

From the repository root:

```bash
python3 -m pip install -r corner/requirements.txt
```

The `corner` package is self-contained and does not require imports from `suspension_kinematics` at runtime.

For a standalone deployment, copy the whole `corner/` folder (including `core/`, `utils/`, `models/`, and this README).

## Run

```bash
python3 -m corner.run_corner \
  --model corner/models/single_corner_double_wishbone.yaml \
  --travel-min -100 --travel-max 100 --step 5 \
  --ride-height 0 \
  --selected-travel -50 0 50
```

## Run Browser UI (Cross-OS)

This UI runs in a browser and is compatible with Windows, macOS, and Linux.

```bash
python3 -m corner.launch_ui --host 127.0.0.1 --port 8502
```

For Streamlit deployment platforms, set the app entrypoint to:

```bash
python3 -m streamlit run corner/ui.py --server.port $PORT --server.address 0.0.0.0
```

Alternative entrypoint file (useful for Streamlit Cloud app file setting):

```bash
python3 -m streamlit run corner/streamlit_app.py --server.port $PORT --server.address 0.0.0.0
```

Then open:

- `http://127.0.0.1:8502` (local machine)
- `http://<your-ip>:8502` (another device on the same network)

Notes:

- The launcher uses `sys.executable`, so it works with your active Python environment on each OS.
- The UI supports default model and uploaded YAML workflows.
- You can download CSV/JSON directly from the UI.
- The UI includes a Geometry + Constraints tab showing:
  - fixed vs movable points
  - rigid-link constraints
  - input DOF direction
  - optional damper and axle-spin constraint annotations
  - interactive camera controls (rotate, pan, zoom)

Optional flags:

- `--right-side` to mirror geometry to the right corner
- `--output-dir` to choose an output folder

## Input Model Notes

Main geometry fields are in `hardpoints` and `rigid_links`.

For optional motion ratio output, define:

```yaml
optional_channels:
  damper_points:
    - damper_upright_mount
    - damper_chassis_mount
```

The first point is the moving damper point, the second is fixed on chassis.

## Output Files

Default output folder: `corner/outputs/`

- `single_corner_results.csv`
- `single_corner_results.json`
- `ride_height_summary.csv`
- `selected_travel_points.csv`
- `angles_vs_travel.png`
- `track_and_scrub_vs_travel.png`
- `wireframe_*.png`
