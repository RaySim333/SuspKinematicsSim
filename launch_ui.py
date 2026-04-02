from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="corner.launch_ui")
    parser.add_argument("--host", default="127.0.0.1", help="UI host address")
    parser.add_argument("--port", type=int, default=8502, help="UI port")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    app_path = Path(__file__).resolve().parent / "ui.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        str(args.host),
        "--server.port",
        str(args.port),
    ]

    env = os.environ.copy()
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    env.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
