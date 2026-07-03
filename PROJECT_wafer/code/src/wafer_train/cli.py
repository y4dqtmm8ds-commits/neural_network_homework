"""Command-line entrypoint for the wafer training engine."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ENGINE_PATH = Path(__file__).resolve().with_name("engine.py")


def main(args: list[str] | None = None) -> None:
    """Run the shared training engine with an explicit argument list."""
    argv = [] if args is None else list(args)
    sys.argv = [str(ENGINE_PATH), *argv]
    runpy.run_path(str(ENGINE_PATH), run_name="__main__")


if __name__ == "__main__":
    main(sys.argv[1:])
