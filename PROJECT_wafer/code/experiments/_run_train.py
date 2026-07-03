"""Small helpers for launching modular wafer training experiments."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_SRC = PROJECT_ROOT / "code" / "src"
if str(CODE_SRC) not in sys.path:
    sys.path.insert(0, str(CODE_SRC))

from wafer_train.cli import main as train_main


def run_training(args: list[str]) -> None:
    """Run the modular wafer_train engine with a compact experiment arg list."""
    train_main(args)
