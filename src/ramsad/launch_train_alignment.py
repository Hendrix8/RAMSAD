"""Console entry: run vendored `alignment/train.py` with the same argv (Hydra training)."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    train = repo / "alignment" / "train.py"
    if not train.is_file():
        raise SystemExit(f"Missing alignment trainer: {train}")
    os.execv(sys.executable, [sys.executable, str(train), *sys.argv[1:]])


if __name__ == "__main__":
    main()
