from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_build_ood_splits_from_id_smoke(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    id_root = repo / "data" / "splits" / "id"
    if not (id_root / "train_segments.parquet").is_file():
        import pytest

        pytest.skip("bundled ID data not present")
    out = tmp_path / "ood"
    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "build_ood_splits_from_id.py"),
            "--id-root",
            str(id_root),
            "--out-root",
            str(out),
            "--domains",
            "traffic",
        ],
        check=True,
        cwd=str(repo),
    )
    assert (out / "OOD_traffic" / "train_segments_OOD_traffic.parquet").is_file()
    assert (out / "Datasets" / "OOD" / "test_perf_OOD_traffic.csv").is_file()
