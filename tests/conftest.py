from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _emb(ix: int, dim: int = 8) -> list[float]:
    rng = np.random.default_rng(ix)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / (np.linalg.norm(v) + 1e-9)).tolist()


def write_fixture_tree(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    train = root / "train_segments.parquet"
    test = root / "test_segments.parquet"
    tr_ser = root / "train_series.parquet"
    te_ser = root / "test_series.parquet"

    # Two train series, two segments each (seg_pos 0 then 1)
    train_rows = []
    for sid, name in enumerate(["010_a", "020_b"]):
        for sp in (0, 1):
            L = 16
            s = np.linspace(0, 1, L).astype(np.float32)
            m = np.ones(L, dtype=bool)
            train_rows.append(
                {
                    "series": s.tolist(),
                    "mask": m.tolist(),
                    "label": 0,
                    "dataset_name": name,
                    "seg_pos": sp,
                    "csv_name": f"{name}.csv",
                    "amazon_chronos-2": _emb(100 * sid + sp),
                }
            )
    pd.DataFrame(train_rows).to_parquet(train, index=False)

    test_rows = []
    for sp in (0, 1):
        L = 16
        s = np.linspace(0.2, 1.2, L).astype(np.float32)
        m = np.ones(L, dtype=bool)
        test_rows.append(
            {
                "series": s.tolist(),
                "mask": m.tolist(),
                "label": 0,
                "dataset_name": "005_q",
                "seg_pos": sp,
                "csv_name": "005_q.csv",
                "amazon_chronos-2": _emb(500 + sp),
            }
        )
    pd.DataFrame(test_rows).to_parquet(test, index=False)

    pd.DataFrame(
        {
            "csv_name": ["010_a.csv", "020_b.csv"],
            "class": ["c1", "c2"],
            "dataset_name": ["a", "b"],
        }
    ).to_parquet(tr_ser, index=False)

    pd.DataFrame(
        {
            "csv_name": ["005_q.csv"],
            "class": ["cq"],
            "dataset_name": ["q"],
        }
    ).to_parquet(te_ser, index=False)

    pd.DataFrame(
        {
            "file": ["010_a", "020_b"],
            "CNN": [0.9, 0.3],
            "IForest": [0.2, 0.95],
        }
    ).to_csv(root / "train.csv", index=False)

    pd.DataFrame({"file": ["005_q"]}).to_csv(root / "test.csv", index=False)


def setup_tmp_repo(tmp_path: Path) -> Path:
    data = tmp_path / "splits" / "id"
    write_fixture_tree(data)
    return data


@pytest.fixture
def id_fixture_dir(tmp_path: Path) -> Path:
    return setup_tmp_repo(tmp_path)
