from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from ramsad.alignment_projector import resolve_alignment_run_dir
from ramsad.retrieve import run_retrieval_to_csv
from ramsad.select import run_selection


def test_resolve_alignment_run_dir_direct(tmp_path: Path) -> None:
    direct = tmp_path / "run-direct"
    (direct / "checkpoints").mkdir(parents=True)
    ckpt = direct / "checkpoints" / "best_model.pt"
    ckpt.write_bytes(b"x")
    assert resolve_alignment_run_dir(direct) == direct.resolve()


def test_resolve_alignment_run_dir_newest_child(tmp_path: Path) -> None:
    base = tmp_path / "outputs" / "alignment"
    old = base / "run-old"
    new = base / "run-new"
    (old / "checkpoints").mkdir(parents=True)
    (new / "checkpoints").mkdir(parents=True)
    (old / "checkpoints" / "best_model.pt").write_bytes(b"o")
    (new / "checkpoints" / "best_model.pt").write_bytes(b"n")
    os.utime(old / "checkpoints" / "best_model.pt", (1, 100))
    os.utime(new / "checkpoints" / "best_model.pt", (1, 200))
    assert resolve_alignment_run_dir(base) == new.resolve()


def test_retrieve_and_select_roundtrip(id_fixture_dir: Path) -> None:
    data = id_fixture_dir
    out_csv = data.parent.parent / "retrieval.csv"
    run_retrieval_to_csv(
        train_segments_parquet=data / "train_segments.parquet",
        test_segments_parquet=data / "test_segments.parquet",
        train_series_parquet=data / "train_series.parquet",
        test_series_parquet=data / "test_series.parquet",
        out_csv=out_csv,
        top_k=2,
        batch_size=8,
        aggregation_strategy="mean",
        seg_pos_col="seg_pos",
        embed_column="amazon_chronos-2",
    )
    df = pd.read_csv(out_csv)
    assert len(df) == 1
    assert "top-1" in df.columns

    sel = data.parent.parent / "model_selection.csv"
    run_selection(
        train_perf_path=data / "train.csv",
        test_perf_path=data / "test.csv",
        retrieval_csv=out_csv,
        out_csv=sel,
        k=2,
        perf_col="file",
        valid_threshold=-0.5,
        start_id=1,
        end_id=999,
        allowlist=["CNN", "IForest"],
        verbose=False,
    )
    ms = pd.read_csv(sel)
    assert len(ms) == 1
    assert ms.iloc[0]["selected_model"] in ("CNN", "IForest")


def test_pick_top_n_mean_vus_orders_and_breaks_ties() -> None:
    from ramsad.select import pick_top_n_mean_vus

    avg = {"B": 0.5, "A": 0.5, "C": 0.9, "D": -1.0}
    assert pick_top_n_mean_vus(avg, 1, valid_threshold=-0.5) == ["C"]
    assert pick_top_n_mean_vus(avg, 3, valid_threshold=-0.5) == ["C", "A", "B"]
    assert pick_top_n_mean_vus(avg, 10, valid_threshold=-0.5) == ["C", "A", "B"]
