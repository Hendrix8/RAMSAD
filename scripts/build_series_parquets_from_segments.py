#!/usr/bin/env python3
"""Build train_series.parquet / test_series.parquet from segment parquets (csv_name + metadata)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def series_rows_in_segment_order(df: pd.DataFrame, name_col: str) -> pd.DataFrame:
    """One row per series, in **segment row order** (first occurrence); matches retrieval indexing."""

    def csv_name(x: str) -> str:
        s = str(x).strip()
        return s if s.lower().endswith(".csv") else f"{s}.csv"

    seen: set[str] = set()
    rows: list[dict] = []
    for i in range(len(df)):
        name = str(df.iloc[i][name_col]).strip()
        if name in seen:
            continue
        seen.add(name)
        r: dict = {"csv_name": csv_name(name)}
        for c in ("class", "dataset_name", "label"):
            if c in df.columns and c != name_col:
                r[c] = df.iloc[i][c]
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-segments", type=Path, required=True)
    p.add_argument("--test-segments", type=Path, required=True)
    p.add_argument("--out-train-series", type=Path, required=True)
    p.add_argument("--out-test-series", type=Path, required=True)
    p.add_argument("--name-col", type=str, default="dataset_name")
    args = p.parse_args()

    tr = pd.read_parquet(args.train_segments)
    te = pd.read_parquet(args.test_segments)
    series_rows_in_segment_order(tr, args.name_col).to_parquet(args.out_train_series, index=False)
    series_rows_in_segment_order(te, args.name_col).to_parquet(args.out_test_series, index=False)
    print(f"Wrote {args.out_train_series} ({tr[args.name_col].nunique()} series)")
    print(f"Wrote {args.out_test_series} ({te[args.name_col].nunique()} series)")


if __name__ == "__main__":
    main()
