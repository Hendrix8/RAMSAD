#!/usr/bin/env python3
"""
Generate RAMSAD ID segment/series parquets from train/test performance CSVs.

The performance CSVs provide the raw file names. The raw time-series CSVs are
expected to contain value and label columns, by default ``Data`` and ``Label``.
Each series is split into non-overlapping fixed-length chunks:

  - ``series``: float32 chunk padded with NaN
  - ``mask``: boolean validity mask, False on padding
  - ``label``: scalar label at the first valid point of the chunk
  - ``dataset_name``: raw file stem
  - ``seg_pos``: zero-based chunk position within the series
  - ``desc``: source-level dataset description, when known

Use ``--id`` to write only ``data/splits/id`` artifacts. Use ``--ood`` to first
refresh ID artifacts, then build leave-domain-out OOD splits via
``build_ood_splits_from_id.py``.
"""
from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"

SEGMENT_LEN = 8192

SOURCE_DESC: dict[str, str | None] = {
    "CATSv2": "The second version of the Controlled Anomalies Time Series dataset, with commands, external stimuli, and telemetry from a simulated complex dynamical system containing 200 injected anomalies.",
    "Daphnet": "Annotated readings from three acceleration sensors (hip and leg) on Parkinson’s patients experiencing freezing of gait (FoG) during walking tasks.",
    "Exathlon": "Based on real traces from a Spark cluster over 2.5 months; includes ground-truth labels for both root-cause and effect intervals of anomalies.",
    "IOPS": "A dataset of performance indicators reflecting web service scale and quality, as well as machine health status.",
    "LTDB": "A collection of seven long-duration ECG recordings (14–22 hours each) with manually reviewed beat annotations.",
    "MGAB": "Mackey-Glass time series where anomalies show chaotic behavior that is difficult to distinguish visually.",
    "MITDB": "Contains 48 half-hour excerpts of two-channel ambulatory ECG recordings from 47 subjects studied by the BIH Arrhythmia Laboratory (1975–1979).",
    "MSL": "Collected from the Curiosity Rover mission on Mars.",
    "NAB": "Labeled real-world and artificial time series, including AWS server metrics, online ad click rates, real-time traffic data, and Twitter mentions of publicly traded companies.",
    "NEK": "Collected from real production network equipment.",
    "OPPORTUNITY": None,
    "Power": "Power consumption data for a Dutch research facility over the full year of 1997.",
    "SED": "Simulated engine disk data from the NASA Rotary Dynamics Laboratory, representing disk revolutions across several runs at 3K rpm.",
    "SMAP": "Real spacecraft telemetry with anomalies from the Soil Moisture Active Passive satellite; each series includes one sensor measurement feature and binary-encoded command features.",
    "SMD": "A 5-week dataset from a large Internet company, containing three groups of entities from 28 different machines.",
    "SVDB": "Includes 78 half-hour ECG recordings selected to supplement supraventricular arrhythmia examples in the MIT-BIH Arrhythmia Database.",
    "SWaT": "A secure water treatment dataset collected from 51 sensors and actuators; anomalies represent abnormal behavior under attack scenarios.",
    "Stock": "A stock trading traces dataset with one million transaction records collected during one trading day.",
    "TAO": None,
    "TODS": "A synthetic dataset including global, contextual, shapelet, seasonal, and trend anomalies.",
    "UCR": "A collection of univariate time series across multiple domains (e.g., air temperature, arterial blood pressure, astronomy, ECG), where most anomalies are artificially introduced.",
    "WSD": "A web service dataset containing real-world KPIs collected from large Internet companies.",
    "YAHOO": "A Yahoo Labs dataset containing both real and synthetic time series based on production traffic to Yahoo systems.",
}

SOURCE_RE = re.compile(r"^\d+_([^_]+)_id_")


def series_rows_in_segment_order(df: pd.DataFrame, name_col: str) -> pd.DataFrame:
    def csv_name(x: str) -> str:
        value = str(x).strip()
        return value if value.lower().endswith(".csv") else f"{value}.csv"

    seen: set[str] = set()
    rows: list[dict] = []
    for i in range(len(df)):
        name = str(df.iloc[i][name_col]).strip()
        if name in seen:
            continue
        seen.add(name)
        row: dict = {"csv_name": csv_name(name)}
        for col in ("class", "dataset_name", "label"):
            if col in df.columns and col != name_col:
                row[col] = df.iloc[i][col]
        rows.append(row)
    return pd.DataFrame(rows)


def csv_stem(name: str) -> str:
    return Path(str(name).strip()).stem


def source_from_name(name: str) -> str | None:
    match = SOURCE_RE.match(csv_stem(name))
    return match.group(1) if match else None


def load_descriptions(path: Path | None) -> dict[str, str | None]:
    if path is None or not path.is_file():
        return SOURCE_DESC
    df = pd.read_csv(path)
    required = {"source", "description"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{path}: missing required columns {sorted(missing)}")
    out: dict[str, str | None] = {}
    for row in df.itertuples(index=False):
        source = str(getattr(row, "source")).strip()
        raw_desc = getattr(row, "description")
        desc = None if pd.isna(raw_desc) else str(raw_desc)
        out[source] = desc
    return {**SOURCE_DESC, **out}


def desc_from_name(name: str, descriptions: dict[str, str | None]) -> str | None:
    source = source_from_name(name)
    return descriptions.get(source) if source else None


def candidate_raw_dirs(split: str, perf_csv: Path, raw_root: Path | None) -> list[Path]:
    split_title = "Train" if split == "train" else "Test"
    roots: list[Path] = []
    if raw_root is not None:
        roots.extend([raw_root / split_title, raw_root / split, raw_root])

    base = perf_csv.resolve().parent
    roots.extend(
        [
            base / split_title,
            base / split,
            base.parent / split_title,
            base.parent / split,
            base.parent / "ad" / split_title,
            base.parent / "ad" / split,
        ]
    )

    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        root = root.expanduser()
        if root not in seen:
            seen.add(root)
            out.append(root)
    return out


def resolve_raw_csv(file_name: str, raw_dirs: Iterable[Path]) -> Path:
    file_name = str(file_name).strip()
    direct = Path(file_name).expanduser()
    if direct.is_file():
        return direct.resolve()
    for raw_dir in raw_dirs:
        path = raw_dir / file_name
        if path.is_file():
            return path.resolve()
    searched = "\n  ".join(str(x) for x in raw_dirs)
    raise FileNotFoundError(f"Could not find raw CSV {file_name!r}. Searched:\n  {searched}")


def pick_column(df: pd.DataFrame, requested: str | None, candidates: tuple[str, ...], kind: str, path: Path) -> str:
    if requested:
        if requested not in df.columns:
            raise KeyError(f"{path}: requested {kind} column {requested!r} not found in {list(df.columns)}")
        return requested
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"{path}: cannot infer {kind} column from {list(df.columns)}")


def build_segments_for_raw_csv(
    raw_csv: Path,
    dataset_name: str,
    segment_len: int,
    value_col: str | None,
    label_col: str | None,
    descriptions: dict[str, str | None],
) -> list[dict]:
    df = pd.read_csv(raw_csv)
    value = pick_column(df, value_col, ("Data", "value", "series", "timestamp_value"), "value", raw_csv)
    label = pick_column(df, label_col, ("Label", "label", "is_anomaly", "anomaly"), "label", raw_csv)

    values = pd.to_numeric(df[value], errors="coerce").to_numpy(dtype=np.float32)
    labels = pd.to_numeric(df[label], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    if len(values) != len(labels):
        raise ValueError(f"{raw_csv}: value and label columns have different lengths")
    if len(values) == 0:
        raise ValueError(f"{raw_csv}: empty raw series")

    rows: list[dict] = []
    n_segments = max(1, int(math.ceil(len(values) / segment_len)))
    desc = desc_from_name(dataset_name, descriptions)
    for seg_pos in range(n_segments):
        start = seg_pos * segment_len
        end = min(start + segment_len, len(values))
        valid = end - start

        series = np.full(segment_len, np.nan, dtype=np.float32)
        mask = np.zeros(segment_len, dtype=bool)
        series[:valid] = values[start:end]
        mask[:valid] = True

        rows.append(
            {
                "series": series,
                "mask": mask,
                "label": int(labels[start]),
                "dataset_name": dataset_name,
                "seg_pos": int(seg_pos),
                "desc": desc,
            }
        )
    return rows


def build_split_segments(
    perf_csv: Path,
    split: str,
    out_segments: Path,
    out_series: Path,
    raw_root: Path | None,
    file_col: str,
    segment_len: int,
    value_col: str | None,
    label_col: str | None,
    descriptions: dict[str, str | None],
    force: bool,
) -> None:
    if out_segments.is_file() and out_series.is_file() and not force:
        print(f"[skip] {split}: existing {out_segments} and {out_series}")
        return

    perf = pd.read_csv(perf_csv)
    if file_col not in perf.columns:
        raise KeyError(f"{perf_csv}: expected file column {file_col!r}")

    raw_dirs = candidate_raw_dirs(split, perf_csv, raw_root)
    rows: list[dict] = []
    for i, file_name in enumerate(perf[file_col].astype(str).tolist(), start=1):
        raw_csv = resolve_raw_csv(file_name, raw_dirs)
        dataset_name = csv_stem(file_name)
        rows.extend(build_segments_for_raw_csv(raw_csv, dataset_name, segment_len, value_col, label_col, descriptions))
        if i % 100 == 0:
            print(f"[{split}] processed {i}/{len(perf)} raw CSVs")

    seg_df = pd.DataFrame(rows)
    out_segments.parent.mkdir(parents=True, exist_ok=True)
    seg_df.to_parquet(out_segments, index=False)
    series_rows_in_segment_order(seg_df, "dataset_name").to_parquet(out_series, index=False)
    print(f"[{split}] wrote {out_segments} ({len(seg_df)} segments, {seg_df['dataset_name'].nunique()} series)")
    print(f"[{split}] wrote {out_series}")


def run_build_ood(id_root: Path, ood_root: Path, domains: str, train_csv: Path, test_csv: Path) -> None:
    cmd = [
        sys.executable,
        str(_SCRIPTS_DIR / "build_ood_splits_from_id.py"),
        "--id-root",
        str(id_root),
        "--out-root",
        str(ood_root),
        "--domains",
        domains,
        "--train-csv",
        str(train_csv),
        "--test-csv",
        str(test_csv),
    ]
    print("[ood] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--id", action="store_true", help="Generate ID segment/series parquets only")
    mode.add_argument("--ood", action="store_true", help="Generate ID artifacts, then OOD leave-domain-out artifacts")
    p.add_argument("--train-csv", type=Path, default=Path("data/raw/VUS/train.csv"))
    p.add_argument("--test-csv", type=Path, default=Path("data/raw/VUS/test.csv"))
    p.add_argument("--data-root", type=Path, default=Path("data/splits"))
    p.add_argument("--raw-root", type=Path, default=None, help="Directory containing Train/ and Test/ raw CSV folders")
    p.add_argument("--file-col", default="file")
    p.add_argument("--value-col", default=None, help="Raw CSV value column; inferred by default")
    p.add_argument("--label-col", default=None, help="Raw CSV label column; inferred by default")
    p.add_argument(
        "--descriptions-csv",
        type=Path,
        default=None,
        help="CSV with source,description columns; defaults to <data-root>/descriptions.csv when present",
    )
    p.add_argument("--segment-len", type=int, default=SEGMENT_LEN)
    p.add_argument("--domains", default="auto", help="OOD domains for --ood; comma list or auto")
    p.add_argument("--force", action="store_true", help="Regenerate even when outputs already exist")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    id_root = data_root / "id"

    train_csv = args.train_csv.resolve()
    test_csv = args.test_csv.resolve()
    for path in (train_csv, test_csv):
        if not path.is_file():
            raise SystemExit(f"Missing performance CSV: {path}")

    id_root.mkdir(parents=True, exist_ok=True)
    descriptions_csv = args.descriptions_csv
    if descriptions_csv is None:
        candidate = data_root / "descriptions.csv"
        descriptions_csv = candidate if candidate.is_file() else None
    descriptions = load_descriptions(descriptions_csv)
    if descriptions_csv is not None:
        print(f"[desc] using {descriptions_csv}")

    id_train_csv = id_root / "train.csv"
    id_test_csv = id_root / "test.csv"
    # User requested not to resave train.csv and test.csv:
    # if train_csv != id_train_csv.resolve():
    #     shutil.copyfile(train_csv, id_train_csv)
    # if test_csv != id_test_csv.resolve():
    #     shutil.copyfile(test_csv, id_test_csv)

    build_split_segments(
        perf_csv=train_csv,
        split="train",
        out_segments=id_root / "train_segments.parquet",
        out_series=id_root / "train_series.parquet",
        raw_root=args.raw_root,
        file_col=args.file_col,
        segment_len=args.segment_len,
        value_col=args.value_col,
        label_col=args.label_col,
        descriptions=descriptions,
        force=args.force,
    )
    build_split_segments(
        perf_csv=test_csv,
        split="test",
        out_segments=id_root / "test_segments.parquet",
        out_series=id_root / "test_series.parquet",
        raw_root=args.raw_root,
        file_col=args.file_col,
        segment_len=args.segment_len,
        value_col=args.value_col,
        label_col=args.label_col,
        descriptions=descriptions,
        force=args.force,
    )

    if args.ood:
        run_build_ood(id_root=id_root, ood_root=data_root / "ood", domains=args.domains, train_csv=train_csv, test_csv=test_csv)


if __name__ == "__main__":
    main()
