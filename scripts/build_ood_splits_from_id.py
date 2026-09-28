#!/usr/bin/env python3
"""
Build OOD Hydra layouts from a bundled **ID** split only (train/test segments + perf CSVs).

The ID train and test series are pooled (870 TSB-AD series), then for each domain D present
in the data, writes (leave-one-domain-out, as in the paper):
  - **Train** segments & train perf (knowledge base): all series whose domain ≠ D
  - **Test** segments & test perf (queries): all series whose domain = D
Across the 9 domains every series is queried exactly once (870 queries in total).

Output mirrors ``data/splits/ood/``:
  ood/OOD_<domain>/{train,test}_segments_OOD_<domain>.parquet
  ood/OOD_<domain>/{train,test}_series_OOD_<domain>.parquet
  ood/Datasets/OOD/{train,test}_perf_OOD_<domain>.csv

Domain is parsed from ``dataset_name`` / perf ``file`` stem using ``_id_<n>_<Domain>_tr_``
(e.g. ``..._Medical_tr_...`` → ``medical``).

Example:
  python scripts/build_ood_splits_from_id.py \\
    --id-root data/splits/id \\
    --out-root data/splits/ood
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from build_series_parquets_from_segments import series_rows_in_segment_order

DOMAIN_RE = re.compile(r"_id_\d+_([^_]+)_tr_")


def domain_slug(dataset_name_or_file_stem: str) -> str:
    m = DOMAIN_RE.search(str(dataset_name_or_file_stem))
    if not m:
        raise ValueError(f"Cannot parse domain from: {dataset_name_or_file_stem!r}")
    return m.group(1).lower()


def add_domain_column_segments(df: pd.DataFrame, name_col: str) -> pd.DataFrame:
    out = df.copy()
    out["_domain"] = out[name_col].astype(str).map(domain_slug)
    return out


def add_domain_column_perf(df: pd.DataFrame, file_col: str) -> pd.DataFrame:
    out = df.copy()
    keys = out[file_col].astype(str).str.strip().str.removesuffix(".csv")
    out["_domain"] = keys.map(domain_slug)
    return out


def sort_segment_rows(df: pd.DataFrame, name_col: str, seg_pos_col: str) -> pd.DataFrame:
    """Stable order so each series block starts with seg_pos == 0 (required by retrieval)."""
    if df.empty:
        return df
    out = df.sort_values([name_col, seg_pos_col], kind="mergesort").reset_index(drop=True)
    if int(out[seg_pos_col].iloc[0]) != 0:
        raise ValueError(
            "After filtering, first segment row does not have seg_pos==0; "
            "check data or series ordering."
        )
    return out


def drop_helper_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in df.columns if c.startswith("_")])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--id-root", type=Path, required=True, help="Directory with train_* and test_* (segments + csv)")
    p.add_argument("--out-root", type=Path, required=True, help="Usually .../data/splits/ood")
    p.add_argument(
        "--domains",
        type=str,
        default="auto",
        help="Comma-separated slugs (e.g. sensor,medical) or 'auto' for all domains in ID data",
    )
    p.add_argument("--name-col", type=str, default="dataset_name")
    p.add_argument("--seg-pos-col", type=str, default="seg_pos")
    p.add_argument("--perf-file-column", type=str, default="file")
    p.add_argument("--train-csv", type=Path, default=None, help="Path to ID train.csv")
    p.add_argument("--test-csv", type=Path, default=None, help="Path to ID test.csv")
    args = p.parse_args()

    id_root = args.id_root.resolve()
    out_root = args.out_root.resolve()
    name_col = args.name_col
    seg_col = args.seg_pos_col
    fcol = args.perf_file_column

    train_seg_p = id_root / "train_segments.parquet"
    test_seg_p = id_root / "test_segments.parquet"
    train_csv_p = args.train_csv.resolve() if args.train_csv else id_root / "train.csv"
    test_csv_p = args.test_csv.resolve() if args.test_csv else id_root / "test.csv"
    for path in (train_seg_p, test_seg_p, train_csv_p, test_csv_p):
        if not path.is_file():
            raise SystemExit(f"Missing ID artifact: {path}")

    tr_seg = pd.read_parquet(train_seg_p)
    te_seg = pd.read_parquet(test_seg_p)
    tr_csv = pd.read_csv(train_csv_p)
    test_csv = pd.read_csv(test_csv_p)

    if name_col not in tr_seg.columns or name_col not in te_seg.columns:
        raise SystemExit(f"Expected column {name_col!r} in segment parquets")
    if fcol not in tr_csv.columns or fcol not in test_csv.columns:
        raise SystemExit(f"Expected column {fcol!r} in perf CSVs")

    tr_seg = add_domain_column_segments(tr_seg, name_col)
    te_seg = add_domain_column_segments(te_seg, name_col)
    tr_csv = add_domain_column_perf(tr_csv, fcol)
    test_csv = add_domain_column_perf(test_csv, fcol)

    all_domains = sorted(set(tr_seg["_domain"]) | set(te_seg["_domain"]))
    if args.domains.strip().lower() == "auto":
        domains = all_domains
    else:
        want = {x.strip().lower() for x in args.domains.split(",") if x.strip()}
        unknown = want - set(all_domains)
        if unknown:
            raise SystemExit(f"Unknown domains (not in ID data): {sorted(unknown)}")
        domains = sorted(want)

    ood_ds = out_root / "Datasets" / "OOD"
    ood_ds.mkdir(parents=True, exist_ok=True)

    # Leave-one-domain-out over the pooled benchmark: ID train + ID test series.
    all_seg = pd.concat([tr_seg, te_seg], ignore_index=True)
    all_perf = pd.concat([tr_csv, test_csv], ignore_index=True)
    for d in domains:
        tr_rows = all_seg[all_seg["_domain"] != d]
        te_rows = all_seg[all_seg["_domain"] == d]
        tr_perf = all_perf[all_perf["_domain"] != d]
        te_perf = all_perf[all_perf["_domain"] == d]

        if te_rows.empty or te_perf.empty:
            print(f"[skip] OOD_{d}: empty test (segments={len(te_rows)}, perf={len(te_perf)})")
            continue
        if tr_rows.empty or tr_perf.empty:
            print(f"[skip] OOD_{d}: empty train after holdout")
            continue

        tr_rows = sort_segment_rows(tr_rows, name_col, seg_col)
        te_rows = sort_segment_rows(te_rows, name_col, seg_col)

        sub = out_root / f"OOD_{d}"
        sub.mkdir(parents=True, exist_ok=True)

        tsn = f"_OOD_{d}"
        ts_seg = sub / f"train_segments{tsn}.parquet"
        tst_seg = sub / f"test_segments{tsn}.parquet"
        ts_ser = sub / f"train_series{tsn}.parquet"
        tst_ser = sub / f"test_series{tsn}.parquet"

        drop_helper_columns(tr_rows).to_parquet(ts_seg, index=False)
        drop_helper_columns(te_rows).to_parquet(tst_seg, index=False)

        series_rows_in_segment_order(tr_rows, name_col).to_parquet(ts_ser, index=False)
        series_rows_in_segment_order(te_rows, name_col).to_parquet(tst_ser, index=False)

        drop_helper_columns(tr_perf).to_csv(ood_ds / f"train_perf{tsn}.csv", index=False)
        drop_helper_columns(te_perf).to_csv(ood_ds / f"test_perf{tsn}.csv", index=False)

        print(
            f"[ood/{d}] train seg {len(tr_rows)} | test seg {len(te_rows)} | "
            f"train perf {len(tr_perf)} | test perf {len(te_perf)}"
        )

    print(f"Done. OOD root: {out_root}")


if __name__ == "__main__":
    main()
