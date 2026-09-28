from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

DEFAULT_MODEL_ORDER = [
    "Sub-IForest",
    "IForest",
    "Sub-LOF",
    "LOF",
    "POLY",
    "MatrixProfile",
    "KShapeAD",
    "SAND",
    "Series2Graph",
    "SR",
    "Sub-PCA",
    "Sub-HBOS",
    "Sub-OCSVM",
    "Sub-MCD",
    "Sub-KNN",
    "KMeansAD",
    "AutoEncoder",
    "CNN",
    "LSTMAD",
    "TranAD",
    "AnomalyTransformer",
    "OmniAnomaly",
    "USAD",
    "Donut",
    "TimesNet",
    "FITS",
    "OFA",
    "Lag-Llama",
    "Chronos",
    "TimesFM",
    "MOMENT (ZS)",
    "MOMENT (FT)",
]


def normalize_ts_key(name: str) -> str:
    return str(name).strip().removesuffix(".csv")


def load_perf_csv(path: Path, perf_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[perf_col] = df[perf_col].astype(str).str.strip().str.removesuffix(".csv")
    return df


def load_allowed_keys(perf_file: Path, perf_col: str) -> set[str]:
    df = pd.read_csv(perf_file)
    return set(df[perf_col].astype(str).str.strip().str.removesuffix(".csv"))


def numeric_model_columns(df: pd.DataFrame, perf_col: str) -> list[str]:
    skip = {perf_col}
    out: list[str] = []
    for c in df.columns:
        if c in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
        else:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > len(df) * 0.5:
                out.append(c)
    return out


def pick_mean_vus_argmax(
    avg_perf: dict[str, float],
    *,
    valid_threshold: float,
) -> tuple[str, str]:
    """Match Chronos_avgVUS with tie-break flags off."""
    valid = {m: v for m, v in avg_perf.items() if v > valid_threshold}
    if not valid:
        m = max(avg_perf, key=avg_perf.get)
        return m, "fallback max avg perf (no valid scores)"
    best_score = max(valid.values())
    tied = sorted([m for m, v in valid.items() if v == best_score])
    m = tied[0]
    if len(tied) > 1:
        return m, f"highest mean VUS ({best_score:.4f}); tie-break lexicographic among {len(tied)}"
    return m, f"highest mean VUS ({best_score:.4f})"


def pick_top_n_mean_vus(
    avg_perf: dict[str, float],
    n: int,
    *,
    valid_threshold: float,
) -> list[str]:
    """Top-``n`` detectors by neighbor mean VUS (ties broken lexicographically)."""
    valid = {m: v for m, v in avg_perf.items() if v > valid_threshold} or avg_perf
    ranked = sorted(valid.items(), key=lambda mv: (-mv[1], mv[0]))
    return [m for m, _ in ranked[:n]]


def compute_mean_vus_per_model(
    retrieved_df: pd.DataFrame,
    models: list[str],
    perf_col: str,
    neighbor_weights: np.ndarray | None = None,
) -> dict[str, float]:
    if neighbor_weights is None:
        w_row = np.ones(len(retrieved_df), dtype=np.float64)
    else:
        w_row = np.asarray(neighbor_weights, dtype=np.float64)
        if len(w_row) != len(retrieved_df):
            raise ValueError("neighbor_weights length mismatch")

    perf_sums = {m: 0.0 for m in models}
    valid_weight = {m: 0.0 for m in models}

    for wi, (_, row) in zip(w_row, retrieved_df.iterrows()):
        row_values = row[models].apply(pd.to_numeric, errors="coerce")
        for model in models:
            val = row_values[model]
            if pd.notna(val):
                perf_sums[model] += float(wi) * float(val)
                valid_weight[model] += float(wi)

    avg_perf: dict[str, float] = {}
    for m in models:
        if valid_weight[m] > 0:
            avg_perf[m] = perf_sums[m] / valid_weight[m]
        else:
            avg_perf[m] = -1.0
    return avg_perf


def retrieve_neighbor_rows(
    df_perf: pd.DataFrame,
    ts_name: str,
    k: int,
    rag_df: pd.DataFrame,
    perf_col: str,
) -> pd.DataFrame | None:
    want = normalize_ts_key(ts_name)
    norm_q = rag_df["query"].astype(str).map(normalize_ts_key)
    row_df = rag_df[norm_q == want]
    if row_df.empty:
        return None
    row = row_df.iloc[0]
    names: list[str] = []
    for i in range(1, k + 1):
        tcol = f"top-{i}"
        if tcol not in row.index:
            break
        raw = row[tcol]
        if pd.isna(raw) or str(raw).strip() == "":
            continue
        names.append(normalize_ts_key(raw))
    if not names:
        return None
    perf_indexed = df_perf.set_index(perf_col)
    rows_list = []
    for name in names:
        if name not in perf_indexed.index:
            continue
        rows_list.append(perf_indexed.loc[name])
    if not rows_list:
        return None
    return pd.DataFrame(rows_list).reset_index(drop=True)


def reorder_results_csv(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)

    def ts_num(x: str) -> float:
        m = re.match(r"(\d+)_", str(x))
        return int(m.group(1)) if m else float("inf")

    df["_tsn"] = df["time_series"].map(ts_num)
    df = df.sort_values("_tsn").drop(columns=["_tsn"])
    df.to_csv(csv_path, index=False)


def candidate_models_for_pool(
    train_df: pd.DataFrame,
    allowlist: list[str] | None,
    perf_col: str,
) -> list[str]:
    numeric = numeric_model_columns(train_df, perf_col)
    if allowlist:
        allow = [m for m in allowlist if m in train_df.columns]
        pool = [m for m in allow if m in numeric]
        return pool if pool else [m for m in DEFAULT_MODEL_ORDER if m in numeric]
    return [m for m in DEFAULT_MODEL_ORDER if m in numeric] or sorted(numeric)


def run_selection(
    *,
    train_perf_path: Path,
    test_perf_path: Path,
    retrieval_csv: Path,
    out_csv: Path,
    k: int,
    perf_col: str,
    valid_threshold: float,
    start_id: int,
    end_id: int,
    allowlist: list[str] | None,
    n: int = 1,
    verbose: bool = True,
) -> Path:
    df_train = load_perf_csv(train_perf_path, perf_col)
    allowed = load_allowed_keys(test_perf_path, perf_col)
    rag_df = pd.read_csv(retrieval_csv)
    models = candidate_models_for_pool(df_train, allowlist, perf_col)
    if not models:
        raise RuntimeError("No model columns found in train perf")

    out_csv = out_csv.expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if n < 1:
        raise ValueError(f"select.n must be >= 1, got {n}")
    out_csv.write_text("time_series,selected_model,selected_models,rationale\n", encoding="utf-8")

    def ts_id(name: str) -> int:
        return int(str(name).split("_")[0])

    for ts_name in sorted(allowed):
        if ts_id(ts_name) < start_id or ts_id(ts_name) > end_id:
            continue
        retrieved = retrieve_neighbor_rows(df_train, ts_name, k, rag_df, perf_col)
        if retrieved is None:
            continue
        use_models = [m for m in models if m in retrieved.columns]
        if not use_models:
            continue
        avg = compute_mean_vus_per_model(retrieved, use_models, perf_col, None)
        best, detail = pick_mean_vus_argmax(avg, valid_threshold=valid_threshold)
        top_n = pick_top_n_mean_vus(avg, n, valid_threshold=valid_threshold)
        rationale = f"neighbor mean VUS argmax | K={k} | N={n} | {detail}"
        line = f"{ts_name},{best},\"{';'.join(top_n)}\",\"{rationale}\"\n"
        with open(out_csv, "a", encoding="utf-8") as f:
            f.write(line)
        if verbose:
            print(f"[select] {ts_name} -> {best}")

    reorder_results_csv(out_csv)
    if verbose:
        print(f"[select] wrote {out_csv}")
    return out_csv


def run_select(cfg: DictConfig, output_dir: Path, retrieval_path: Path | None = None) -> Path:
    from hydra.utils import to_absolute_path

    d = cfg.data
    rc_override = cfg.select.get("retrieval_csv")
    if (
        rc_override is not None
        and str(rc_override).strip() not in ("", "~", "null", "None")
    ):
        rpath = Path(to_absolute_path(str(rc_override)))
    else:
        rpath = retrieval_path or (output_dir / str(cfg.retrieve.output_csv_name))
    allowlist = None
    if cfg.select.get("model_allowlist") is not None:
        ml = cfg.select.model_allowlist
        allowlist = list(ml) if not isinstance(ml, (str, int, float)) else None

    out = output_dir / str(cfg.select.output_csv_name)
    return run_selection(
        train_perf_path=Path(to_absolute_path(str(d.train_perf))),
        test_perf_path=Path(to_absolute_path(str(d.test_perf))),
        retrieval_csv=rpath,
        out_csv=out,
        k=int(cfg.select.k),
        perf_col=str(cfg.select.perf_file_column),
        valid_threshold=float(cfg.select.valid_vus_threshold),
        start_id=int(cfg.select.start_id),
        end_id=int(cfg.select.end_id),
        allowlist=allowlist,
        n=int(cfg.select.get("n", 1)),
        verbose=bool(cfg.select.get("verbose", True)),
    )
