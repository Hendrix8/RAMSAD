from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm


def _normalize_refs(refs) -> list[int]:
    if refs is None:
        return []
    if isinstance(refs, np.ndarray):
        refs = refs.tolist()
    if not isinstance(refs, (list, tuple)):
        refs = [refs]
    out: list[int] = []
    for x in refs:
        if isinstance(x, np.generic):
            x = x.item()
        out.append(int(float(x)))
    return out


def _normalize_float_list(x) -> list[float]:
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if not isinstance(x, (list, tuple)):
        x = [x]
    out: list[float] = []
    for v in x:
        if isinstance(v, np.generic):
            v = v.item()
        out.append(float(v))
    return out


def _pick_existing_col(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find {label}. Tried: {candidates}")


def infer_series_ids_from_seg_pos(seg_pos_values: np.ndarray) -> np.ndarray:
    if len(seg_pos_values) == 0:
        return np.array([], dtype=np.int64)
    out = np.empty(len(seg_pos_values), dtype=np.int64)
    series_id = -1
    for i, sp in enumerate(seg_pos_values):
        if int(sp) == 0:
            series_id += 1
        if series_id < 0:
            raise ValueError("First seg_pos is not 0; cannot infer grouping reliably.")
        out[i] = series_id
    return out


def group_segment_indices_by_series(series_ids: np.ndarray) -> list[np.ndarray]:
    n_series = int(series_ids.max()) + 1 if len(series_ids) else 0
    groups: list[list[int]] = [[] for _ in range(n_series)]
    for idx, sid in enumerate(series_ids):
        groups[int(sid)].append(idx)
    return [np.array(g, dtype=np.int64) for g in groups]


def aggregate_query_series_topk(
    query_seg_indices: np.ndarray,
    topk_vals: list,
    topk_dists: list,
    train_seg_to_train_series: np.ndarray,
    k: int,
    aggregation_strategy: str = "frequency_penalty",
) -> tuple[list[int], list[float]]:
    dist_all: dict[int, list[float]] = defaultdict(list)
    freq_segments: dict[int, int] = defaultdict(int)
    m = int(len(query_seg_indices))

    for seg_idx in query_seg_indices:
        refs = _normalize_refs(topk_vals[int(seg_idx)])
        dists = _normalize_float_list(topk_dists[int(seg_idx)])
        if len(refs) < k or len(dists) < k:
            continue

        appeared_this_segment = set()
        for r in range(k):
            train_seg_idx = int(refs[r])
            if train_seg_idx < 0 or train_seg_idx >= len(train_seg_to_train_series):
                continue
            train_series_id = int(train_seg_to_train_series[train_seg_idx])
            dist = float(dists[r])
            if not np.isfinite(dist):
                continue
            dist_all[train_series_id].append(dist)
            appeared_this_segment.add(train_series_id)

        for train_series_id in appeared_this_segment:
            freq_segments[train_series_id] += 1

    if not dist_all:
        return [], []

    scores: dict[int, float] = {}
    for train_series_id, vals in dist_all.items():
        if aggregation_strategy == "frequency_penalty":
            base = float(np.mean(vals))
            freq = max(1, int(freq_segments.get(train_series_id, 0)))
            scores[train_series_id] = (m / freq) * base
        elif aggregation_strategy == "mean":
            scores[train_series_id] = float(np.mean(vals))
        elif aggregation_strategy == "min":
            scores[train_series_id] = float(np.min(vals))
        else:
            raise ValueError(f"Unknown aggregation strategy: {aggregation_strategy}")

    ranked = sorted(scores.items(), key=lambda x: x[1])
    top_pairs = ranked[:k]
    ids = [int(t) for t, _ in top_pairs]
    dists_out = [float(s) for _, s in top_pairs]
    return ids, dists_out


def compute_segment_topk_cosine(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_vec_col: str,
    train_vec_col: str,
    k: int,
    batch_size: int,
) -> tuple[list[list[int]], list[list[float]]]:
    if k <= 0:
        raise ValueError("top-k must be > 0")
    if len(train_df) == 0:
        raise ValueError("Empty train dataframe")
    n_train = len(train_df)
    if n_train < k:
        raise ValueError(f"Train set smaller than top-k ({n_train} < {k})")

    queries = np.stack([np.asarray(v, dtype=np.float32) for v in test_df[test_vec_col].tolist()])
    db = np.stack([np.asarray(v, dtype=np.float32) for v in train_df[train_vec_col].tolist()])
    qn = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-9)
    dn = db / (np.linalg.norm(db, axis=1, keepdims=True) + 1e-9)
    k_eff = min(k, db.shape[0])

    topk_indices_all: list[list[int]] = []
    topk_dists_all: list[list[float]] = []
    for s in tqdm(range(0, qn.shape[0], batch_size), desc="Segment retrieval (cosine)"):
        e = min(qn.shape[0], s + batch_size)
        sims = qn[s:e] @ dn.T
        idx = np.argpartition(-sims, k_eff - 1, axis=1)[:, :k_eff]
        rows = np.arange(idx.shape[0])[:, None]
        ord_idx = np.argsort(-sims[rows, idx], axis=1)
        idx = idx[rows, ord_idx]
        dists = 1.0 - sims[rows, idx]
        topk_indices_all.extend(idx.astype(np.int64).tolist())
        topk_dists_all.extend(dists.astype(np.float64).tolist())
    return topk_indices_all, topk_dists_all


def resolve_chronos_column(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    embed_column: str | None,
) -> tuple[str, str]:
    if embed_column and str(embed_column).strip():
        c = str(embed_column).strip()
        return c, c
    candidates = ["chronos_amazon-2", "amazon_chronos-2"]
    return (
        _pick_existing_col(test_df, candidates, "chronos embedding column in test"),
        _pick_existing_col(train_df, candidates, "chronos embedding column in train"),
    )


def run_retrieval_to_csv(
    *,
    train_segments_parquet: Path,
    test_segments_parquet: Path,
    train_series_parquet: Path,
    test_series_parquet: Path,
    out_csv: Path,
    top_k: int,
    batch_size: int,
    aggregation_strategy: str,
    seg_pos_col: str,
    embed_column: str | None,
    retrieval_mode: str = "chronos",
    alignment_run_dir: Path | None = None,
    alignment_emb_column: str | None = None,
    alignment_source_column: str | None = None,
    alignment_device: str = "cuda",
    skip_alignment_if_present: bool = True,
) -> Path:
    train_seg_df = pd.read_parquet(train_segments_parquet)
    test_seg_df = pd.read_parquet(test_segments_parquet)
    train_series_df = pd.read_parquet(train_series_parquet)
    test_series_df = pd.read_parquet(test_series_parquet)

    if seg_pos_col not in test_seg_df.columns or seg_pos_col not in train_seg_df.columns:
        raise KeyError(f"'{seg_pos_col}' required in train/test segment parquets")
    if "csv_name" not in test_series_df.columns or "csv_name" not in train_series_df.columns:
        raise KeyError("train_series and test_series parquets must include 'csv_name'")

    mode = (retrieval_mode or "chronos").strip().lower()
    if mode == "alignment":
        if alignment_run_dir is None:
            raise ValueError("retrieval_mode=alignment requires alignment_run_dir")
        if not alignment_emb_column:
            raise ValueError("retrieval_mode=alignment requires alignment_emb_column (target column name)")
        from ramsad.alignment_projector import load_alignment_projector, project_column_into_df

        src_col_t, src_col_tr = resolve_chronos_column(
            test_seg_df,
            train_seg_df,
            alignment_source_column or embed_column,
        )
        if src_col_t != src_col_tr:
            raise ValueError(
                f"Train/test source embedding columns differ ({src_col_tr!r} vs {src_col_t!r}); use one Chronos column."
            )
        source_col = src_col_t

        import torch

        dev = alignment_device
        if str(dev).startswith("cuda") and not torch.cuda.is_available():
            dev = "cpu"

        ts_dim, proj_fn = load_alignment_projector(Path(alignment_run_dir), device=str(dev))

        for label, df in ("train", train_seg_df), ("test", test_seg_df):
            if (
                skip_alignment_if_present
                and alignment_emb_column in df.columns
                and df[alignment_emb_column].notna().all()
            ):
                print(
                    f"[retrieve] alignment column {alignment_emb_column!r} complete on {label} segments; skip project"
                )
                continue
            project_column_into_df(
                df,
                source_col=source_col,
                out_col=alignment_emb_column,
                ts_emb_dim=ts_dim,
                project_fn=proj_fn,
                batch_size=batch_size,
            )
        test_vec_col = train_vec_col = alignment_emb_column
    elif mode == "chronos":
        test_vec_col, train_vec_col = resolve_chronos_column(
            test_seg_df, train_seg_df, embed_column
        )
    else:
        raise ValueError(f"Unknown retrieval_mode: {retrieval_mode!r} (use chronos or alignment)")

    top_vals_all, dists_per_segment = compute_segment_topk_cosine(
        test_seg_df,
        train_seg_df,
        test_vec_col,
        train_vec_col,
        top_k,
        batch_size,
    )

    test_seg_to_test_series = infer_series_ids_from_seg_pos(test_seg_df[seg_pos_col].to_numpy())
    train_seg_to_train_series = infer_series_ids_from_seg_pos(train_seg_df[seg_pos_col].to_numpy())
    query_groups = group_segment_indices_by_series(test_seg_to_test_series)

    if len(query_groups) != len(test_series_df):
        raise ValueError(
            f"Inferred {len(query_groups)} test series from segments but test_series has {len(test_series_df)} rows"
        )

    top_ids_per_query: list[list[int]] = []
    top_dists_per_query: list[list[float]] = []
    for q_series_id in tqdm(range(len(query_groups)), desc="Aggregate segment -> series"):
        seg_indices = query_groups[q_series_id]
        top_ids, top_agg_dists = aggregate_query_series_topk(
            query_seg_indices=seg_indices,
            topk_vals=top_vals_all,
            topk_dists=dists_per_segment,
            train_seg_to_train_series=train_seg_to_train_series,
            k=top_k,
            aggregation_strategy=aggregation_strategy,
        )
        if len(top_ids) < top_k:
            top_ids = top_ids + [-1] * (top_k - len(top_ids))
            top_agg_dists = top_agg_dists + [float("nan")] * (top_k - len(top_agg_dists))
        top_ids_per_query.append(top_ids)
        top_dists_per_query.append(top_agg_dists[:top_k])

    train_csv_names = train_series_df["csv_name"].astype(str).tolist()

    def idx_to_csv_name(i: int) -> str:
        return train_csv_names[i] if 0 <= i < len(train_csv_names) else ""

    rows = []
    for test_row, topk, topd in zip(
        test_series_df.itertuples(index=False),
        top_ids_per_query,
        top_dists_per_query,
    ):
        out = {"query": str(test_row.csv_name)}
        for i in range(top_k):
            out[f"top-{i + 1}"] = idx_to_csv_name(int(topk[i]))
            d_i = topd[i] if i < len(topd) else float("nan")
            out[f"dist-{i + 1}"] = d_i if int(topk[i]) >= 0 else float("nan")
        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_csv = out_csv.expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"[retrieve] saved {out_csv} ({len(out_df)} queries)")
    return out_csv


def run_retrieve(cfg: DictConfig, output_dir: Path) -> Path:
    from hydra.utils import to_absolute_path

    d = cfg.data
    out = output_dir / str(cfg.retrieve.output_csv_name)
    ec = cfg.retrieve.get("embed_column")
    if ec is None or str(ec).strip() == "" or str(ec).lower() == "auto":
        embed_column = None
    else:
        embed_column = str(ec)
    mode = str(cfg.retrieve.get("mode", "chronos")).strip().lower()
    ard = cfg.retrieve.get("alignment_run_dir")
    if mode == "alignment" and (
        ard is None or (isinstance(ard, str) and ard.strip() == "")
    ):
        out_root = OmegaConf.select(cfg, "paths.outputs_root") or "./outputs"
        ard = f"{str(out_root).rstrip('/')}/alignment"
    alignment_run_dir = (
        Path(to_absolute_path(str(ard))) if ard and str(ard).strip() else None
    )
    aec = cfg.retrieve.get("alignment_emb_column")
    alignment_emb_column = (
        str(aec).strip() if aec is not None and str(aec).strip() else None
    )
    if mode == "alignment" and not alignment_emb_column:
        alignment_emb_column = "emb_aligned"
    asrc = cfg.retrieve.get("alignment_source_column")
    alignment_source_column = (
        None
        if asrc is None or str(asrc).strip() == "" or str(asrc).lower() == "auto"
        else str(asrc)
    )
    return run_retrieval_to_csv(
        train_segments_parquet=Path(to_absolute_path(str(d.train_segments))),
        test_segments_parquet=Path(to_absolute_path(str(d.test_segments))),
        train_series_parquet=Path(to_absolute_path(str(d.train_series))),
        test_series_parquet=Path(to_absolute_path(str(d.test_series))),
        out_csv=out,
        top_k=int(cfg.retrieve.top_k),
        batch_size=int(cfg.retrieve.batch_size),
        aggregation_strategy=str(cfg.retrieve.aggregation_strategy),
        seg_pos_col=str(cfg.retrieve.seg_pos_col),
        embed_column=embed_column,
        retrieval_mode=mode,
        alignment_run_dir=alignment_run_dir,
        alignment_emb_column=alignment_emb_column,
        alignment_source_column=alignment_source_column,
        alignment_device=str(cfg.retrieve.get("alignment_device", "cuda")),
        skip_alignment_if_present=bool(cfg.retrieve.get("skip_alignment_if_present", True)),
    )
