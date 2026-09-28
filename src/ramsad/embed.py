from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm.auto import tqdm


def embed_parquet_chronos(
    parquet_path: Path,
    *,
    model_id: str,
    embed_column: str,
    batch_size: int,
    series_col: str,
    mask_col: str,
    skip_if_column_present: bool,
) -> None:
    """Add Chronos encoder column to a segment parquet in place."""
    parquet_path = parquet_path.expanduser().resolve()
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)

    df = pd.read_parquet(parquet_path)
    if skip_if_column_present and embed_column in df.columns:
        filled = df[embed_column].notna().sum()
        if filled == len(df):
            print(f"[embed] skip (column {embed_column!r} complete): {parquet_path}")
            return
        print(f"[embed] column {embed_column!r} partial ({filled}/{len(df)}); recomputing all rows")

    if series_col not in df.columns or mask_col not in df.columns:
        raise ValueError(f"Expected columns {series_col!r}, {mask_col!r}; got {list(df.columns)}")

    from chronos import BaseChronosPipeline, Chronos2Pipeline

    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[embed] loading {model_id} on {device_map} ...")
    pipeline = BaseChronosPipeline.from_pretrained(model_id, device_map=device_map)

    series_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    valid_indices: list[int] = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"rows:{parquet_path.name}"):
        ts = row[series_col]
        mask = row[mask_col]
        if isinstance(ts, list):
            ts = np.array(ts, dtype=np.float32)
        elif isinstance(ts, np.ndarray):
            ts = ts.astype(np.float32)
        else:
            continue
        if isinstance(mask, list):
            mask = np.array(mask, dtype=bool)
        elif isinstance(mask, np.ndarray):
            mask = mask.astype(bool)
        else:
            continue
        if len(ts) > 0 and len(mask) == len(ts) and np.any(mask):
            series_list.append(ts)
            mask_list.append(mask)
            valid_indices.append(int(idx))

    if not series_list:
        raise RuntimeError(f"No valid series in {parquet_path}")

    masked_series = [np.where(m, s, np.nan).astype(np.float32) for s, m in zip(series_list, mask_list)]

    if isinstance(pipeline, Chronos2Pipeline):
        # chronos>=2.x: embed(inputs=..., batch_size=...) → list of (n_variates, n_tokens, d_model)
        emb_list, _ = pipeline.embed(inputs=masked_series, batch_size=batch_size)
        embeddings = np.stack(
            [t.float().mean(dim=(0, 1)).cpu().numpy().astype(np.float32) for t in emb_list],
            axis=0,
        )
    else:
        # Chronos / Bolt: embed(context=2d tensor with nan padding) → (batch, seq_tokens, d_model)
        context = np.stack(masked_series, axis=0)
        emb_tensor, _state = pipeline.embed(context=context)
        if not isinstance(emb_tensor, torch.Tensor):
            raise TypeError(f"Unexpected embed return type: {type(emb_tensor)}")
        if emb_tensor.ndim != 3:
            raise ValueError(f"Expected 3D encoder output, got shape {tuple(emb_tensor.shape)}")
        embeddings = emb_tensor.float().mean(dim=1).cpu().numpy().astype(np.float32)

    emb_col: list = [None] * len(df)
    for i, row_idx in enumerate(valid_indices):
        emb_col[row_idx] = embeddings[i].tolist()
    df[embed_column] = emb_col

    df.to_parquet(parquet_path, index=False, engine="pyarrow", compression="snappy")
    n_ok = sum(x is not None for x in emb_col)
    print(f"[embed] wrote {embed_column!r} for {n_ok}/{len(df)} rows -> {parquet_path}")


def run_embed(cfg: DictConfig) -> None:
    """Embed train and test segment parquets from Hydra config."""
    from hydra.utils import to_absolute_path

    data = cfg.data
    paths = [data.train_segments, data.test_segments]

    for p in paths:
        path = Path(to_absolute_path(str(p)))
        embed_parquet_chronos(
            path,
            model_id=str(cfg.embed.model_id),
            embed_column=str(cfg.embed.embed_column),
            batch_size=int(cfg.embed.batch_size),
            series_col=str(cfg.embed.series_col),
            mask_col=str(cfg.embed.mask_col),
            skip_if_column_present=bool(cfg.embed.skip_if_column_present),
        )
