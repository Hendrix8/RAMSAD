"""Add sentence-transformer text embeddings to segment parquets (in place)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig


def embed_parquet_desc(
    parquet_path: Path,
    *,
    model_id: str,
    text_column: str,
    out_column: str,
    batch_size: int,
    device: str | None,
    skip_if_column_present: bool,
) -> None:
    """Encode ``text_column`` with MPNet (or ``model_id``) and write ``out_column`` as list[float] per row."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "Missing sentence-transformers. Install: pip install 'sentence-transformers>=2.2' "
            "or pip install -e '.[desc]'"
        ) from e

    parquet_path = parquet_path.expanduser().resolve()
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)

    df = pd.read_parquet(parquet_path)
    if text_column not in df.columns:
        raise ValueError(
            f"Column {text_column!r} not in {parquet_path.name}; "
            f"set embed_desc.text_column or add text to the parquet."
        )

    if skip_if_column_present and out_column in df.columns:
        filled = df[out_column].notna().sum()
        if filled == len(df):
            print(f"[embed-desc] skip (column {out_column!r} complete): {parquet_path}")
            return
        print(
            f"[embed-desc] column {out_column!r} partial ({filled}/{len(df)}); overwriting all rows"
        )

    texts = df[text_column].astype(str).tolist()
    dev: str | None
    if device is None or str(device).strip().lower() in ("auto", "null", "none", ""):
        dev = None
    else:
        dev = str(device).strip()

    print(f"[embed-desc] loading {model_id} (device={dev or 'auto'}) …")
    model = SentenceTransformer(model_id, device=dev)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    )
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {emb.shape}")

    df[out_column] = [row.tolist() for row in emb]

    df.to_parquet(parquet_path, index=False, engine="pyarrow", compression="snappy")
    print(f"[embed-desc] wrote {out_column!r} shape={emb.shape} -> {parquet_path}")


def run_embed_desc(cfg: DictConfig) -> None:
    """Embed train + test segment parquets from Hydra ``cfg.embed_desc`` and ``cfg.data`` paths."""
    from hydra.utils import to_absolute_path

    ed = cfg.embed_desc
    data = cfg.data
    paths = [Path(to_absolute_path(str(data.train_segments))), Path(to_absolute_path(str(data.test_segments)))]
    raw_dev = ed.get("device")
    if raw_dev is None or str(raw_dev).strip().lower() in ("auto", "null", "none", ""):
        dev = None
    else:
        dev = str(raw_dev).strip()
    for p in paths:
        embed_parquet_desc(
            p,
            model_id=str(ed.model_id),
            text_column=str(ed.text_column),
            out_column=str(ed.out_column),
            batch_size=int(ed.batch_size),
            device=dev,
            skip_if_column_present=bool(ed.skip_if_column_present),
        )
