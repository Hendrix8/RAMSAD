"""Load a trained time-series projector from an alignment run directory (3-tsfm–compatible checkpoints)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def _alignment_code_root() -> Path:
    # src/ramsad/alignment_projector.py -> repo root is parents[2]
    repo = Path(__file__).resolve().parents[2]
    align = repo / "alignment"
    if (align / "model" / "embedding_alignment.py").is_file():
        return align
    raise FileNotFoundError(
        f"Expected vendored alignment code at {align}/model/embedding_alignment.py"
    )


def resolve_alignment_run_dir(run_dir: Path) -> Path:
    """
    If ``run_dir`` already contains ``checkpoints/best_model.pt``, return it.

    Otherwise, if ``run_dir`` is a directory (e.g. default ``outputs/alignment``), use the
    immediate subdirectory whose ``checkpoints/best_model.pt`` has the newest mtime.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    ckpt = run_dir / "checkpoints" / "best_model.pt"
    if ckpt.is_file():
        return run_dir
    if not run_dir.is_dir():
        return run_dir
    best: tuple[float, Path] | None = None
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        c = child / "checkpoints" / "best_model.pt"
        if c.is_file():
            m = c.stat().st_mtime
            if best is None or m > best[0]:
                best = (m, child)
    if best is not None:
        return best[1]
    return run_dir


def load_alignment_projector(
    run_dir: Path,
    device: str = "cuda",
) -> tuple[int, callable]:
    """
    Returns (ts_emb_dim, project_fn) where project_fn(np.ndarray [N,D]) -> np.ndarray [N,proj_dim] L2-normalized.
    """
    requested = Path(run_dir).expanduser().resolve()
    run_dir = resolve_alignment_run_dir(requested)
    if run_dir != requested:
        print(f"[retrieve] alignment: using latest run {run_dir} (under {requested})")
    ckpt_path = run_dir / "checkpoints" / "best_model.pt"
    cfg_path = run_dir / "config.yaml"
    if not ckpt_path.is_file():
        hint = ""
        if "abs/path" in str(run_dir) or "/DATE/" in str(run_dir):
            hint = (
                " This path looks like a README placeholder; set retrieve.alignment_run_dir to the "
                "directory printed as 'Output directory:' after training (contains checkpoints/ and config.yaml). "
                "See README: 'Alignment retrieval paths'."
            )
        elif run_dir.is_dir() and not (run_dir / "checkpoints").is_dir():
            hint = (
                " Under this directory, expected a run subfolder with checkpoints/best_model.pt "
                "(train with alignment/ — default base is outputs/alignment), or pass the inner "
                "run-* folder explicitly."
            )
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}.{hint}")
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    root = _alignment_code_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from model.embedding_alignment import EmbeddingAlignmentModel

    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    m = cfg["model"]
    model = EmbeddingAlignmentModel(
        ts_emb_dim=m["ts_emb_dim"],
        desc_emb_dim=m["desc_emb_dim"],
        projection_dim=m["projection_dim"],
        temperature=m["temperature"],
        loss_function=m["loss_function"],
        desc_match_mode=m.get("desc_match_mode", "exact"),
        desc_similarity_threshold=m.get("desc_similarity_threshold", 0.95),
        per_series_loss=m.get("per_series_loss", False),
        preserve_temporal_identity=m.get("preserve_temporal_identity", False),
        identity_penalty_weight=m.get("identity_penalty_weight", 1.0),
        series_level_alignment=m.get("series_level_alignment", False),
        align_to_performance=m.get("align_to_performance", False),
        performance_dim=m.get("performance_dim", 32),
    )

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict")
    if state is None:
        raise KeyError("Checkpoint has neither 'model_state_dict' nor 'state_dict'")
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model.to(dev)
    model.eval()

    ts_emb_dim = int(m["ts_emb_dim"])
    projection_dim = int(m["projection_dim"])

    @torch.no_grad()
    def project_ts_embeddings(embs: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        embs = np.asarray(embs, dtype=np.float32)
        if embs.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape={embs.shape}")
        outs: list[np.ndarray] = []
        for i in range(0, len(embs), batch_size):
            b = torch.tensor(embs[i : i + batch_size], device=dev, dtype=torch.float32)
            p = model.ts_projection(b)
            p = torch.nn.functional.normalize(p, dim=1)
            outs.append(p.cpu().numpy())
        if outs:
            return np.vstack(outs)
        return np.empty((0, projection_dim), dtype=np.float32)

    return ts_emb_dim, project_ts_embeddings


def project_column_into_df(
    df,
    *,
    source_col: str,
    out_col: str,
    ts_emb_dim: int,
    project_fn: callable,
    batch_size: int,
) -> None:
    """Mutate dataframe: fill out_col with projected lists for rows where source_col is valid."""
    from tqdm.auto import tqdm

    def _to_valid_embedding(x, expected_dim: int) -> np.ndarray | None:
        if x is None:
            return None
        arr = np.asarray(x)
        if arr.dtype == object:
            try:
                arr = np.asarray(list(arr), dtype=np.float32)
            except Exception:
                return None
        try:
            arr = np.asarray(arr, dtype=np.float32).ravel()
        except Exception:
            return None
        if arr.shape[0] != expected_dim:
            return None
        if not np.all(np.isfinite(arr)):
            return None
        return arr

    out_values = df[out_col].tolist() if out_col in df.columns else [None] * len(df)
    need_idx = [i for i, v in enumerate(out_values) if v is None]
    if not need_idx:
        return

    valid_idx: list[int] = []
    valid_embs: list[np.ndarray] = []
    src = df[source_col].tolist()
    for i in tqdm(need_idx, desc=f"Project {source_col!r} -> {out_col!r}"):
        arr = _to_valid_embedding(src[i], expected_dim=ts_emb_dim)
        if arr is None:
            continue
        valid_idx.append(i)
        valid_embs.append(arr)

    if valid_embs:
        proj = project_fn(np.vstack(valid_embs), batch_size=batch_size)
        for j, row_idx in enumerate(valid_idx):
            out_values[row_idx] = proj[j].tolist()
    df[out_col] = out_values
