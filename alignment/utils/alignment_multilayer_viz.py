#!/usr/bin/env python3
"""
Alignment visualizations for a trained run:

1) **Triangle pair (default):** Two figures for the same four segments (two per dataset):
   - Pre-alignment: joint PCA of **Chronos** + **sentence-transformer** descriptions (L2-normalized,
     padded to a common dim). Per-dataset colors; one triangle per dataset (two Chronos vertices +
     mean description vertex) to summarize spread vs text.
   - Post-alignment: joint PCA of **run ts_projection** + **run desc_projection** (model outputs,
     L2-normalized as in training). Same layout.

   PCA mode (``--pca-mode``): ``separate`` fits PCA independently on each panel (literal “PCA of
   Chronos+desc” then “PCA after the run”). Use ``fit_pre`` to fit PCA on the pre panel only and
   transform the post panel with the same axes—then triangle areas are comparable and usually
   shrink after alignment. ``joint_16`` fits one PCA on all 16 points (pre+post stacked).

2) **Legacy multilayer:** optional single PCA mixing raw series, Chronos, run, ST, and aligned desc
   (`--legacy-multilayer`).

Usage:
  cd /path/to/3-tsfm/model && python utils/alignment_multilayer_viz.py \\
    --train-then-id-eval-dir outputs/train_then_id_eval_example

  Or:
  python utils/alignment_multilayer_viz.py --exp-dir /data/.../run-YYYYMMDD-HHMMSS-multi_positive
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.decomposition import PCA

# model/ package (parent of utils/)
_MODEL_ROOT = Path(__file__).resolve().parent.parent
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from model.embedding_alignment import EmbeddingAlignmentModel


def _apply_darth_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 40,
            "axes.titlesize": 40,
            "axes.labelsize": 32,
            "xtick.labelsize": 28,
            "ytick.labelsize": 28,
            "legend.fontsize": 22,
        }
    )


def _flatten_series(x, dim: int) -> np.ndarray:
    if x is None:
        return np.zeros(dim, dtype=np.float32)
    a = np.asarray(x, dtype=np.float32).ravel()
    if a.size >= dim:
        return a[:dim].astype(np.float32)
    out = np.zeros(dim, dtype=np.float32)
    out[: a.size] = a
    return out


def _to_arr(x, dim: int) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32).ravel()
    if a.size >= dim:
        return a[:dim].copy()
    out = np.zeros(dim, dtype=np.float32)
    out[: a.size] = a
    return out


def _pad_to_dim(vec: np.ndarray, dim: int) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).ravel()
    if v.size >= dim:
        return v[:dim].copy()
    out = np.zeros(dim, dtype=np.float32)
    out[: v.size] = v
    return out


def _l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64).ravel()
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)


def _triangle_area_2d(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
    """Signed area * 2 of triangle (p,q,r); return absolute geometric area."""
    p, q, r = [np.asarray(x, dtype=np.float64).ravel()[:2] for x in (p, q, r)]
    return 0.5 * abs((q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1]))


def _normalize_dataset_id(s: str) -> str:
    """Strip edges and collapse internal whitespace so 'Foo' and '  Foo ' are the same dataset."""
    return " ".join(str(s).strip().split())


def _short_legend_labels_for_pair(d_a: str, d_b: str) -> tuple[str, str, str, str]:
    """
    Compact legend text: e.g. Facility1 / Facility1_desc / Facility2 / Facility2_desc
    when ids contain ``Facility``; otherwise D1 / D1_desc / D2 / D2_desc.
    """
    def base(full: str, idx: int) -> str:
        if re.search(r"Facility", full, re.I):
            return f"Facility{idx}"
        return f"D{idx}"

    a, b = base(d_a, 1), base(d_b, 2)
    return (a, f"{a}_desc", b, f"{b}_desc")


def select_two_datasets(
    df: pd.DataFrame,
    series_col: str,
    rng: np.random.Generator,
    dataset_a: str | None = None,
    dataset_b: str | None = None,
    require_cross_domain: bool = False,
) -> tuple[str, str, np.ndarray, np.ndarray]:
    """
    Pick two **distinct** dataset ids (after normalization), each with at least two rows.

    If ``require_cross_domain`` is True, the two ids must also map to different heuristic domain
    labels (same rule as ``ood_embedding_triptych_viz``), e.g. not ``Medical[0]`` vs ``Medical[1]``.

    Returns ``(d_a, d_b, idx_a, idx_b)`` where ``idx_*`` are row positions (length 2).
    """
    raw = df[series_col].astype(str).values
    norm = np.array([_normalize_dataset_id(x) for x in raw], dtype=object)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, n in enumerate(norm):
        groups[str(n)].append(i)

    eligible = sorted([n for n, ix in groups.items() if len(ix) >= 2 and str(n).strip()])
    if len(eligible) < 2:
        raise RuntimeError(
            f"Need at least two distinct {series_col!r} values with >=2 rows each "
            f"(after normalizing whitespace); found {len(eligible)}: {eligible[:30]}"
        )

    if dataset_a is not None or dataset_b is not None:
        if dataset_a is None or dataset_b is None:
            raise ValueError("Pass both --dataset-a and --dataset-b, or neither")
        na = _normalize_dataset_id(dataset_a)
        nb = _normalize_dataset_id(dataset_b)
        if na == nb:
            raise ValueError(
                f"--dataset-a and --dataset-b must be two different datasets; both normalize to {na!r}"
            )
        for key, name in ((na, dataset_a), (nb, dataset_b)):
            if key not in groups or len(groups[key]) < 2:
                raise ValueError(
                    f"Dataset {name!r} (normalized {key!r}) must appear on at least two rows in the parquet"
                )
        d_a, d_b = na, nb
        if require_cross_domain:
            from utils.ood_embedding_triptych_viz import _domain_label_from_dataset_id

            la = _domain_label_from_dataset_id(d_a).lower()
            lb = _domain_label_from_dataset_id(d_b).lower()
            if la == lb:
                raise ValueError(
                    "require_cross_domain: --dataset-a and --dataset-b must map to different "
                    f"domain labels; both map to {la!r}"
                )
    else:
        if require_cross_domain:
            from utils.ood_embedding_triptych_viz import _domain_label_from_dataset_id

            cross_pairs: list[tuple[str, str]] = []
            for i in range(len(eligible)):
                for j in range(i + 1, len(eligible)):
                    da, db = eligible[i], eligible[j]
                    if (
                        _domain_label_from_dataset_id(da).lower()
                        != _domain_label_from_dataset_id(db).lower()
                    ):
                        cross_pairs.append((da, db))
            if not cross_pairs:
                raise RuntimeError(
                    f"require_cross_domain: no two distinct {series_col!r} values with >=2 rows each "
                    f"from different heuristic domain labels. Eligible ids: {eligible[:40]}"
                )
            rng.shuffle(cross_pairs)
            d_a, d_b = cross_pairs[0]
        else:
            rng.shuffle(eligible)
            d_a, d_b = eligible[0], eligible[1]

    if d_a == d_b:
        raise RuntimeError(f"Internal error: duplicate dataset id after selection: {d_a!r}")

    idx_a = np.array(groups[d_a][:2], dtype=np.int64)
    idx_b = np.array(groups[d_b][:2], dtype=np.int64)
    return d_a, d_b, idx_a, idx_b


def parse_run_dir_from_id_train_log(log_path: Path) -> Path:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if line.startswith("Using new run:"):
            return Path(line.split("Using new run:", 1)[1].strip())
    raise ValueError(f"Could not find 'Using new run:' in {log_path}")


def load_model(cfg: dict, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
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
        raise KeyError("Checkpoint missing model_state_dict/state_dict")
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def _plot_dataset_triangle_pca(
    ax: plt.Axes,
    Z: np.ndarray,
    X_high: np.ndarray,
    pca: PCA,
    *,
    color_a: str,
    color_b: str,
    legend_labels: tuple[str, str, str, str],
) -> dict:
    """
    Scatter Chronos/TS as circles and descriptions as triangles (colors = dataset).
    One triangle per dataset: two TS/Chronos points + mean(description) in high-D, then PCA.
    """
    # Rows: [ts_0, ts_1, ts_2, ts_3], [desc_0, desc_1, desc_2, desc_3]
    z_ts = Z[:4]
    z_de = Z[4:8]
    ds_ids = [0, 0, 1, 1]  # segment order: two from A, two from B

    for i in range(4):
        c = color_a if ds_ids[i] == 0 else color_b
        ax.scatter(
            z_ts[i, 0],
            z_ts[i, 1],
            c=c,
            s=420,
            marker="o",
            edgecolors="black",
            linewidths=1.0,
            zorder=4,
        )

    for i in range(4):
        c = color_a if ds_ids[i] == 0 else color_b
        ax.scatter(
            z_de[i, 0],
            z_de[i, 1],
            c=c,
            s=420,
            marker="^",
            edgecolors="black",
            linewidths=1.0,
            zorder=4,
        )

    areas: dict[str, float] = {}
    # Dataset A triangle: ts rows 0,1 + mean desc rows 4,5
    m_de_a = 0.5 * (X_high[4] + X_high[5])
    z_m_a = pca.transform(m_de_a.reshape(1, -1).astype(np.float32))[0]
    poly_a = np.vstack([Z[0], Z[1], z_m_a])
    areas["dataset_a"] = _triangle_area_2d(poly_a[0], poly_a[1], poly_a[2])
    patch_a = Polygon(
        poly_a,
        closed=True,
        facecolor=color_a,
        edgecolor=color_a,
        linewidth=2.5,
        alpha=0.22,
        zorder=1,
    )
    ax.add_patch(patch_a)

    m_de_b = 0.5 * (X_high[6] + X_high[7])
    z_m_b = pca.transform(m_de_b.reshape(1, -1).astype(np.float32))[0]
    poly_b = np.vstack([Z[2], Z[3], z_m_b])
    areas["dataset_b"] = _triangle_area_2d(poly_b[0], poly_b[1], poly_b[2])
    patch_b = Polygon(
        poly_b,
        closed=True,
        facecolor=color_b,
        edgecolor=color_b,
        linewidth=2.5,
        alpha=0.22,
        zorder=1,
    )
    ax.add_patch(patch_b)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)
    _mk = 22
    ax.legend(
        [
            plt.Line2D([0], [0], linestyle="none", marker="o", color="w", markerfacecolor=color_a, markersize=_mk),
            plt.Line2D([0], [0], linestyle="none", marker="^", color="w", markerfacecolor=color_a, markersize=_mk),
            plt.Line2D([0], [0], linestyle="none", marker="o", color="w", markerfacecolor=color_b, markersize=_mk),
            plt.Line2D([0], [0], linestyle="none", marker="^", color="w", markerfacecolor=color_b, markersize=_mk),
        ],
        list(legend_labels),
        loc="best",
        frameon=False,
        prop={"size": 38},
        labelspacing=1.2,
        handletextpad=1.0,
    )
    return areas


def run_alignment_triangle_pair_viz(
    exp_dir: Path,
    out_dir: Path | None,
    device_str: str,
    random_seed: int,
    common_dim: int | None = None,
    pca_mode: str = "separate",
    dataset_a: str | None = None,
    dataset_b: str | None = None,
) -> dict:
    """
    Two figures: (1) Chronos + ST desc joint PCA with per-dataset triangles.
    (2) Post-run aligned TS + aligned desc joint PCA with same triangle construction.

    ``pca_mode``:
    - ``separate`` (default): PCA fit independently on the 8 pre rows and on the 8 post rows
      (matches “PCA of Chronos+desc” then “PCA after the run”).
    - ``fit_pre``: fit PCA on pre rows only, then ``transform`` both pre and post (same PC axes;
      post is viewed in the Chronos+ST coordinate system).
    - ``joint_16``: one PCA on 16 rows (pre stacked with post); both panels share axes but mixes
      scales (use sparingly).
    """
    exp_dir = exp_dir.resolve()
    cfg_path = exp_dir / "config.yaml"
    if not cfg_path.is_file():
        hydra = exp_dir / ".hydra" / "config.yaml"
        if hydra.is_file():
            cfg_path = hydra
    if not cfg_path.is_file():
        raise FileNotFoundError(f"No config.yaml under {exp_dir}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("model", {}).get("align_to_performance"):
        raise ValueError("align_to_performance=True not supported for this viz")

    ckpt = exp_dir / "checkpoints" / "best_model.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    data = cfg["data"]
    parquet_path = Path(data["parquet_file"])
    ts_col = data["ts_emb_col"]
    desc_col = data["desc_emb_col"]
    series_col = data.get("series_col") or "dataset_name"
    ts_dim = int(cfg["model"]["ts_emb_dim"])
    desc_dim = int(cfg["model"]["desc_emb_dim"])
    proj_dim = int(cfg["model"]["projection_dim"])

    if common_dim is None:
        common_dim = max(ts_dim, desc_dim)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, ckpt, device)

    need_cols = [ts_col, desc_col, series_col, "series"]
    df = pd.read_parquet(parquet_path, columns=need_cols)

    rng = np.random.default_rng(random_seed)
    d_a, d_b, idx_a, idx_b = select_two_datasets(
        df, series_col, rng, dataset_a=dataset_a, dataset_b=dataset_b
    )
    print(f"Two distinct datasets ({series_col}, normalized): {d_a!r} vs {d_b!r}")

    segment_indices = [idx_a[0], idx_a[1], idx_b[0], idx_b[1]]
    chronos_batch: list[torch.Tensor] = []
    desc_batch: list[torch.Tensor] = []

    rows_ch: list[np.ndarray] = []
    rows_de: list[np.ndarray] = []
    for ix in segment_indices:
        ch = _to_arr(df.iloc[ix][ts_col], ts_dim)
        de = _to_arr(df.iloc[ix][desc_col], desc_dim)
        ch_n = _l2_normalize(ch)
        de_n = _l2_normalize(de)
        rows_ch.append(_pad_to_dim(ch_n, common_dim))
        rows_de.append(_pad_to_dim(de_n, common_dim))
        chronos_batch.append(torch.tensor(ch, device=device).unsqueeze(0))
        desc_batch.append(torch.tensor(de, device=device).unsqueeze(0))

    X_pre = np.stack(rows_ch + rows_de, axis=0).astype(np.float32)

    aligned_ts: list[np.ndarray] = []
    aligned_de: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(4):
            t = chronos_batch[i]
            d = desc_batch[i]
            ts_p, desc_p = model(t, d)
            aligned_ts.append(ts_p.cpu().numpy().ravel()[:proj_dim])
            aligned_de.append(desc_p.cpu().numpy().ravel()[:proj_dim])

    X_post = np.stack(aligned_ts + aligned_de, axis=0).astype(np.float32)

    joint_dim = max(common_dim, proj_dim)
    X_pre_j = np.stack([_pad_to_dim(X_pre[i], joint_dim) for i in range(8)], axis=0).astype(np.float32)
    X_post_j = np.stack([_pad_to_dim(X_post[i], joint_dim) for i in range(8)], axis=0).astype(np.float32)

    if pca_mode == "joint_16":
        X_joint = np.vstack([X_pre_j, X_post_j])
        n_comp = min(2, X_joint.shape[0] - 1, X_joint.shape[1])
        pca_pre = PCA(n_components=n_comp, random_state=random_seed)
        Z_joint = pca_pre.fit_transform(X_joint)
        Z_pre = Z_joint[:8]
        Z_post = Z_joint[8:]
        pca_post = pca_pre
        X_pre_for_tri = X_pre_j
        X_post_for_tri = X_post_j
    elif pca_mode == "fit_pre":
        n_comp = min(2, X_pre_j.shape[0] - 1, X_pre_j.shape[1])
        pca_pre = PCA(n_components=n_comp, random_state=random_seed)
        pca_pre.fit(X_pre_j)
        Z_pre = pca_pre.transform(X_pre_j)
        Z_post = pca_pre.transform(X_post_j)
        pca_post = pca_pre
        X_pre_for_tri = X_pre_j
        X_post_for_tri = X_post_j
    elif pca_mode == "separate":
        n_comp_pre = min(2, X_pre.shape[0] - 1, X_pre.shape[1])
        pca_pre = PCA(n_components=n_comp_pre, random_state=random_seed)
        Z_pre = pca_pre.fit_transform(X_pre)
        n_comp_post = min(2, X_post.shape[0] - 1, X_post.shape[1])
        pca_post = PCA(n_components=n_comp_post, random_state=random_seed)
        Z_post = pca_post.fit_transform(X_post)
        X_pre_for_tri = X_pre
        X_post_for_tri = X_post
    else:
        raise ValueError(f"Unknown pca_mode: {pca_mode!r}")

    out_dir = Path(out_dir or (exp_dir / "plots"))
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_pre = out_dir / "alignment_triangle_pre_chronos_desc.pdf"
    pdf_post = out_dir / "alignment_triangle_post_run.pdf"
    csv_path = out_dir / "alignment_triangle_areas.csv"
    meta_path = out_dir / "alignment_triangle_meta.json"

    color_a, color_b = "#1f77b4", "#ff7f0e"
    leg_labels = _short_legend_labels_for_pair(d_a, d_b)
    print(f"Legend labels: {list(leg_labels)}")

    _apply_darth_style()
    fig1, ax1 = plt.subplots(figsize=(16, 10))
    areas_pre = _plot_dataset_triangle_pca(
        ax1,
        Z_pre,
        X_pre_for_tri,
        pca_pre,
        color_a=color_a,
        color_b=color_b,
        legend_labels=leg_labels,
    )
    fig1.tight_layout()
    fig1.savefig(pdf_pre, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(16, 10))
    areas_post = _plot_dataset_triangle_pca(
        ax2,
        Z_post,
        X_post_for_tri,
        pca_post,
        color_a=color_a,
        color_b=color_b,
        legend_labels=leg_labels,
    )
    fig2.tight_layout()
    fig2.savefig(pdf_post, bbox_inches="tight")
    plt.close(fig2)

    pd.DataFrame(
        [
            {"stage": "pre_chronos_st", "dataset": d_a, "triangle_area": areas_pre["dataset_a"]},
            {"stage": "pre_chronos_st", "dataset": d_b, "triangle_area": areas_pre["dataset_b"]},
            {"stage": "post_run", "dataset": d_a, "triangle_area": areas_post["dataset_a"]},
            {"stage": "post_run", "dataset": d_b, "triangle_area": areas_post["dataset_b"]},
        ]
    ).to_csv(csv_path, index=False)

    meta = {
        "exp_dir": str(exp_dir),
        "parquet_file": str(parquet_path),
        "ts_emb_col": ts_col,
        "desc_emb_col": desc_col,
        "series_col": series_col,
        "common_dim_pre": common_dim,
        "joint_pca_dim": joint_dim,
        "pca_mode": pca_mode,
        "projection_dim": proj_dim,
        "datasets_normalized": [d_a, d_b],
        "legend_labels_short": list(leg_labels),
        "segment_row_indices": [int(x) for x in segment_indices],
        "triangle_areas_pre": areas_pre,
        "triangle_areas_post": areas_post,
        "pdfs": {"pre": str(pdf_pre), "post": str(pdf_post)},
        "csv": str(csv_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote: {pdf_pre}")
    print(f"Wrote: {pdf_post}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {meta_path}")
    if pca_mode == "separate":
        print(
            "pca_mode=separate: post/pre triangle areas use different PCA scales; "
            "compare visually within each panel, or use --pca-mode fit_pre / joint_16 for shared axes."
        )
    else:
        for key in ("dataset_a", "dataset_b"):
            ap, aq = areas_pre[key], areas_post[key]
            ratio = (aq / ap) if ap > 1e-20 else float("nan")
            print(f"Triangle area ratio post/pre ({key}): {ratio:.4f} (pre={ap:.6g}, post={aq:.6g})")
    return meta


def run_alignment_multilayer_viz(
    exp_dir: Path,
    out_dir: Path | None,
    device_str: str,
    random_seed: int,
    common_dim: int | None = None,
    dataset_a: str | None = None,
    dataset_b: str | None = None,
) -> dict:
    """
    Build one PCA on padded feature rows:
      - per segment (4): raw_series, chronos, run_ts_proj (512 padded)
      - per dataset (2): desc_st, desc_al (512 padded)
    """
    exp_dir = exp_dir.resolve()
    cfg_path = exp_dir / "config.yaml"
    if not cfg_path.is_file():
        hydra = exp_dir / ".hydra" / "config.yaml"
        if hydra.is_file():
            cfg_path = hydra
    if not cfg_path.is_file():
        raise FileNotFoundError(f"No config.yaml under {exp_dir}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("model", {}).get("align_to_performance"):
        raise ValueError("align_to_performance=True not supported for this viz")

    ckpt = exp_dir / "checkpoints" / "best_model.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    data = cfg["data"]
    parquet_path = Path(data["parquet_file"])
    ts_col = data["ts_emb_col"]
    desc_col = data["desc_emb_col"]
    series_col = data.get("series_col") or "dataset_name"
    ts_dim = int(cfg["model"]["ts_emb_dim"])
    desc_dim = int(cfg["model"]["desc_emb_dim"])
    proj_dim = int(cfg["model"]["projection_dim"])

    if common_dim is None:
        common_dim = max(ts_dim, desc_dim, proj_dim)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, ckpt, device)

    need_cols = [ts_col, desc_col, series_col, "series"]
    df = pd.read_parquet(parquet_path, columns=need_cols)

    rng = np.random.default_rng(random_seed)
    d_a, d_b, idx_a, idx_b = select_two_datasets(
        df, series_col, rng, dataset_a=dataset_a, dataset_b=dataset_b
    )
    print(f"Two distinct datasets ({series_col}, normalized): {d_a!r} vs {d_b!r}")
    idx_a_set = set(int(x) for x in idx_a)

    def _dataset_label(row_ix: int) -> str:
        return d_a if int(row_ix) in idx_a_set else d_b

    rows_meta: list[dict] = []
    vectors: list[np.ndarray] = []

    def add_row(modality: str, row_idx: int | None, dataset: str, vec: np.ndarray, extra: str = ""):
        rows_meta.append(
            {
                "modality": modality,
                "row_index": row_idx if row_idx is not None else -1,
                "dataset": dataset,
                "extra": extra,
            }
        )
        vectors.append(_pad_to_dim(vec, common_dim))

    # Four segment rows: raw, chronos, run (projector on chronos)
    segment_indices = [idx_a[0], idx_a[1], idx_b[0], idx_b[1]]
    chronos_batch = []
    desc_batch = []
    for ix in segment_indices:
        raw = _flatten_series(df.iloc[ix]["series"], ts_dim)
        ch = _to_arr(df.iloc[ix][ts_col], ts_dim)
        de = _to_arr(df.iloc[ix][desc_col], desc_dim)
        add_row("raw_segment", ix, _dataset_label(ix), raw)
        add_row("chronos", ix, _dataset_label(ix), ch)
        chronos_batch.append(torch.tensor(ch, device=device).unsqueeze(0))
        desc_batch.append(torch.tensor(de, device=device).unsqueeze(0))

    with torch.no_grad():
        for i, ix in enumerate(segment_indices):
            t = chronos_batch[i]
            d = desc_batch[i]
            ts_p, desc_p = model(t, d)
            # Pre-normalization vectors (as in forward before normalize — take LN output)
            ts_vec = ts_p.cpu().numpy().ravel()
            add_row("run_ts_proj", ix, _dataset_label(ix), ts_vec[:proj_dim])

    # Description-only (one per dataset): ST embedding + aligned desc from zero TS
    z_ts = torch.zeros(1, ts_dim, device=device, dtype=torch.float32)
    for tag, ix in [("desc_st", idx_a[0]), ("desc_st", idx_b[0])]:
        de = _to_arr(df.iloc[ix][desc_col], desc_dim)
        add_row(tag, None, _dataset_label(ix), de)

    with torch.no_grad():
        for tag, ix in [("desc_al", idx_a[0]), ("desc_al", idx_b[0])]:
            de = torch.tensor(_to_arr(df.iloc[ix][desc_col], desc_dim), device=device).unsqueeze(0)
            _, dp = model(z_ts, de)
            add_row(tag, None, _dataset_label(ix), dp.cpu().numpy().ravel()[:proj_dim])

    X = np.stack(vectors, axis=0)
    pca = PCA(n_components=min(2, X.shape[0] - 1, X.shape[1]), random_state=random_seed)
    Z = pca.fit_transform(X.astype(np.float32))

    out_dir = Path(out_dir or (exp_dir / "plots"))
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "alignment_multilayer_pca.pdf"
    csv_path = out_dir / "alignment_multilayer_data.csv"
    meta_path = out_dir / "alignment_multilayer_meta.json"

    _apply_darth_style()
    fig, ax = plt.subplots(figsize=(18, 11))
    styles = {
        "raw_segment": ("o", "#7f7f7f", 280),
        "chronos": ("o", "#1f77b4", 380),
        "run_ts_proj": ("o", "#2ca02c", 480),
        "desc_st": ("^", "#9467bd", 520),
        "desc_al": ("^", "#d62728", 520),
    }
    for i, meta in enumerate(rows_meta):
        mod = meta["modality"]
        mk, color, sz = styles.get(mod, ("o", "#333333", 300))
        ax.scatter(
            Z[i, 0],
            Z[i, 1],
            c=color,
            s=sz,
            marker=mk,
            edgecolors="black",
            linewidths=1.0,
            zorder=3,
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)
    _leg_handles = [
        plt.Line2D([0], [0], linestyle="none", marker="o", color="w", markerfacecolor="#7f7f7f", markersize=12),
        plt.Line2D([0], [0], linestyle="none", marker="o", color="w", markerfacecolor="#1f77b4", markersize=12),
        plt.Line2D([0], [0], linestyle="none", marker="o", color="w", markerfacecolor="#2ca02c", markersize=12),
        plt.Line2D([0], [0], linestyle="none", marker="^", color="w", markerfacecolor="#9467bd", markersize=12),
        plt.Line2D([0], [0], linestyle="none", marker="^", color="w", markerfacecolor="#d62728", markersize=12),
    ]
    _leg_labels = [
        "Raw segment (padded)",
        "Chronos (ts_emb_col)",
        "Run ts_projection(Chronos)",
        "Sentence-T desc (raw)",
        "Aligned desc_projection",
    ]
    ax.legend(_leg_handles, _leg_labels, loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    out_df = pd.DataFrame(rows_meta)
    out_df["pc1"] = Z[:, 0]
    out_df["pc2"] = Z[:, 1]
    out_df["common_dim"] = common_dim
    out_df["dataset_a"] = d_a
    out_df["dataset_b"] = d_b
    out_df["exp_dir"] = str(exp_dir)
    out_df.to_csv(csv_path, index=False)

    meta = {
        "exp_dir": str(exp_dir),
        "parquet_file": str(parquet_path),
        "ts_emb_col": ts_col,
        "desc_emb_col": desc_col,
        "series_col": series_col,
        "common_dim_pad": common_dim,
        "point_order": [m["modality"] for m in rows_meta],
        "datasets_chosen": [d_a, d_b],
        "segment_row_indices": [int(x) for x in segment_indices],
        "pdf": str(pdf_path),
        "csv": str(csv_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote: {pdf_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {meta_path}")
    return meta


def main() -> None:
    p = argparse.ArgumentParser(
        description="Alignment PCA viz: default = two triangle figures (pre Chronos+ST, post run); optional legacy multilayer plot"
    )
    p.add_argument("--exp-dir", type=Path, default=None, help="Training run directory (config + best_model.pt)")
    p.add_argument(
        "--train-then-id-eval-dir",
        type=Path,
        default=None,
        help="Folder from run_train_then_id_eval.sh (uses id_train.log → run path)",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: run_dir/plots)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--common-dim", type=int, default=None, help="Pad vectors for pre-alignment PCA (default: max(ts,desc))")
    p.add_argument(
        "--legacy-multilayer",
        action="store_true",
        help="Also write alignment_multilayer_pca.pdf (single PCA mixing raw/Chronos/run/ST/aligned desc)",
    )
    p.add_argument(
        "--pca-mode",
        choices=("separate", "fit_pre", "joint_16"),
        default="separate",
        help="separate=independent PCA per figure; fit_pre=PCA on pre then transform post; joint_16=one PCA on 16 rows",
    )
    p.add_argument(
        "--dataset-a",
        type=str,
        default=None,
        help="First dataset id (must use with --dataset-b). Compared after normalizing whitespace.",
    )
    p.add_argument(
        "--dataset-b",
        type=str,
        default=None,
        help="Second dataset id, distinct from --dataset-a.",
    )
    args = p.parse_args()

    if args.train_then_id_eval_dir and args.exp_dir:
        raise SystemExit("Pass only one of --exp-dir or --train-then-id-eval-dir")
    if args.train_then_id_eval_dir:
        log = args.train_then_id_eval_dir / "id_train.log"
        if not log.is_file():
            raise SystemExit(f"Missing {log}")
        exp_dir = parse_run_dir_from_id_train_log(log)
        out_dir = args.out_dir or (args.train_then_id_eval_dir.resolve())
    elif args.exp_dir:
        exp_dir = args.exp_dir
        out_dir = args.out_dir
    else:
        raise SystemExit("Provide --exp-dir or --train-then-id-eval-dir")

    run_alignment_triangle_pair_viz(
        exp_dir=exp_dir,
        out_dir=out_dir,
        device_str=args.device,
        random_seed=args.seed,
        common_dim=args.common_dim,
        pca_mode=args.pca_mode,
        dataset_a=args.dataset_a,
        dataset_b=args.dataset_b,
    )
    if args.legacy_multilayer:
        run_alignment_multilayer_viz(
            exp_dir=exp_dir,
            out_dir=out_dir,
            device_str=args.device,
            random_seed=args.seed,
            common_dim=args.common_dim,
            dataset_a=args.dataset_a,
            dataset_b=args.dataset_b,
        )


if __name__ == "__main__":
    main()
