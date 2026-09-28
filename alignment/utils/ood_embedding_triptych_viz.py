#!/usr/bin/env python3
"""
Two separate 2D PCA figures for OOD alignment runs:

1) **Pre (one figure):** Chronos (TS) and sentence-transformer **description** embeddings in the
   **same** plot (circles vs triangles; colors = dataset).

2) **Post (one figure):** Aligned TS + aligned description embeddings, **same PC basis** as (1)
   (PCA fit on the 8 pre rows; post rows transformed with the same axes).

**Identical axis limits** on both figures so scale is comparable.

Legend: TS/Chronos as ``{domain}_series`` (e.g. ``Facility_series``); description rows as
``{domain}_desc``.

Dataset pairs are chosen so the two series map to **different** heuristic domain labels (e.g. not
two distinct ids both labeled ``Medical``). With ``--num-pair-iterations N`` (N>1), samples **N** distinct
unordered cross-domain pairs (no replacement among eligible combinations), each in its own subfolder
``pair_00_*__vs__* /``.

Usage (from repo ``model/`` directory, or with PYTHONPATH including ``model/``):

  python utils/ood_embedding_triptych_viz.py --exp-dir /path/to/run-...-multi_positive

  python utils/ood_embedding_triptych_viz.py --exp-dir /path/to/run --num-pair-iterations 5

  # Optional path figures: single-dataset + two-dataset (TS polygons with transparent fill)
  python utils/ood_embedding_triptych_viz.py --exp-dir /path/to/run --single-dataset-path \\
    --path-max-segments 64 --path-fill-alpha 0.22

  # Single-dataset path only (skip two-dataset):
  python utils/ood_embedding_triptych_viz.py --exp-dir /path/to/run --single-dataset-path \\
    --no-two-dataset-path

  python utils/ood_embedding_triptych_viz.py \\
    --exp-dirs /path/run_a /path/run_b \\
    --out-dir /path/to/ood_viz_outputs

  # After a run, ``ood_embedding_all_plots_combined.pdf`` is written (all figures in order).
  # Re-merge without re-running the model:
  python utils/ood_embedding_triptych_viz.py --merge-pdfs-dir /path/to/plots_ood_triptych

  # One PDF for many experiment subfolders under a parent (e.g. pcafigs/run_a, run_b, ...):
  python utils/ood_embedding_triptych_viz.py --merge-pdfs-root /path/to/pcafigs
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.decomposition import PCA

_MODEL_ROOT = Path(__file__).resolve().parent.parent
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

from utils.alignment_multilayer_viz import (  # noqa: E402
    _l2_normalize,
    _normalize_dataset_id,
    _pad_to_dim,
    _to_arr,
    load_model,
    select_two_datasets,
)


# Substrings matched longest-first so e.g. "Facility" wins over "UCR" in composite ids.
_KNOWN_DOMAIN_LABELS: tuple[str, ...] = (
    "HumanActivity",
    "OPPORTUNITY",
    "WebService",
    "Exathlon",
    "Synthetic",
    "Environment",
    "Facility",
    "Medical",
    "Sensor",
    "Traffic",
    "Weather",
    "Finance",
    "LTDB",
    "MITDB",
    "SVDB",
    "MGAB",
    "TAO",
    "SMD",
    "WSD",
    "UCR",
)

# PDF filenames (single source of truth for merge + writers).
_PAIR_PDF_PRE = "ood_embedding_pre_chronos_and_desc_shared_scale.pdf"
_PAIR_PDF_POST = "ood_embedding_post_aligned_shared_scale.pdf"
_SINGLE_PATH_PRE = "ood_single_dataset_pre_ts_path_around_mean_desc.pdf"
_SINGLE_PATH_POST = "ood_single_dataset_post_ts_path_around_mean_desc.pdf"
_TWO_PATH_PRE = "ood_two_dataset_pre_ts_paths_around_mean_desc.pdf"
_TWO_PATH_POST = "ood_two_dataset_post_ts_paths_around_mean_desc.pdf"
_COMBINED_PDF_NAME = "ood_embedding_all_plots_combined.pdf"


def _domain_label_from_dataset_id(s: str) -> str:
    """Short domain name for legend (e.g. Facility, WebService, Environment)."""
    s = str(s).strip()
    sl = s.lower()
    for k in sorted(_KNOWN_DOMAIN_LABELS, key=len, reverse=True):
        if len(k) >= 3 and k.lower() in sl:
            return k
    if re.search(r"HumanA", s, re.I):
        return "HumanActivity"
    m = re.search(r"([A-Za-z][A-Za-z0-9]*)_tr(?:_|$)", s)
    if m:
        return m.group(1)
    m2 = re.search(r"[A-Za-z][A-Za-z0-9]{2,}", s)
    return m2.group(0) if m2 else "Unknown"


def _legend_labels_for_ood_pair(d_a: str, d_b: str) -> tuple[str, str, str, str]:
    """
    Legend strings: TS/Chronos as ``{domain}_series``; ST/aligned desc as ``{domain}_desc``.
    """
    dom_a = _domain_label_from_dataset_id(d_a)
    dom_b = _domain_label_from_dataset_id(d_b)
    ts_a = f"{dom_a}_series"
    ts_b = f"{dom_b}_series"
    desc_a = f"{dom_a}_desc"
    desc_b = f"{dom_b}_desc"
    return (ts_a, desc_a, ts_b, desc_b)


def select_diverse_dataset_pairs(
    df: pd.DataFrame,
    series_col: str,
    rng: np.random.Generator,
    num_pairs: int,
) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """
    Return up to ``num_pairs`` distinct unordered pairs of dataset ids (each with >=2 rows).
    Only pairs whose heuristic domain labels differ (e.g. not two ids both labeled ``Medical``) are allowed.
    Pairs are sampled uniformly without replacement from eligible cross-domain combinations, then shuffled.
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

    all_pairs = [
        (a, b)
        for a, b in combinations(eligible, 2)
        if _domain_label_from_dataset_id(a).lower() != _domain_label_from_dataset_id(b).lower()
    ]
    if not all_pairs:
        raise RuntimeError(
            f"No cross-domain pair of distinct {series_col!r} values with >=2 rows each "
            f"(heuristic domain labels all matched within-pair). Eligible ids: {eligible[:40]}"
        )
    rng.shuffle(all_pairs)
    k = min(num_pairs, len(all_pairs))
    if k < num_pairs:
        print(
            f"Note: requested {num_pairs} distinct cross-domain dataset pairs but only {len(all_pairs)} "
            f"exist; producing {k} pair(s)."
        )

    out: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for d_a, d_b in all_pairs[:k]:
        idx_a = np.array(groups[d_a][:2], dtype=np.int64)
        idx_b = np.array(groups[d_b][:2], dtype=np.int64)
        out.append((d_a, d_b, idx_a, idx_b))
    return out


def select_one_dataset_many_segments(
    df: pd.DataFrame,
    series_col: str,
    rng: np.random.Generator,
    max_segments: int,
    dataset_name: str | None = None,
) -> tuple[str, np.ndarray]:
    """
    Pick one dataset id with at least ``max_segments`` rows (or as many as available, min 3).
    Returns ``(normalized_dataset_id, row_indices)`` shuffled then truncated to ``max_segments``.
    """
    raw = df[series_col].astype(str).values
    norm = np.array([_normalize_dataset_id(x) for x in raw], dtype=object)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, n in enumerate(norm):
        groups[str(n)].append(i)

    if dataset_name is not None:
        key = _normalize_dataset_id(dataset_name)
        if key not in groups or len(groups[key]) < 3:
            raise ValueError(
                f"--path-dataset {dataset_name!r} (normalized {key!r}) needs >=3 rows in the parquet"
            )
        ix = np.array(groups[key], dtype=np.int64)
    else:
        eligible = [n for n, ix in groups.items() if len(ix) >= 3 and str(n).strip()]
        if not eligible:
            raise RuntimeError(f"No {series_col!r} with at least 3 rows")
        best = max(eligible, key=lambda n: len(groups[n]))
        ix = np.array(groups[best], dtype=np.int64)
        key = best

    rng.shuffle(ix)
    k = min(max_segments, len(ix))
    return key, ix[:k]


def select_two_datasets_many_segments(
    df: pd.DataFrame,
    series_col: str,
    rng: np.random.Generator,
    max_segments: int,
    dataset_a: str | None = None,
    dataset_b: str | None = None,
) -> tuple[str, np.ndarray, str, np.ndarray]:
    """
    Two **distinct** datasets, each contributing up to ``max_segments`` rows (min 3 each if available).
    """
    max_segments = max(3, int(max_segments))
    raw = df[series_col].astype(str).values
    norm = np.array([_normalize_dataset_id(x) for x in raw], dtype=object)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, n in enumerate(norm):
        groups[str(n)].append(i)

    if dataset_a is not None or dataset_b is not None:
        if dataset_a is None or dataset_b is None:
            raise ValueError("Pass both --path-dataset-a and --path-dataset-b for two-dataset path, or neither")
        na = _normalize_dataset_id(dataset_a)
        nb = _normalize_dataset_id(dataset_b)
        if na == nb:
            raise ValueError("--path-dataset-a and --path-dataset-b must name two different datasets")
        for key, name in ((na, dataset_a), (nb, dataset_b)):
            if key not in groups or len(groups[key]) < 3:
                raise ValueError(
                    f"Path dataset {name!r} (normalized {key!r}) needs at least three rows in the parquet"
                )
        d_a, d_b = na, nb
    else:
        eligible = sorted([n for n, ix in groups.items() if len(ix) >= 3 and str(n).strip()])
        if len(eligible) < 2:
            raise RuntimeError(
                "two_dataset_path: need at least two distinct datasets with >=3 rows each "
                f"(found {len(eligible)} eligible)"
            )
        cross_pairs = [
            (eligible[i], eligible[j])
            for i in range(len(eligible))
            for j in range(i + 1, len(eligible))
            if _domain_label_from_dataset_id(eligible[i]).lower()
            != _domain_label_from_dataset_id(eligible[j]).lower()
        ]
        if not cross_pairs:
            raise RuntimeError(
                "two_dataset_path: need two distinct datasets with >=3 rows each from different "
                f"heuristic domain labels; eligible ids: {eligible[:40]}"
            )
        rng.shuffle(cross_pairs)
        d_a, d_b = cross_pairs[0]

    ix_a = np.array(groups[d_a], dtype=np.int64)
    ix_b = np.array(groups[d_b], dtype=np.int64)
    rng.shuffle(ix_a)
    rng.shuffle(ix_b)
    k = max_segments
    return d_a, ix_a[: min(k, len(ix_a))], d_b, ix_b[: min(k, len(ix_b))]


def _angular_sort_closed_path(Z: np.ndarray, center: np.ndarray) -> np.ndarray:
    """
    Sort rows of Z (n,2) by polar angle around ``center`` (2,), return closed polyline (n+1,2).
    """
    Z = np.asarray(Z, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64).ravel()[:2]
    dx = Z[:, 0] - c[0]
    dy = Z[:, 1] - c[1]
    ang = np.arctan2(dy, dx)
    order = np.argsort(ang)
    path = Z[order]
    return np.vstack([path, path[0:1]])


def _fill_closed_path(
    ax: plt.Axes,
    path_xy: np.ndarray,
    *,
    facecolor: str,
    alpha: float,
    zorder: float = 1,
) -> None:
    """Semi-transparent fill inside the closed polygon (path includes closing vertex)."""
    path_xy = np.asarray(path_xy, dtype=np.float64)
    if path_xy.shape[0] < 3:
        return
    ax.fill(
        path_xy[:, 0],
        path_xy[:, 1],
        facecolor=facecolor,
        edgecolor="none",
        alpha=float(alpha),
        zorder=zorder,
    )


def _pair_subdir_name(idx: int, d_a: str, d_b: str) -> str:
    """Short, filesystem-safe subfolder: pair_00_nameA__vs__nameB."""

    def slug(s: str, max_len: int = 28) -> str:
        t = re.sub(r"[^\w\-]+", "_", str(s).strip(), flags=re.UNICODE)
        t = re.sub(r"_+", "_", t).strip("_")
        return (t[:max_len] or "ds").rstrip("_")

    return f"pair_{idx:02d}_{slug(d_a)}__vs__{slug(d_b)}"


# Matplotlib tick label size (px); x and y forced equal via ``tick_params(axis="both", ...)``.
_AXIS_TICK_LABELSIZE = 24
# Legend text size (pair PCA + path figures); keep readable vs. axis ticks.
_LEGEND_FONTSIZE = 22
# Line2D markers in PCA legends: scale with text so symbols match cap height (~0.6--0.7 em).
_LEGEND_LINE2D_MARKERSIZE = _LEGEND_FONTSIZE * 0.65
# Plot+scatter legends: inflate default handle markers vs. small fontsize baseline (~11--12 pt).
_LEGEND_MARKERSCALE = _LEGEND_FONTSIZE / 11.5


def _resolve_cfg_path(exp_dir: Path) -> Path:
    exp_dir = exp_dir.resolve()
    cfg_path = exp_dir / "config.yaml"
    if not cfg_path.is_file():
        hydra = exp_dir / ".hydra" / "config.yaml"
        if hydra.is_file():
            cfg_path = hydra
    if not cfg_path.is_file():
        raise FileNotFoundError(f"No config.yaml under {exp_dir}")
    return cfg_path


def _apply_triptych_style() -> None:
    ts = _AXIS_TICK_LABELSIZE
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": ts,
            "ytick.labelsize": ts,
            "legend.fontsize": _LEGEND_FONTSIZE,
        }
    )


def _set_equal_axis_tick_labels(ax: plt.Axes) -> None:
    """Force x and y tick label font sizes to match (major ticks)."""
    ax.tick_params(axis="both", which="major", labelsize=_AXIS_TICK_LABELSIZE)


def _square_limits_from_points(
    pts: np.ndarray, pad_frac: float = 0.08
) -> tuple[float, float, float, float]:
    """Return (xmin, xmax, ymin, ymax) with equal x/y half-range for aspect='equal'."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0:
        return -1.0, 1.0, -1.0, 1.0
    xmin, xmax = float(pts[:, 0].min()), float(pts[:, 0].max())
    ymin, ymax = float(pts[:, 1].min()), float(pts[:, 1].max())
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half_x = 0.5 * (xmax - xmin) or 0.5
    half_y = 0.5 * (ymax - ymin) or 0.5
    half = max(half_x, half_y) * (1.0 + 2.0 * pad_frac)
    return cx - half, cx + half, cy - half, cy + half


def _scatter_pre_ts_and_desc_on_one_ax(
    ax: plt.Axes,
    Z_ch: np.ndarray,
    Z_st: np.ndarray,
    *,
    ds_ids: list[int],
    color_a: str,
    color_b: str,
    leg: tuple[str, str, str, str],
    size_ts: float = 120,
    size_desc: float = 130,
) -> None:
    """Chronos (circles) and ST desc (triangles) on the same axes; ds_ids length 4."""
    for i in range(4):
        c = color_a if ds_ids[i] == 0 else color_b
        ax.scatter(
            Z_ch[i, 0],
            Z_ch[i, 1],
            c=c,
            s=size_ts,
            marker="o",
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )
    for i in range(4):
        c = color_a if ds_ids[i] == 0 else color_b
        ax.scatter(
            Z_st[i, 0],
            Z_st[i, 1],
            c=c,
            s=size_desc,
            marker="^",
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )
    _mk = _LEGEND_LINE2D_MARKERSIZE
    ax.legend(
        [
            plt.Line2D([0], [0], linestyle="none", marker="o", color="w", markerfacecolor=color_a, markersize=_mk),
            plt.Line2D([0], [0], linestyle="none", marker="^", color="w", markerfacecolor=color_a, markersize=_mk),
            plt.Line2D([0], [0], linestyle="none", marker="o", color="w", markerfacecolor=color_b, markersize=_mk),
            plt.Line2D([0], [0], linestyle="none", marker="^", color="w", markerfacecolor=color_b, markersize=_mk),
        ],
        [leg[0], leg[1], leg[2], leg[3]],
        loc="best",
        frameon=False,
        ncol=1,
        fontsize=_LEGEND_FONTSIZE,
    )


def _scatter_post_aligned_on_ax(
    ax: plt.Axes,
    Z_post: np.ndarray,
    *,
    ds_ids: list[int],
    color_a: str,
    color_b: str,
    leg: tuple[str, str, str, str],
) -> None:
    """Aligned TS (rows 0–3) and aligned desc (rows 4–7)."""
    for i in range(4):
        c = color_a if ds_ids[i] == 0 else color_b
        ax.scatter(
            Z_post[i, 0],
            Z_post[i, 1],
            c=c,
            s=120,
            marker="o",
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )
    for i in range(4):
        c = color_a if ds_ids[i] == 0 else color_b
        ax.scatter(
            Z_post[i + 4, 0],
            Z_post[i + 4, 1],
            c=c,
            s=130,
            marker="^",
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )
    _mk = _LEGEND_LINE2D_MARKERSIZE
    ax.legend(
        [
            plt.Line2D([0], [0], linestyle="none", marker="o", color="w", markerfacecolor=color_a, markersize=_mk),
            plt.Line2D([0], [0], linestyle="none", marker="^", color="w", markerfacecolor=color_a, markersize=_mk),
            plt.Line2D([0], [0], linestyle="none", marker="o", color="w", markerfacecolor=color_b, markersize=_mk),
            plt.Line2D([0], [0], linestyle="none", marker="^", color="w", markerfacecolor=color_b, markersize=_mk),
        ],
        [leg[0], leg[1], leg[2], leg[3]],
        loc="best",
        frameon=False,
        ncol=1,
        fontsize=_LEGEND_FONTSIZE,
    )


def _run_one_pair_ood_embedding(
    *,
    exp_dir: Path,
    cfg: dict,
    parquet_path: Path,
    df: pd.DataFrame,
    model: torch.nn.Module,
    device: torch.device,
    ts_col: str,
    desc_col: str,
    series_col: str,
    desc_text_col: str | None,
    ts_dim: int,
    desc_dim: int,
    proj_dim: int,
    common_dim: int,
    d_a: str,
    d_b: str,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    pair_out_dir: Path,
    pca_random_seed: int,
    pair_index: int,
    desc_max_chars_meta: int,
) -> dict:
    segment_indices = [int(idx_a[0]), int(idx_a[1]), int(idx_b[0]), int(idx_b[1])]
    ds_ids = [0, 0, 1, 1]

    chronos_batch: list[torch.Tensor] = []
    desc_batch: list[torch.Tensor] = []
    rows_ch: list[np.ndarray] = []
    rows_de: list[np.ndarray] = []
    for ix in segment_indices:
        ch = np.nan_to_num(_to_arr(df.iloc[ix][ts_col], ts_dim), nan=0.0, posinf=0.0, neginf=0.0)
        de = np.nan_to_num(_to_arr(df.iloc[ix][desc_col], desc_dim), nan=0.0, posinf=0.0, neginf=0.0)
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
    X_pre_j = np.stack([_pad_to_dim(X_pre[i], joint_dim) for i in range(8)], axis=0).astype(
        np.float32
    )
    X_post_j = np.stack([_pad_to_dim(X_post[i], joint_dim) for i in range(8)], axis=0).astype(
        np.float32
    )

    n_comp = min(2, X_pre_j.shape[0] - 1, X_pre_j.shape[1])
    pca = PCA(n_components=n_comp, random_state=pca_random_seed)
    pca.fit(X_pre_j)
    Z_pre = pca.transform(X_pre_j)
    Z_post = pca.transform(X_post_j)

    Z_ch = Z_pre[:4]
    Z_st = Z_pre[4:8]
    Z_pts_all = np.vstack([Z_pre, Z_post])
    x0, x1, y0, y1 = _square_limits_from_points(Z_pts_all)

    color_a, color_b = "#1f77b4", "#ff7f0e"
    leg = _legend_labels_for_ood_pair(d_a, d_b)

    pair_out_dir.mkdir(parents=True, exist_ok=True)
    pdf_pre = pair_out_dir / _PAIR_PDF_PRE
    pdf_post = pair_out_dir / _PAIR_PDF_POST
    meta_path = pair_out_dir / "ood_embedding_figures_meta.json"

    desc_a_txt = ""
    desc_b_txt = ""
    if desc_text_col and str(desc_text_col) in df.columns:
        r_a = df[df[series_col].map(_normalize_dataset_id) == d_a].iloc[0]
        r_b = df[df[series_col].map(_normalize_dataset_id) == d_b].iloc[0]
        desc_a_txt = str(r_a[desc_text_col])
        desc_b_txt = str(r_b[desc_text_col])

    _apply_triptych_style()

    def _style_axis(ax: plt.Axes) -> None:
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal", adjustable="box")
        _set_equal_axis_tick_labels(ax)

    fig1, ax1 = plt.subplots(figsize=(9, 7))
    _style_axis(ax1)
    _scatter_pre_ts_and_desc_on_one_ax(ax1, Z_ch, Z_st, ds_ids=ds_ids, color_a=color_a, color_b=color_b, leg=leg)
    fig1.tight_layout()
    fig1.savefig(pdf_pre, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(9, 7))
    _style_axis(ax2)
    _scatter_post_aligned_on_ax(ax2, Z_post, ds_ids=ds_ids, color_a=color_a, color_b=color_b, leg=leg)
    fig2.tight_layout()
    fig2.savefig(pdf_post, bbox_inches="tight")
    plt.close(fig2)

    def _trunc_meta(s: str, max_len: int) -> str:
        if max_len <= 0 or len(s) <= max_len:
            return s
        return s[:max_len] + "…"

    meta = {
        "exp_dir": str(exp_dir),
        "pair_index": pair_index,
        "parquet_file": str(parquet_path),
        "ts_emb_col": ts_col,
        "desc_emb_col": desc_col,
        "series_col": series_col,
        "desc_text_col": desc_text_col,
        "joint_pca_dim": joint_dim,
        "pca_mode": "fit_pre_joint_basis",
        "projection_dim": proj_dim,
        "datasets_normalized": [d_a, d_b],
        "legend_labels": list(leg),
        "legend_labels_short": list(leg),
        "segment_row_indices": segment_indices,
        "axis_limits_shared_both_figures": {"xmin": x0, "xmax": x1, "ymin": y0, "ymax": y1},
        "dataset_descriptions": {
            "a": _trunc_meta(desc_a_txt, desc_max_chars_meta),
            "b": _trunc_meta(desc_b_txt, desc_max_chars_meta),
        },
        "pdfs": {"pre_chronos_and_desc": str(pdf_pre), "post_aligned": str(pdf_post)},
        "pair_out_dir": str(pair_out_dir),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote: {pdf_pre}")
    print(f"Wrote: {pdf_post}")
    print(f"Wrote: {meta_path}")
    return meta


def _run_single_dataset_path_viz(
    *,
    exp_dir: Path,
    parquet_path: Path,
    df: pd.DataFrame,
    model: torch.nn.Module,
    device: torch.device,
    ts_col: str,
    desc_col: str,
    series_col: str,
    ts_dim: int,
    desc_dim: int,
    proj_dim: int,
    common_dim: int,
    out_dir: Path,
    rng: np.random.Generator,
    pca_random_seed: int,
    path_max_segments: int,
    path_dataset: str | None,
    path_fill_alpha: float = 0.22,
) -> dict | None:
    """
    One dataset, many segments: TS embeddings connected in angular order around the **mean**
    description point (fixed triangle) to show a closed path; same PCA basis as fit on stacked
    Chronos+desc rows. Post panel uses the same PCA on aligned stacks; mean aligned desc is the
    fixed anchor. Polygon interior is filled with a transparent color.
    """
    path_max_segments = max(3, int(path_max_segments))
    d_name, row_ix = select_one_dataset_many_segments(
        df, series_col, rng, path_max_segments, dataset_name=path_dataset
    )
    k = len(row_ix)
    if k < 3:
        print(f"single_dataset_path: skip (only {k} rows for {d_name!r})")
        return None

    chronos_batch: list[torch.Tensor] = []
    desc_batch: list[torch.Tensor] = []
    rows_ch: list[np.ndarray] = []
    rows_de: list[np.ndarray] = []
    desc_raw: list[np.ndarray] = []
    for ix in row_ix:
        ch = np.nan_to_num(_to_arr(df.iloc[int(ix)][ts_col], ts_dim), nan=0.0, posinf=0.0, neginf=0.0)
        de = np.nan_to_num(_to_arr(df.iloc[int(ix)][desc_col], desc_dim), nan=0.0, posinf=0.0, neginf=0.0)
        desc_raw.append(de)
        ch_n = _l2_normalize(ch)
        de_n = _l2_normalize(de)
        rows_ch.append(_pad_to_dim(ch_n, common_dim))
        rows_de.append(_pad_to_dim(de_n, common_dim))
        chronos_batch.append(torch.tensor(ch, device=device).unsqueeze(0))
        desc_batch.append(torch.tensor(de, device=device).unsqueeze(0))

    mean_desc_raw = np.mean(np.stack(desc_raw, axis=0), axis=0).astype(np.float32)
    mean_desc_n = _l2_normalize(mean_desc_raw)

    X_pre = np.stack(rows_ch + rows_de, axis=0).astype(np.float32)

    aligned_ts: list[np.ndarray] = []
    aligned_de: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(k):
            ts_p, desc_p = model(chronos_batch[i], desc_batch[i])
            aligned_ts.append(ts_p.cpu().numpy().ravel()[:proj_dim])
            aligned_de.append(desc_p.cpu().numpy().ravel()[:proj_dim])

    X_post = np.stack(aligned_ts + aligned_de, axis=0).astype(np.float32)
    mean_aligned_desc = np.mean(np.stack(aligned_de, axis=0), axis=0).astype(np.float32)

    joint_dim = max(common_dim, proj_dim)
    X_pre_j = np.stack([_pad_to_dim(X_pre[i], joint_dim) for i in range(2 * k)], axis=0).astype(np.float32)
    X_post_j = np.stack([_pad_to_dim(X_post[i], joint_dim) for i in range(2 * k)], axis=0).astype(np.float32)

    n_comp = min(2, X_pre_j.shape[0] - 1, X_pre_j.shape[1])
    pca = PCA(n_components=n_comp, random_state=pca_random_seed)
    pca.fit(X_pre_j)
    Z_pre = pca.transform(X_pre_j)
    Z_post = pca.transform(X_post_j)

    Z_ts_pre = Z_pre[:k]
    mean_desc_pre_j = _pad_to_dim(_pad_to_dim(mean_desc_n, common_dim), joint_dim)
    Z_desc_mean_pre = pca.transform(mean_desc_pre_j.reshape(1, -1))[0, :2]

    Z_ts_post = Z_post[:k]
    mean_desc_post_j = _pad_to_dim(mean_aligned_desc, joint_dim)
    Z_desc_mean_post = pca.transform(mean_desc_post_j.reshape(1, -1))[0, :2]

    path_pre = _angular_sort_closed_path(Z_ts_pre, Z_desc_mean_pre)
    path_post = _angular_sort_closed_path(Z_ts_post, Z_desc_mean_post)

    pts = np.vstack(
        [
            path_pre,
            path_post,
            Z_desc_mean_pre.reshape(1, 2),
            Z_desc_mean_post.reshape(1, 2),
        ]
    )
    x0, x1, y0, y1 = _square_limits_from_points(pts)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_pre = out_dir / _SINGLE_PATH_PRE
    pdf_post = out_dir / _SINGLE_PATH_POST
    meta_path = out_dir / "ood_single_dataset_path_meta.json"

    color_path = "#1f77b4"
    color_desc = "#ff7f0e"

    _apply_triptych_style()

    def _style_axis(ax: plt.Axes) -> None:
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal", adjustable="box")
        _set_equal_axis_tick_labels(ax)

    fig1, ax1 = plt.subplots(figsize=(9, 7))
    _style_axis(ax1)
    _fill_closed_path(ax1, path_pre, facecolor=color_path, alpha=path_fill_alpha, zorder=1)
    ax1.plot(
        path_pre[:, 0],
        path_pre[:, 1],
        color=color_path,
        lw=2.8,
        alpha=0.9,
        solid_capstyle="round",
        zorder=2,
        label="TS path (angular order)",
    )
    ax1.scatter(
        Z_desc_mean_pre[0],
        Z_desc_mean_pre[1],
        c=color_desc,
        s=420,
        marker="^",
        edgecolors="black",
        linewidths=1.0,
        zorder=5,
        label="Mean ST desc (fixed)",
    )
    ax1.legend(
        loc="best",
        frameon=False,
        fontsize=_LEGEND_FONTSIZE,
        markerscale=_LEGEND_MARKERSCALE,
    )
    fig1.tight_layout()
    fig1.savefig(pdf_pre, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(9, 7))
    _style_axis(ax2)
    _fill_closed_path(ax2, path_post, facecolor=color_path, alpha=path_fill_alpha, zorder=1)
    ax2.plot(
        path_post[:, 0],
        path_post[:, 1],
        color=color_path,
        lw=2.8,
        alpha=0.9,
        solid_capstyle="round",
        zorder=2,
        label="Aligned TS path",
    )
    ax2.scatter(
        Z_desc_mean_post[0],
        Z_desc_mean_post[1],
        c=color_desc,
        s=420,
        marker="^",
        edgecolors="black",
        linewidths=1.0,
        zorder=5,
        label="Mean aligned desc (fixed)",
    )
    ax2.legend(
        loc="best",
        frameon=False,
        fontsize=_LEGEND_FONTSIZE,
        markerscale=_LEGEND_MARKERSCALE,
    )
    fig2.tight_layout()
    fig2.savefig(pdf_post, bbox_inches="tight")
    plt.close(fig2)

    meta = {
        "exp_dir": str(exp_dir),
        "parquet_file": str(parquet_path),
        "dataset_normalized": d_name,
        "num_segments": int(k),
        "row_indices": [int(x) for x in row_ix],
        "path_max_segments": path_max_segments,
        "pca_mode": "fit_pre_joint_basis",
        "joint_pca_dim": joint_dim,
        "pdfs": {"pre_path": str(pdf_pre), "post_path": str(pdf_post)},
        "axis_limits_shared_both_figures": {"xmin": x0, "xmax": x1, "ymin": y0, "ymax": y1},
        "path_fill_alpha": path_fill_alpha,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote: {pdf_pre}")
    print(f"Wrote: {pdf_post}")
    print(f"Wrote: {meta_path}")
    return meta


def _run_two_dataset_path_viz(
    *,
    exp_dir: Path,
    parquet_path: Path,
    df: pd.DataFrame,
    model: torch.nn.Module,
    device: torch.device,
    ts_col: str,
    desc_col: str,
    series_col: str,
    ts_dim: int,
    desc_dim: int,
    proj_dim: int,
    common_dim: int,
    out_dir: Path,
    rng: np.random.Generator,
    pca_random_seed: int,
    path_max_segments: int,
    path_dataset_a: str | None,
    path_dataset_b: str | None,
    path_fill_alpha: float = 0.22,
) -> dict | None:
    """
    Two datasets: each has a closed TS path (angular order around its mean ST desc) and a fixed
    mean-description triangle; both in one PCA (fit on all Chronos+desc rows). Filled polygons.
    """
    try:
        d_a, ix_a, d_b, ix_b = select_two_datasets_many_segments(
            df, series_col, rng, path_max_segments, dataset_a=path_dataset_a, dataset_b=path_dataset_b
        )
    except (RuntimeError, ValueError) as e:
        print(f"two_dataset_path: skipped ({e})")
        return None

    k_a, k_b = len(ix_a), len(ix_b)
    if k_a < 3 or k_b < 3:
        print(f"two_dataset_path: skip (k_a={k_a}, k_b={k_b})")
        return None

    def _collect(row_indices: np.ndarray) -> tuple[list, list, list, list, list]:
        ch_list: list[torch.Tensor] = []
        d_list: list[torch.Tensor] = []
        r_ch: list[np.ndarray] = []
        r_de: list[np.ndarray] = []
        d_raw: list[np.ndarray] = []
        for row_idx in row_indices:
            ch = np.nan_to_num(
                _to_arr(df.iloc[int(row_idx)][ts_col], ts_dim), nan=0.0, posinf=0.0, neginf=0.0
            )
            de = np.nan_to_num(
                _to_arr(df.iloc[int(row_idx)][desc_col], desc_dim), nan=0.0, posinf=0.0, neginf=0.0
            )
            d_raw.append(de)
            ch_n = _l2_normalize(ch)
            de_n = _l2_normalize(de)
            r_ch.append(_pad_to_dim(ch_n, common_dim))
            r_de.append(_pad_to_dim(de_n, common_dim))
            ch_list.append(torch.tensor(ch, device=device).unsqueeze(0))
            d_list.append(torch.tensor(de, device=device).unsqueeze(0))
        return ch_list, d_list, r_ch, r_de, d_raw

    ch_a, db_a, rows_ch_a, rows_de_a, raw_a = _collect(ix_a)
    ch_b, db_b, rows_ch_b, rows_de_b, raw_b = _collect(ix_b)

    mean_desc_raw_a = np.mean(np.stack(raw_a, axis=0), axis=0).astype(np.float32)
    mean_desc_raw_b = np.mean(np.stack(raw_b, axis=0), axis=0).astype(np.float32)
    mean_desc_na = _l2_normalize(mean_desc_raw_a)
    mean_desc_nb = _l2_normalize(mean_desc_raw_b)

    rows_ch = rows_ch_a + rows_ch_b
    rows_de = rows_de_a + rows_de_b
    X_pre = np.stack(rows_ch + rows_de, axis=0).astype(np.float32)

    chronos_batch = ch_a + ch_b
    desc_batch = db_a + db_b
    n_tot = k_a + k_b

    aligned_ts: list[np.ndarray] = []
    aligned_de: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(n_tot):
            ts_p, desc_p = model(chronos_batch[i], desc_batch[i])
            aligned_ts.append(ts_p.cpu().numpy().ravel()[:proj_dim])
            aligned_de.append(desc_p.cpu().numpy().ravel()[:proj_dim])

    X_post = np.stack(aligned_ts + aligned_de, axis=0).astype(np.float32)
    aligned_de_a = aligned_de[:k_a]
    aligned_de_b = aligned_de[k_a : k_a + k_b]
    mean_aligned_a = np.mean(np.stack(aligned_de_a, axis=0), axis=0).astype(np.float32)
    mean_aligned_b = np.mean(np.stack(aligned_de_b, axis=0), axis=0).astype(np.float32)

    joint_dim = max(common_dim, proj_dim)
    X_pre_j = np.stack([_pad_to_dim(X_pre[i], joint_dim) for i in range(2 * n_tot)], axis=0).astype(np.float32)
    X_post_j = np.stack([_pad_to_dim(X_post[i], joint_dim) for i in range(2 * n_tot)], axis=0).astype(np.float32)

    n_comp = min(2, X_pre_j.shape[0] - 1, X_pre_j.shape[1])
    pca = PCA(n_components=n_comp, random_state=pca_random_seed)
    pca.fit(X_pre_j)
    Z_pre = pca.transform(X_pre_j)
    Z_post = pca.transform(X_post_j)

    Z_ts_a = Z_pre[0:k_a]
    Z_ts_b = Z_pre[k_a : k_a + k_b]
    mean_a_j = _pad_to_dim(_pad_to_dim(mean_desc_na, common_dim), joint_dim)
    mean_b_j = _pad_to_dim(_pad_to_dim(mean_desc_nb, common_dim), joint_dim)
    Z_desc_mean_a_pre = pca.transform(mean_a_j.reshape(1, -1))[0, :2]
    Z_desc_mean_b_pre = pca.transform(mean_b_j.reshape(1, -1))[0, :2]

    Z_ts_post_a = Z_post[0:k_a]
    Z_ts_post_b = Z_post[k_a : k_a + k_b]
    Z_desc_mean_a_post = pca.transform(_pad_to_dim(mean_aligned_a, joint_dim).reshape(1, -1))[0, :2]
    Z_desc_mean_b_post = pca.transform(_pad_to_dim(mean_aligned_b, joint_dim).reshape(1, -1))[0, :2]

    path_pre_a = _angular_sort_closed_path(Z_ts_a, Z_desc_mean_a_pre)
    path_pre_b = _angular_sort_closed_path(Z_ts_b, Z_desc_mean_b_pre)
    path_post_a = _angular_sort_closed_path(Z_ts_post_a, Z_desc_mean_a_post)
    path_post_b = _angular_sort_closed_path(Z_ts_post_b, Z_desc_mean_b_post)

    pts = np.vstack(
        [
            path_pre_a,
            path_pre_b,
            path_post_a,
            path_post_b,
            Z_desc_mean_a_pre.reshape(1, 2),
            Z_desc_mean_b_pre.reshape(1, 2),
            Z_desc_mean_a_post.reshape(1, 2),
            Z_desc_mean_b_post.reshape(1, 2),
        ]
    )
    x0, x1, y0, y1 = _square_limits_from_points(pts)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_pre = out_dir / _TWO_PATH_PRE
    pdf_post = out_dir / _TWO_PATH_POST
    meta_path = out_dir / "ood_two_dataset_path_meta.json"

    c_a, c_b = "#1f77b4", "#ff7f0e"

    _apply_triptych_style()

    def _style_axis(ax: plt.Axes) -> None:
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal", adjustable="box")
        _set_equal_axis_tick_labels(ax)

    fig1, ax1 = plt.subplots(figsize=(9, 7))
    _style_axis(ax1)
    _fill_closed_path(ax1, path_pre_a, facecolor=c_a, alpha=path_fill_alpha, zorder=1)
    _fill_closed_path(ax1, path_pre_b, facecolor=c_b, alpha=path_fill_alpha, zorder=1)
    ax1.plot(path_pre_a[:, 0], path_pre_a[:, 1], color=c_a, lw=2.6, alpha=0.95, zorder=2, label="Dataset A TS path")
    ax1.plot(path_pre_b[:, 0], path_pre_b[:, 1], color=c_b, lw=2.6, alpha=0.95, zorder=2, label="Dataset B TS path")
    ax1.scatter(
        Z_desc_mean_a_pre[0],
        Z_desc_mean_a_pre[1],
        c=c_a,
        s=380,
        marker="^",
        edgecolors="black",
        linewidths=1.0,
        zorder=5,
        label="Mean ST desc A",
    )
    ax1.scatter(
        Z_desc_mean_b_pre[0],
        Z_desc_mean_b_pre[1],
        c=c_b,
        s=380,
        marker="^",
        edgecolors="black",
        linewidths=1.0,
        zorder=5,
        label="Mean ST desc B",
    )
    ax1.legend(
        loc="best",
        frameon=False,
        fontsize=_LEGEND_FONTSIZE,
        markerscale=_LEGEND_MARKERSCALE,
    )
    fig1.tight_layout()
    fig1.savefig(pdf_pre, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(9, 7))
    _style_axis(ax2)
    _fill_closed_path(ax2, path_post_a, facecolor=c_a, alpha=path_fill_alpha, zorder=1)
    _fill_closed_path(ax2, path_post_b, facecolor=c_b, alpha=path_fill_alpha, zorder=1)
    ax2.plot(path_post_a[:, 0], path_post_a[:, 1], color=c_a, lw=2.6, alpha=0.95, zorder=2, label="Aligned TS A")
    ax2.plot(path_post_b[:, 0], path_post_b[:, 1], color=c_b, lw=2.6, alpha=0.95, zorder=2, label="Aligned TS B")
    ax2.scatter(
        Z_desc_mean_a_post[0],
        Z_desc_mean_a_post[1],
        c=c_a,
        s=380,
        marker="^",
        edgecolors="black",
        linewidths=1.0,
        zorder=5,
        label="Mean aligned desc A",
    )
    ax2.scatter(
        Z_desc_mean_b_post[0],
        Z_desc_mean_b_post[1],
        c=c_b,
        s=380,
        marker="^",
        edgecolors="black",
        linewidths=1.0,
        zorder=5,
        label="Mean aligned desc B",
    )
    ax2.legend(
        loc="best",
        frameon=False,
        fontsize=_LEGEND_FONTSIZE,
        markerscale=_LEGEND_MARKERSCALE,
    )
    fig2.tight_layout()
    fig2.savefig(pdf_post, bbox_inches="tight")
    plt.close(fig2)

    meta = {
        "exp_dir": str(exp_dir),
        "parquet_file": str(parquet_path),
        "datasets_normalized": [d_a, d_b],
        "num_segments": [int(k_a), int(k_b)],
        "row_indices_a": [int(x) for x in ix_a],
        "row_indices_b": [int(x) for x in ix_b],
        "path_max_segments": path_max_segments,
        "path_fill_alpha": path_fill_alpha,
        "pca_mode": "fit_pre_joint_basis",
        "joint_pca_dim": joint_dim,
        "pdfs": {"pre_path": str(pdf_pre), "post_path": str(pdf_post)},
        "axis_limits_shared_both_figures": {"xmin": x0, "xmax": x1, "ymin": y0, "ymax": y1},
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote: {pdf_pre}")
    print(f"Wrote: {pdf_post}")
    print(f"Wrote: {meta_path}")
    return meta


def collect_ood_triptych_pdf_paths(base_out: Path) -> list[Path]:
    """
    Ordered PDF paths for one OOD triptych output directory: pair pre/post (per pair subdir),
    then single-dataset path pre/post (if present), then two-dataset path pre/post (if present).
    """
    base_out = base_out.resolve()
    if not base_out.is_dir():
        return []
    out: list[Path] = []
    if (base_out / _PAIR_PDF_PRE).is_file():
        out.append(base_out / _PAIR_PDF_PRE)
        out.append(base_out / _PAIR_PDF_POST)
    else:
        pair_dirs = sorted(
            [p for p in base_out.iterdir() if p.is_dir() and p.name.startswith("pair_")],
            key=lambda p: p.name,
        )
        for pd in pair_dirs:
            pre_p = pd / _PAIR_PDF_PRE
            post_p = pd / _PAIR_PDF_POST
            if pre_p.is_file():
                out.append(pre_p)
            if post_p.is_file():
                out.append(post_p)
    single_dir = base_out / "single_dataset_path"
    for name in (_SINGLE_PATH_PRE, _SINGLE_PATH_POST):
        p = single_dir / name
        if p.is_file():
            out.append(p)
    two_dir = base_out / "two_dataset_path"
    for name in (_TWO_PATH_PRE, _TWO_PATH_POST):
        p = two_dir / name
        if p.is_file():
            out.append(p)
    return out


def _merge_pdf_paths(pdf_paths: list[Path], out_path: Path) -> None:
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError as e:
        raise ImportError(
            "Merging PDFs requires the 'pypdf' package. Install with: pip install pypdf"
        ) from e

    writer = PdfWriter()
    for p in pdf_paths:
        reader = PdfReader(str(p))
        for page in reader.pages:
            writer.add_page(page)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        writer.write(f)


def merge_ood_triptych_pdfs(
    base_out: Path,
    out_name: str = _COMBINED_PDF_NAME,
) -> Path | None:
    """
    Concatenate all OOD triptych PDFs under ``base_out`` into one multi-page PDF.
    Returns the output path, or ``None`` if nothing to merge.
    """
    paths = collect_ood_triptych_pdf_paths(base_out)
    if not paths:
        return None
    out_path = base_out.resolve() / out_name
    try:
        _merge_pdf_paths(paths, out_path)
    except ImportError as e:
        print(f"Warning: skipped combined PDF ({e})", file=sys.stderr)
        return None
    print(f"Wrote combined PDF ({len(paths)} files): {out_path}")
    return out_path


def _looks_like_ood_triptych_out(d: Path) -> bool:
    if (d / _PAIR_PDF_PRE).is_file():
        return True
    try:
        return any(
            p.is_dir() and p.name.startswith("pair_") for p in d.iterdir()
        )
    except OSError:
        return False


def merge_ood_triptych_pdfs_under_root(
    root: Path,
    out_name: str = "ood_triptych_all_runs_combined.pdf",
) -> Path | None:
    """
    For each immediate subdirectory of ``root`` that looks like an OOD triptych run output,
    append that run's PDF sequence (same order as :func:`collect_ood_triptych_pdf_paths`)
    into one multi-page PDF written under ``root``.
    """
    root = root.resolve()
    if not root.is_dir():
        return None
    subdirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
    all_paths: list[Path] = []
    used = 0
    for d in subdirs:
        if not _looks_like_ood_triptych_out(d):
            continue
        seq = collect_ood_triptych_pdf_paths(d)
        if seq:
            used += 1
            all_paths.extend(seq)
    if not all_paths:
        return None
    out_path = root / out_name
    try:
        _merge_pdf_paths(all_paths, out_path)
    except ImportError as e:
        print(f"Warning: skipped combined PDF ({e})", file=sys.stderr)
        return None
    print(f"Wrote combined PDF ({used} runs, {len(all_paths)} figure PDFs): {out_path}")
    return out_path


def run_ood_embedding_triptych_viz(
    exp_dir: Path,
    out_dir: Path | None,
    device_str: str,
    random_seed: int,
    common_dim: int | None = None,
    dataset_a: str | None = None,
    dataset_b: str | None = None,
    desc_max_chars: int = 900,
    num_pair_iterations: int = 1,
    single_dataset_path: bool = False,
    two_dataset_path: bool = True,
    path_max_segments: int = 48,
    path_dataset: str | None = None,
    path_dataset_a: str | None = None,
    path_dataset_b: str | None = None,
    path_fill_alpha: float = 0.22,
    combine_pdf: bool = True,
) -> dict | list[dict]:
    exp_dir = exp_dir.resolve()
    cfg_path = _resolve_cfg_path(exp_dir)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("model", {}).get("align_to_performance"):
        raise ValueError("align_to_performance=True not supported for this viz")

    if num_pair_iterations < 1:
        raise ValueError("num_pair_iterations must be >= 1")
    if num_pair_iterations > 1 and (dataset_a is not None or dataset_b is not None):
        raise ValueError(
            "Use --num-pair-iterations 1 when passing --dataset-a and --dataset-b (fixed pair only)"
        )

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
    desc_text_col = data.get("desc_text_col")

    if common_dim is None:
        common_dim = max(ts_dim, desc_dim)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, ckpt, device)

    need_cols = [ts_col, desc_col, series_col, "series"]
    if desc_text_col:
        need_cols.append(str(desc_text_col))
    df = pd.read_parquet(parquet_path, columns=need_cols)

    rng = np.random.default_rng(random_seed)

    if num_pair_iterations == 1:
        pairs: list[tuple[str, str, np.ndarray, np.ndarray]] = [
            select_two_datasets(
                df,
                series_col,
                rng,
                dataset_a=dataset_a,
                dataset_b=dataset_b,
                require_cross_domain=True,
            )
        ]
    else:
        pairs = select_diverse_dataset_pairs(df, series_col, rng, num_pair_iterations)

    base_out = Path(out_dir or (exp_dir / "plots_ood_triptych"))
    base_out.mkdir(parents=True, exist_ok=True)

    metas: list[dict] = []
    for pi, (d_a, d_b, idx_a, idx_b) in enumerate(pairs):
        if len(pairs) == 1:
            pair_out = base_out
        else:
            pair_out = base_out / _pair_subdir_name(pi, d_a, d_b)
        m = _run_one_pair_ood_embedding(
            exp_dir=exp_dir,
            cfg=cfg,
            parquet_path=parquet_path,
            df=df,
            model=model,
            device=device,
            ts_col=ts_col,
            desc_col=desc_col,
            series_col=series_col,
            desc_text_col=desc_text_col,
            ts_dim=ts_dim,
            desc_dim=desc_dim,
            proj_dim=proj_dim,
            common_dim=common_dim,
            d_a=d_a,
            d_b=d_b,
            idx_a=idx_a,
            idx_b=idx_b,
            pair_out_dir=pair_out,
            pca_random_seed=random_seed + pi,
            pair_index=pi,
            desc_max_chars_meta=desc_max_chars,
        )
        metas.append(m)

    single_path_meta: dict | None = None
    two_path_meta: dict | None = None
    if single_dataset_path:
        single_path_meta = _run_single_dataset_path_viz(
            exp_dir=exp_dir,
            parquet_path=parquet_path,
            df=df,
            model=model,
            device=device,
            ts_col=ts_col,
            desc_col=desc_col,
            series_col=series_col,
            ts_dim=ts_dim,
            desc_dim=desc_dim,
            proj_dim=proj_dim,
            common_dim=common_dim,
            out_dir=base_out / "single_dataset_path",
            rng=rng,
            pca_random_seed=random_seed + 10000,
            path_max_segments=path_max_segments,
            path_dataset=path_dataset,
            path_fill_alpha=path_fill_alpha,
        )
        if two_dataset_path:
            two_path_meta = _run_two_dataset_path_viz(
                exp_dir=exp_dir,
                parquet_path=parquet_path,
                df=df,
                model=model,
                device=device,
                ts_col=ts_col,
                desc_col=desc_col,
                series_col=series_col,
                ts_dim=ts_dim,
                desc_dim=desc_dim,
                proj_dim=proj_dim,
                common_dim=common_dim,
                out_dir=base_out / "two_dataset_path",
                rng=rng,
                pca_random_seed=random_seed + 10001,
                path_max_segments=path_max_segments,
                path_dataset_a=path_dataset_a,
                path_dataset_b=path_dataset_b,
                path_fill_alpha=path_fill_alpha,
            )

    if len(metas) > 1:
        summary_path = base_out / "ood_embedding_all_pairs_summary.json"
        summary_obj: dict = {
            "exp_dir": str(exp_dir),
            "num_pairs": len(metas),
            "pairs": [
                {
                    "pair_index": i,
                    "datasets": [m["datasets_normalized"][0], m["datasets_normalized"][1]],
                    "out_dir": m.get("pair_out_dir"),
                    "meta": m.get("pdfs"),
                }
                for i, m in enumerate(metas)
            ],
        }
        if single_path_meta is not None:
            summary_obj["single_dataset_path"] = single_path_meta
        if two_path_meta is not None:
            summary_obj["two_dataset_path"] = two_path_meta
        summary_path.write_text(json.dumps(summary_obj, indent=2), encoding="utf-8")
        print(f"Wrote: {summary_path}")
        if combine_pdf:
            merge_ood_triptych_pdfs(base_out)
        return metas
    if single_path_meta is not None or two_path_meta is not None:
        extra: dict = {}
        if single_path_meta is not None:
            extra["single_dataset_path"] = single_path_meta
        if two_path_meta is not None:
            extra["two_dataset_path"] = two_path_meta
        metas[0] = {**metas[0], **extra}
    if combine_pdf:
        merge_ood_triptych_pdfs(base_out)
    return metas[0]


def main() -> None:
    p = argparse.ArgumentParser(
        description="OOD: two PDFs — (1) pre Chronos+ST desc together; (2) post-aligned; shared PCA basis and axis limits"
    )
    p.add_argument("--exp-dir", type=Path, default=None, help="Single training run directory")
    p.add_argument("--exp-dirs", type=Path, nargs="*", default=None, help="Multiple run directories")
    p.add_argument("--out-dir", type=Path, default=None, help="Base output directory for batch mode")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--common-dim", type=int, default=None)
    p.add_argument("--dataset-a", type=str, default=None)
    p.add_argument("--dataset-b", type=str, default=None)
    p.add_argument(
        "--desc-max-chars",
        type=int,
        default=900,
        help="Truncate dataset description strings in the JSON meta only (0 = no truncation)",
    )
    p.add_argument(
        "--num-pair-iterations",
        type=int,
        default=1,
        metavar="N",
        help="Number of distinct dataset pairs to visualize (random, no replacement). Incompatible with --dataset-a/b.",
    )
    p.add_argument(
        "--single-dataset-path",
        action="store_true",
        help="Also write single-dataset path plots (TS polyline in angular order around mean desc; pre/post).",
    )
    p.add_argument(
        "--path-max-segments",
        type=int,
        default=48,
        metavar="K",
        help="Max segments for single-dataset path viz (default 48).",
    )
    p.add_argument(
        "--path-dataset",
        type=str,
        default=None,
        help="Normalized dataset_name for single-dataset path viz; default = dataset with the most rows.",
    )
    p.add_argument(
        "--no-two-dataset-path",
        action="store_true",
        help="With --single-dataset-path, skip the two-dataset path figures (single-dataset only).",
    )
    p.add_argument(
        "--path-dataset-a",
        type=str,
        default=None,
        help="Optional first dataset for two-dataset path viz (use with --path-dataset-b).",
    )
    p.add_argument(
        "--path-dataset-b",
        type=str,
        default=None,
        help="Optional second dataset for two-dataset path viz (use with --path-dataset-a).",
    )
    p.add_argument(
        "--path-fill-alpha",
        type=float,
        default=0.22,
        metavar="A",
        help="Alpha for transparent polygon fill inside TS paths (default 0.22).",
    )
    p.add_argument(
        "--no-combine-pdf",
        action="store_true",
        help="Do not write ood_embedding_all_plots_combined.pdf after a viz run.",
    )
    p.add_argument(
        "--merge-pdfs-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Only merge PDFs in this OOD triptych output directory; no model run.",
    )
    p.add_argument(
        "--merge-pdfs-root",
        type=Path,
        default=None,
        metavar="DIR",
        help="Merge each immediate subfolder's OOD PDFs into one file under this parent (sorted by name).",
    )
    args = p.parse_args()

    if args.merge_pdfs_dir is not None:
        out = merge_ood_triptych_pdfs(args.merge_pdfs_dir.resolve())
        if out is None:
            print("No PDFs found to merge.", file=sys.stderr)
            raise SystemExit(1)
        return
    if args.merge_pdfs_root is not None:
        out = merge_ood_triptych_pdfs_under_root(args.merge_pdfs_root.resolve())
        if out is None:
            print("No OOD triptych PDFs found under subdirectories.", file=sys.stderr)
            raise SystemExit(1)
        return

    dirs: list[Path] = []
    if args.exp_dirs:
        dirs.extend(args.exp_dirs)
    if args.exp_dir:
        dirs.append(args.exp_dir)
    if not dirs:
        raise SystemExit("Provide --exp-dir and/or --exp-dirs")

    if len(dirs) > 1 and args.out_dir is None:
        raise SystemExit("With multiple experiment dirs, pass --out-dir (each run gets a subfolder by run name)")

    for exp in dirs:
        exp = exp.resolve()
        if args.out_dir is None:
            sub_out = None
        elif len(dirs) == 1:
            sub_out = args.out_dir
        else:
            sub_out = args.out_dir / exp.name
        run_ood_embedding_triptych_viz(
            exp_dir=exp,
            out_dir=sub_out,
            device_str=args.device,
            random_seed=args.seed,
            common_dim=args.common_dim,
            dataset_a=args.dataset_a,
            dataset_b=args.dataset_b,
            desc_max_chars=args.desc_max_chars,
            num_pair_iterations=args.num_pair_iterations,
            single_dataset_path=args.single_dataset_path,
            two_dataset_path=not args.no_two_dataset_path,
            path_max_segments=args.path_max_segments,
            path_dataset=args.path_dataset,
            path_dataset_a=args.path_dataset_a,
            path_dataset_b=args.path_dataset_b,
            path_fill_alpha=args.path_fill_alpha,
            combine_pdf=not args.no_combine_pdf,
        )


if __name__ == "__main__":
    main()
