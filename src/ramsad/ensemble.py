"""Score the detector ensembles selected by RAMSAD (``select.n=N``).

For every query series in a ``model_selection.csv`` written by :mod:`ramsad.select`, the
``selected_models`` column lists the top-*N* detectors. This module turns that selection into an
anomaly score: each detector's score is min-max normalised to [0, 1] and the normalised scores are
averaged point-wise (the aggregation rule used for "RAMSAD (ens.)" in the paper).

Per-detector scores are read from a TSB-AD-style cache ``<scores_dir>/<Detector>/<series>.npy``.
Missing scores can be computed on the fly with TSB-AD (``pip install -e ".[ensemble]"``), which is
also used to compute VUS-PR when labels are available.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Detector names used in the performance tables (and so in ``selected_models``) -> TSB-AD names.
TSBAD_NAMES = {
    "KMeansAD": "KMeansAD_U",
    "Lag-Llama": "Lag_Llama",
    "MOMENT (ZS)": "MOMENT_ZS",
    "MOMENT (FT)": "MOMENT_FT",
    **{f"Sub-{m}": f"Sub_{m}" for m in ("IForest", "LOF", "PCA", "HBOS", "OCSVM", "MCD", "KNN")},
}


def tsbad_name(model: str) -> str:
    return TSBAD_NAMES.get(model, model)


def minmax(score: np.ndarray) -> np.ndarray:
    s = np.asarray(score, dtype=np.float64).ravel()
    lo, hi = np.nanmin(s), np.nanmax(s)
    if not np.isfinite(hi - lo) or hi == lo:
        return np.zeros_like(s)
    return (s - lo) / (hi - lo)


def combine_scores(scores: list[np.ndarray], decimals: int | None = 3) -> np.ndarray:
    """Average of min-max normalised scores, truncated to the shortest score.

    ``decimals`` rounds each normalised score and the average, as in the paper's runs;
    ``None`` disables rounding.
    """
    if not scores:
        raise ValueError("no detector scores to combine")
    normed = [minmax(s) for s in scores]
    if decimals is not None:
        normed = [np.round(s, decimals) for s in normed]
    n = min(len(s) for s in normed)
    out = np.mean([s[:n] for s in normed], axis=0)
    return np.round(out, decimals) if decimals is not None else out


def read_series(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """TSB-AD CSV: value column(s), then an optional ``Label`` column."""
    df = pd.read_csv(path).dropna()
    if "Label" in df.columns:
        return df.drop(columns=["Label"]).to_numpy(float), df["Label"].to_numpy(int)
    return df.to_numpy(float), None


def train_index_from_name(name: str) -> int:
    """TSB-AD file names carry the train/test split point: ``..._tr_<index>_1st_<first anomaly>``."""
    parts = name.split("_")
    return int(parts[parts.index("tr") + 1])


def run_detector(model: str, data: np.ndarray, train_index: int) -> np.ndarray:
    """Run one detector with TSB-AD's tuned hyperparameters, as in the TSB-AD benchmark."""
    try:
        from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict
        from TSB_AD.model_wrapper import (
            Semisupervise_AD_Pool,
            Unsupervise_AD_Pool,
            run_Semisupervise_AD,
            run_Unsupervise_AD,
        )
    except ImportError as e:
        raise ImportError('Running detectors needs TSB-AD: pip install -e ".[ensemble]"') from e

    name = tsbad_name(model)
    hp = Optimal_Uni_algo_HP_dict.get(name, {})
    if name in Semisupervise_AD_Pool:
        out = run_Semisupervise_AD(name, data[:train_index], data, **hp)
    elif name in Unsupervise_AD_Pool:
        out = run_Unsupervise_AD(name, data, **hp)
    else:
        raise ValueError(f"{model} is not a TSB-AD detector")
    if not isinstance(out, np.ndarray):
        raise RuntimeError(f"{model} failed: {out}")
    return out.ravel()


def find_score_file(scores_dir: Path, model: str, series: str) -> Path | None:
    d = scores_dir / tsbad_name(model)
    exact = d / f"{series}.npy"
    if exact.is_file():
        return exact
    hits = sorted(d.glob(f"*{series}*.npy")) if d.is_dir() else []
    return hits[0] if hits else None


def load_or_run_score(
    model: str,
    series: str,
    data: np.ndarray,
    scores_dir: Path,
    run_missing: bool,
) -> np.ndarray | None:
    path = find_score_file(scores_dir, model, series)
    if path is not None:
        return np.load(path)
    if not run_missing:
        log.warning("no cached score for %s on %s (set ensemble.run_missing=true to compute it)", model, series)
        return None
    score = run_detector(model, data, train_index_from_name(series))
    out = scores_dir / tsbad_name(model) / f"{series}.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, score)
    return score


def vus_pr(
    score: np.ndarray,
    labels: np.ndarray,
    data: np.ndarray,
    thresholds: int = 2000,
    window_factor: float = 2.0,
) -> float:
    """VUS-PR with TSB-AD's implementation; sliding window from the first series column."""
    try:
        from TSB_AD.evaluation.basic_metrics import generate_curve
        from TSB_AD.utils.slidingWindows import find_length_rank
    except ImportError as e:
        raise ImportError('VUS-PR needs TSB-AD: pip install -e ".[ensemble]"') from e
    n = min(len(score), len(labels))
    window = max(1, int(round(find_length_rank(data[:, :1], rank=1) * window_factor)))
    return float(generate_curve(labels[:n].astype(int), score[:n], window, "opt", thresholds)[-1])


def find_series_file(series_dirs: list[Path], series: str) -> Path | None:
    for d in series_dirs:
        p = d / f"{series}.csv"
        if p.is_file():
            return p
    return None


def run_ensemble(
    *,
    selection_csv: Path,
    series_dirs: list[Path],
    scores_dir: Path,
    out_dir: Path,
    decimals: int | None = 3,
    run_missing: bool = False,
    evaluate: bool = True,
    vus_thresholds: int = 2000,
    vus_window_factor: float = 2.0,
    save_scores: bool = True,
    verbose: bool = True,
) -> Path:
    """Score every row of ``selection_csv``; writes ``ensemble_results.csv`` and per-series ``.npy`` scores."""
    sel = pd.read_csv(selection_csv)
    col = "selected_models" if "selected_models" in sel.columns else "selected_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    score_out = out_dir / "ensemble_scores"
    rows = []
    for r in sel.itertuples(index=False):
        series = str(r.time_series).strip().removesuffix(".csv")
        models = [m.strip() for m in str(getattr(r, col)).split(";") if m.strip()]
        path = find_series_file(series_dirs, series)
        if path is None:
            log.warning("series file not found for %s", series)
            continue
        data, labels = read_series(path)
        scores, used = [], []
        for m in models:
            s = load_or_run_score(m, series, data, scores_dir, run_missing)
            if s is not None:
                scores.append(s)
                used.append(m)
        if not scores:
            log.warning("no detector scores available for %s", series)
            continue
        ens = combine_scores(scores, decimals)
        if save_scores:
            score_out.mkdir(exist_ok=True)
            np.save(score_out / f"{series}.npy", ens)
        row = {"time_series": series, "n_selected": len(models), "n_used": len(used), "models": ";".join(used)}
        if evaluate and labels is not None:
            row["vus_pr"] = vus_pr(ens, labels, data, vus_thresholds, vus_window_factor)
        rows.append(row)
        if verbose:
            extra = f" VUS-PR={row['vus_pr']:.4f}" if "vus_pr" in row else ""
            print(f"[ensemble] {series}: {len(used)}/{len(models)} detectors{extra}")

    if not rows:
        raise RuntimeError(f"no series scored from {selection_csv}; check series_dirs and scores_dir")
    res = pd.DataFrame(rows)
    out_csv = out_dir / "ensemble_results.csv"
    res.to_csv(out_csv, index=False)
    if verbose and "vus_pr" in res.columns:
        print(f"[ensemble] mean VUS-PR over {len(res)} series: {res['vus_pr'].mean():.4f}")
    if verbose:
        print(f"[ensemble] wrote {out_csv}")
    return out_csv
