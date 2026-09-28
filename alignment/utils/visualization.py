"""Visualization utilities for embedding alignment."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import json


def _apply_darth_style():
    """Apply DARTH notebook-inspired publication style."""
    plt.rcParams.update({
        "font.size": 40,
        "axes.titlesize": 40,
        "axes.labelsize": 32,
        "xtick.labelsize": 28,
        "ytick.labelsize": 28,
        "legend.fontsize": 26,
    })


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    val_top1_accs: list,
    val_top5_accs: list,
    save_path: Path,
):
    """Plot training curves, export top-k PDFs, and persist plot data."""
    _apply_darth_style()
    fig, axes = plt.subplots(1, 3, figsize=(30, 9))
    
    epochs = range(1, len(train_losses) + 1)
    top1_pct = [acc * 100 for acc in val_top1_accs]
    top5_pct = [acc * 100 for acc in val_top5_accs]
    
    # Loss curve
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss', fontweight='bold')
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.3)
    
    # Top-1 Accuracy
    axes[1].plot(epochs, top1_pct, 'g-', label='Top-1 Acc', linewidth=3, marker='o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Validation Top-1 Accuracy', fontweight='bold')
    axes[1].legend(frameon=False)
    axes[1].grid(True, alpha=0.3)
    
    # Top-5 Accuracy
    axes[2].plot(epochs, top5_pct, 'm-', label='Top-5 Acc', linewidth=3, marker='s')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('Validation Top-5 Accuracy', fontweight='bold')
    axes[2].legend(frameon=False)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  📊 Saved training curves to: {save_path}")
    plt.close()

    save_path = Path(save_path)
    plots_dir = save_path.parent
    stem = save_path.stem
    top1_pdf = plots_dir / f"{stem}_top1.pdf"
    top5_pdf = plots_dir / f"{stem}_top5.pdf"
    # Explicit wandb-like naming for convenience.
    top1_pdf_wandb_name = plots_dir / "val_top1_acc.pdf"
    top5_pdf_wandb_name = plots_dir / "val_top5_acc.pdf"
    data_csv = plots_dir / f"{stem}_data.csv"
    data_json = plots_dir / f"{stem}_data.json"

    # Dedicated Top-1 plot (PDF)
    fig1, ax1 = plt.subplots(figsize=(16, 9))
    ax1.plot(epochs, top1_pct, color="#2ca02c", linewidth=4, marker='o', markersize=8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Top-1 Accuracy (%)")
    ax1.set_title("Validation Top-1 Accuracy", fontweight='bold')
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(top1_pdf, bbox_inches='tight')
    fig1.savefig(top1_pdf_wandb_name, bbox_inches='tight')
    plt.close(fig1)
    print(f"  📊 Saved Top-1 PDF to: {top1_pdf}")
    print(f"  📊 Saved Top-1 PDF to: {top1_pdf_wandb_name}")

    # Dedicated Top-5 plot (PDF)
    fig5, ax5 = plt.subplots(figsize=(16, 9))
    ax5.plot(epochs, top5_pct, color="#9467bd", linewidth=4, marker='s', markersize=8)
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("Top-5 Accuracy (%)")
    ax5.set_title("Validation Top-5 Accuracy", fontweight='bold')
    ax5.grid(True, alpha=0.3)
    fig5.tight_layout()
    fig5.savefig(top5_pdf, bbox_inches='tight')
    fig5.savefig(top5_pdf_wandb_name, bbox_inches='tight')
    plt.close(fig5)
    print(f"  📊 Saved Top-5 PDF to: {top5_pdf}")
    print(f"  📊 Saved Top-5 PDF to: {top5_pdf_wandb_name}")

    # Persist data for exact regeneration.
    df = pd.DataFrame({
        "step": list(epochs),
        "epoch": list(epochs),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_top1_acc": val_top1_accs,
        "val_top5_acc": val_top5_accs,
        "val_top1_pct": top1_pct,
        "val_top5_pct": top5_pct,
    })
    df.to_csv(data_csv, index=False)
    meta = {
        "style_source": "DARTH_plus_conformal.ipynb-inspired",
        "style": {
            "font.size": 40,
            "axes.titlesize": 40,
            "axes.labelsize": 32,
            "xtick.labelsize": 28,
            "ytick.labelsize": 28,
            "legend.fontsize": 26,
        },
        "files": {
            "combined_plot": str(save_path),
            "top1_pdf": str(top1_pdf),
            "top5_pdf": str(top5_pdf),
            "top1_pdf_wandb_name": str(top1_pdf_wandb_name),
            "top5_pdf_wandb_name": str(top5_pdf_wandb_name),
            "data_csv": str(data_csv),
        },
    }
    data_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  💾 Saved plot data to: {data_csv}")
    print(f"  💾 Saved plot metadata to: {data_json}")


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _embeddings_to_2d(X: np.ndarray, method: str, random_state: int) -> np.ndarray:
    """Reduce (n, d) to (n, 2) via PCA or UMAP when feasible."""
    if X.shape[0] < 2:
        raise ValueError("Need at least 2 points for 2D projection")
    method = (method or "pca").lower()
    if method == "umap" and X.shape[0] >= 15:
        try:
            import umap

            return umap.UMAP(n_components=2, random_state=random_state, n_neighbors=min(15, X.shape[0] - 1)).fit_transform(
                X.astype(np.float32)
            )
        except Exception as e:
            print(f"    UMAP failed ({e}), falling back to PCA")
    return PCA(n_components=min(2, X.shape[0] - 1, X.shape[1]), random_state=random_state).fit_transform(
        X.astype(np.float32)
    )


def visualize_two_dataset_pre_post_alignment(
    dataset: Dataset,
    model: nn.Module,
    device: torch.device,
    plots_dir: Path,
    align_to_performance: bool = False,
    random_seed: int = 42,
    reduction: str = "pca",
) -> None:
    """
    Pick two dataset names (series_col), two series segments each + one description vector per dataset.
    Pre-alignment: concat(raw_ts, raw_desc) and desc-only rows (zeros(ts_dim) || desc).
    Post-alignment: concat(ts_proj, desc_proj) from the trained model; desc-only via zero ts input.
    Saves two PDFs in DARTH style + CSV of points.
    """
    if align_to_performance:
        print("  Skipping two-dataset alignment plot (align_to_performance=True).")
        return

    ds = dataset
    if not hasattr(ds, "series_ids") or not hasattr(ds, "ts_embeddings"):
        print("  Skipping two-dataset alignment plot (dataset missing series_ids / ts_embeddings).")
        return

    series_ids = np.asarray(ds.series_ids)
    ts_emb = np.asarray(ds.ts_embeddings, dtype=np.float32)
    desc_emb = np.asarray(ds.desc_embeddings, dtype=np.float32)
    ts_dim = ts_emb.shape[1]
    desc_dim = desc_emb.shape[1]

    uniq, counts = np.unique(series_ids, return_counts=True)
    eligible = [u for u in uniq[counts >= 2] if str(u).strip()]
    if len(eligible) < 2:
        print(
            f"  Skipping two-dataset alignment plot (need 2 dataset names with >=2 rows; got {len(eligible)} eligible)."
        )
        return

    rng = np.random.default_rng(random_seed)
    rng.shuffle(eligible)
    d_a, d_b = str(eligible[0]), str(eligible[1])

    sid = np.asarray(series_ids).astype(str)
    idx_a = np.where(sid == d_a)[0][:2]
    idx_b = np.where(sid == d_b)[0][:2]
    if len(idx_a) < 2 or len(idx_b) < 2:
        print("  Skipping two-dataset alignment plot (insufficient rows after selection).")
        return

    concat_dim = ts_dim + desc_dim

    def pack_pre(i: int) -> np.ndarray:
        return np.concatenate([ts_emb[i], desc_emb[i]], axis=0)

    X_pre = np.stack(
        [
            pack_pre(idx_a[0]),
            pack_pre(idx_a[1]),
            pack_pre(idx_b[0]),
            pack_pre(idx_b[1]),
            np.concatenate([np.zeros(ts_dim, dtype=np.float32), desc_emb[idx_a[0]]]),
            np.concatenate([np.zeros(ts_dim, dtype=np.float32), desc_emb[idx_b[0]]]),
        ]
    )
    labels = [
        f"{d_a} series 1",
        f"{d_a} series 2",
        f"{d_b} series 1",
        f"{d_b} series 2",
        f"{d_a} desc",
        f"{d_b} desc",
    ]
    kinds = ["ts", "ts", "ts", "ts", "desc", "desc"]

    Z_pre = _embeddings_to_2d(X_pre, reduction, random_seed)

    model_ref = _unwrap_model(model)
    model.eval()
    with torch.no_grad():
        rows = []
        for i in list(idx_a) + list(idx_b):
            t = torch.tensor(ts_emb[i], device=device).unsqueeze(0)
            d = torch.tensor(desc_emb[i], device=device).unsqueeze(0)
            tp, dp = model_ref(t, d)
            rows.append(np.concatenate([tp.cpu().numpy().ravel(), dp.cpu().numpy().ravel()]))
        z0 = torch.zeros(1, ts_dim, device=device, dtype=torch.float32)
        da = torch.tensor(desc_emb[idx_a[0]], device=device).unsqueeze(0)
        db = torch.tensor(desc_emb[idx_b[0]], device=device).unsqueeze(0)
        tpa, dpa = model_ref(z0, da)
        tpb, dpb = model_ref(z0, db)
        rows.append(np.concatenate([tpa.cpu().numpy().ravel(), dpa.cpu().numpy().ravel()]))
        rows.append(np.concatenate([tpb.cpu().numpy().ravel(), dpb.cpu().numpy().ravel()]))
    X_post = np.stack(rows)
    Z_post = _embeddings_to_2d(X_post, reduction, random_seed + 1)

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    suffix = reduction.lower()
    pre_pdf = plots_dir / f"two_dataset_alignment_pre_{suffix}.pdf"
    post_pdf = plots_dir / f"two_dataset_alignment_post_{suffix}.pdf"
    data_csv = plots_dir / f"two_dataset_alignment_{suffix}_data.csv"

    _apply_darth_style()
    colors = {"A": "#1f77b4", "B": "#ff7f0e"}
    c_list = [colors["A"], colors["A"], colors["B"], colors["B"], colors["A"], colors["B"]]
    size_ts = 420
    size_desc = 620

    fig_pre, ax_pre = plt.subplots(figsize=(16, 9))
    for i in range(6):
        s = size_desc if kinds[i] == "desc" else size_ts
        m = "^" if kinds[i] == "desc" else "o"
        ax_pre.scatter(
            Z_pre[i, 0], Z_pre[i, 1], c=c_list[i], s=s, marker=m, edgecolors="black", linewidths=1.2, zorder=3
        )
    ax_pre.set_title(f"Pre-alignment (raw concat) — {d_a} vs {d_b}", fontweight="bold")
    ax_pre.set_xlabel("Dim 1")
    ax_pre.set_ylabel("Dim 2")
    ax_pre.grid(True, alpha=0.3)
    ax_pre.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["A"], markersize=18, label=f"{d_a} series"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["B"], markersize=18, label=f"{d_b} series"),
            plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#666666", markersize=18, label="Description (only)"),
        ],
        loc="best",
        frameon=False,
    )
    fig_pre.tight_layout()
    fig_pre.savefig(pre_pdf, bbox_inches="tight")
    plt.close(fig_pre)

    fig_post, ax_post = plt.subplots(figsize=(16, 9))
    for i in range(6):
        s = size_desc if kinds[i] == "desc" else size_ts
        m = "^" if kinds[i] == "desc" else "o"
        ax_post.scatter(
            Z_post[i, 0], Z_post[i, 1], c=c_list[i], s=s, marker=m, edgecolors="black", linewidths=1.2, zorder=3
        )
    ax_post.set_title(f"Post-alignment (projected) — {d_a} vs {d_b}", fontweight="bold")
    ax_post.set_xlabel("Dim 1")
    ax_post.set_ylabel("Dim 2")
    ax_post.grid(True, alpha=0.3)
    ax_post.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["A"], markersize=18, label=f"{d_a} series"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["B"], markersize=18, label=f"{d_b} series"),
            plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#666666", markersize=18, label="Description (only)"),
        ],
        loc="best",
        frameon=False,
    )
    fig_post.tight_layout()
    fig_post.savefig(post_pdf, bbox_inches="tight")
    plt.close(fig_post)

    out_df = pd.DataFrame(
        {
            "label": labels,
            "kind": kinds,
            "x_pre": Z_pre[:, 0],
            "y_pre": Z_pre[:, 1],
            "x_post": Z_post[:, 0],
            "y_post": Z_post[:, 1],
            "dataset_a": d_a,
            "dataset_b": d_b,
            "idx_a0": idx_a[0],
            "idx_a1": idx_a[1],
            "idx_b0": idx_b[0],
            "idx_b1": idx_b[1],
            "reduction": suffix,
            "concat_dim_pre": concat_dim,
            "concat_dim_post": X_post.shape[1],
        }
    )
    out_df.to_csv(data_csv, index=False)
    meta_path = plots_dir / f"two_dataset_alignment_{suffix}_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "style": "DARTH_plus_conformal-inspired (see plot_training_curves)",
                "datasets": [d_a, d_b],
                "reduction": suffix,
                "pre_pdf": str(pre_pdf),
                "post_pdf": str(post_pdf),
                "csv": str(data_csv),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  📊 Saved two-dataset pre-alignment PDF: {pre_pdf}")
    print(f"  📊 Saved two-dataset post-alignment PDF: {post_pdf}")
    print(f"  💾 Saved two-dataset alignment data: {data_csv}")


def visualize_alignment_tsne(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_path: Path,
    n_samples: int = 1000,
    random_seed: int = 42,
):
    """Visualize alignment using t-SNE."""
    print(f"\n  Generating t-SNE visualization (n_samples={n_samples})...")
    
    model.eval()
    ts_embeddings = []
    desc_embeddings = []
    
    np.random.seed(random_seed)
    collected = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if collected >= n_samples:
                break
            
            # Handle both old format (3 items) and new format (4 items with descriptions)
            if len(batch) == 3:
                ts_emb, desc_emb, indices = batch
            else:
                ts_emb, desc_emb, _, indices = batch
            
            ts_emb = ts_emb.to(device)
            desc_emb = desc_emb.to(device)
            
            ts_proj, desc_proj = model(ts_emb, desc_emb)
            
            batch_size = ts_emb.size(0)
            take = min(batch_size, n_samples - collected)
            
            ts_embeddings.append(ts_proj[:take].cpu().numpy())
            desc_embeddings.append(desc_proj[:take].cpu().numpy())
            
            collected += take
    
    ts_embeddings = np.vstack(ts_embeddings)
    desc_embeddings = np.vstack(desc_embeddings)
    
    # Combine for t-SNE
    all_embeddings = np.vstack([ts_embeddings, desc_embeddings])
    
    # Reduce dimensionality with PCA first if needed
    if all_embeddings.shape[1] > 50:
        print(f"    Applying PCA (dim {all_embeddings.shape[1]} -> 50)...")
        pca = PCA(n_components=50, random_state=random_seed)
        all_embeddings = pca.fit_transform(all_embeddings)
    
    # Apply t-SNE
    print(f"    Applying t-SNE (n_samples={len(all_embeddings)})...")
    tsne = TSNE(n_components=2, random_state=random_seed, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # Split back
    n = len(ts_embeddings)
    ts_2d = embeddings_2d[:n]
    desc_2d = embeddings_2d[n:]
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Scatter with connections
    axes[0].scatter(ts_2d[:, 0], ts_2d[:, 1], c='blue', alpha=0.6, s=30, label='Time Series', marker='o')
    axes[0].scatter(desc_2d[:, 0], desc_2d[:, 1], c='red', alpha=0.6, s=30, label='Descriptions', marker='^')
    
    # Draw lines connecting matching pairs (sample a few)
    n_show = min(50, n)
    indices = np.random.choice(n, n_show, replace=False)
    for idx in indices:
        axes[0].plot(
            [ts_2d[idx, 0], desc_2d[idx, 0]],
            [ts_2d[idx, 1], desc_2d[idx, 1]],
            'k-', alpha=0.2, linewidth=0.5
        )
    
    axes[0].set_title('t-SNE: Time Series vs Description Embeddings\n(Connected pairs shown)', 
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[0].set_ylabel('t-SNE Dimension 2', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Distance histogram
    distances = np.linalg.norm(ts_2d - desc_2d, axis=1)
    axes[1].hist(distances, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(distances.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {distances.mean():.3f}')
    axes[1].set_xlabel('Euclidean Distance (2D)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Pairwise Distances\n(Lower = Better Alignment)', 
                     fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  📊 Saved t-SNE visualization to: {save_path}")
    plt.close()


def visualize_sample_retrieval(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    save_path: Path,
    parquet_file: Path,
    n_samples: int = 10,
    random_seed: int = 42,
):
    """Visualize retrieval results for sample time series."""
    print(f"\n  Generating sample retrieval visualization (n_samples={n_samples})...")
    
    model.eval()
    np.random.seed(random_seed)
    
    # Get all embeddings
    all_ts_emb = []
    all_desc_emb = []
    
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    with torch.no_grad():
        for batch in dataloader:
            # Handle both old format (3 items) and new format (4 items with descriptions)
            if len(batch) == 3:
                ts_emb, desc_emb, indices = batch
            else:
                ts_emb, desc_emb, _, indices = batch
            
            ts_emb = ts_emb.to(device)
            desc_emb = desc_emb.to(device)
            ts_proj, desc_proj = model(ts_emb, desc_emb)
            
            all_ts_emb.append(ts_proj.cpu())
            all_desc_emb.append(desc_proj.cpu())
    
    all_ts_emb = torch.cat(all_ts_emb, dim=0)
    all_desc_emb = torch.cat(all_desc_emb, dim=0)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(all_ts_emb, all_desc_emb.t())
    
    # Sample random queries
    n_total = len(all_ts_emb)
    query_indices = np.random.choice(n_total, n_samples, replace=False)
    
    # Load descriptions from parquet for display
    df = pd.read_parquet(parquet_file)
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(16, 3 * n_samples))
    if n_samples == 1:
        axes = [axes]
    
    for i, query_idx in enumerate(query_indices):
        # Get top-5 matches
        similarities = similarity_matrix[query_idx]
        top5_indices = similarities.topk(5).indices.tolist()
        top5_scores = similarities.topk(5).values.tolist()
        
        # Get description text (truncate for display)
        desc_text = df['description'].iloc[query_idx]
        if len(desc_text) > 100:
            desc_text = desc_text[:100] + "..."
        
        # Plot
        ax = axes[i]
        y_pos = np.arange(5)
        colors = ['green' if idx == query_idx else 'blue' for idx in top5_indices]
        
        ax.barh(y_pos, top5_scores, color=colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"Rank {j+1}" for j in range(5)], fontsize=10)
        ax.set_xlabel('Similarity Score', fontsize=11)
        ax.set_title(f'Query {i+1}: "{desc_text}"\n(Green = Correct Match)', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add score labels
        for j, score in enumerate(top5_scores):
            ax.text(score + 0.01, j, f'{score:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  📊 Saved sample retrieval visualization to: {save_path}")
    plt.close()

