#!/usr/bin/env python3
"""
Main training script for embedding alignment model.
Uses Hydra for configuration management.
"""

import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import torch.utils.data
from datetime import datetime
import csv
import numpy as np

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from model import EmbeddingAlignmentModel
from data import EmbeddingAlignmentDataset
from trainer import Trainer, get_lr_scheduler
from utils import (
    plot_training_curves,
    visualize_alignment_tsne,
    visualize_sample_retrieval,
    visualize_two_dataset_pre_post_alignment,
)
import torch.nn as nn

# Import wandb for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed. Install with: pip install wandb")

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    from hydra.utils import to_absolute_path

    OmegaConf.update(cfg, "data.parquet_file", to_absolute_path(cfg.data.parquet_file), merge=True)
    OmegaConf.update(cfg, "output_dir", to_absolute_path(str(cfg.output_dir)), merge=True)
    if cfg.data.get("performance_file"):
        OmegaConf.update(
            cfg, "data.performance_file", to_absolute_path(str(cfg.data.performance_file)), merge=True
        )

    # Set CUDA_VISIBLE_DEVICES from config before any CUDA calls
    cuda_device = getattr(cfg, "cuda_device", None)
    if cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    
    use_ddp = False
    is_rank0 = True
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    
    # Set CUDA device
    if cfg.device == "cuda" and torch.cuda.is_available():
        if cfg.parallelization.multi_gpu == "ddp" and local_rank >= 0:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            dist.init_process_group(backend="nccl")
            use_ddp = True
            is_rank0 = dist.get_rank() == 0
        else:
            device = torch.device("cuda")
        
        # Enable TF32 for faster training on Ampere+ GPUs
        if cfg.parallelization.enable_tf32:
            # Check if GPU supports TF32 (compute capability >= 8.0)
            if torch.cuda.get_device_capability()[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("✅ TF32 enabled for faster training")
            else:
                print("⚠️  TF32 not available on this GPU (requires compute capability >= 8.0)")
    else:
        device = torch.device("cpu")
    
    # Set random seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
    if is_rank0:
        print(f"\n{'='*80}")
        print("Time Series - Description Embedding Alignment Model")
        print(f"{'='*80}\n")
        
        print(f"Configuration:")
        print(OmegaConf.to_yaml(cfg))
        print(f"\nDevice: {device}")
        print(f"Parquet file: {cfg.data.parquet_file}")
        print(f"TS embedding column: {cfg.data.ts_emb_col}")
        print(f"Description embedding column: {cfg.data.desc_emb_col}\n")
    
    # Create unique output directories with timestamp
    # Format: base_dir/run-YYYYMMDD-HHMMSS-loss_function
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name_prefix = getattr(cfg, "run_name_prefix", "run")
    run_name = f"{run_name_prefix}-{timestamp}-{cfg.model.loss_function}"
    base_output_dir = Path(cfg.output_dir)
    
    # Create unique run directory
    output_dir = base_output_dir / run_name
    checkpoint_dir = output_dir / "checkpoints"
    plots_dir = output_dir / "plots"
    
    if is_rank0:
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to output directory for reproducibility
    config_save_path = output_dir / "config.yaml"
    if is_rank0:
        with open(config_save_path, 'w') as f:
            OmegaConf.save(cfg, f)
        
        print(f"Output directory: {output_dir}")
        print(f"Checkpoint directory: {checkpoint_dir}")
        print(f"Plots directory: {plots_dir}")
        print(f"Config saved to: {config_save_path}")
    
    # Print parallelization settings
    if is_rank0:
        if cfg.parallelization.mixed_precision != "none":
            print(f"✅ Mixed precision training: {cfg.parallelization.mixed_precision}")
        if cfg.parallelization.torch_compile:
            print("✅ torch.compile: enabled")
        print()
    
    # Load dataset
    dataset = EmbeddingAlignmentDataset(
        Path(cfg.data.parquet_file),
        cfg.data.ts_emb_col,
        cfg.data.desc_emb_col,
        desc_text_col=getattr(cfg.data, "desc_text_col", None),
        series_col=getattr(cfg.data, "series_col", None),
        val_stratify_col=getattr(cfg.data, "strat_val_col", None),
        use_fast_load=getattr(cfg.data, "use_fast_load", True),
        align_to_performance=cfg.model.get("align_to_performance", False),
        performance_file=cfg.data.get("performance_file", None),
    )

    # Backbone columns can differ in width (e.g. 512 vs 768); default model.*_emb_dim may not match.
    actual_ts_dim = int(dataset.ts_embeddings.shape[1])
    actual_desc_dim = int(dataset.desc_embeddings.shape[1])
    emb_cfg_updated = False
    if actual_ts_dim != int(cfg.model.ts_emb_dim):
        if is_rank0:
            print(
                f"⚠️  model.ts_emb_dim ({cfg.model.ts_emb_dim}) != parquet column "
                f"({actual_ts_dim}); using data dimension."
            )
        OmegaConf.update(cfg, "model.ts_emb_dim", actual_ts_dim, merge=True)
        emb_cfg_updated = True
    if actual_desc_dim != int(cfg.model.desc_emb_dim):
        if is_rank0:
            print(
                f"⚠️  model.desc_emb_dim ({cfg.model.desc_emb_dim}) != parquet column "
                f"({actual_desc_dim}); using data dimension."
            )
        OmegaConf.update(cfg, "model.desc_emb_dim", actual_desc_dim, merge=True)
        emb_cfg_updated = True
    if emb_cfg_updated and is_rank0:
        with open(config_save_path, "w") as f:
            OmegaConf.save(cfg, f)
        print(f"Config re-saved with matched embedding dims: {config_save_path}\n")
    
    # Stratified train/val split.
    # If training.val_samples_per_dataset == 0:
    #   use fractional split per description based on train_val_split.
    # If training.val_samples_per_dataset > 0:
    #   ignore train_val_split and take up to that many validation samples per description.
    train_val_split = cfg.training.train_val_split
    val_samples_per_dataset = getattr(cfg.training, "val_samples_per_dataset", 0)
    rng = np.random.default_rng(cfg.training.random_seed)
    
    stratify_labels = np.array(getattr(dataset, "stratify_labels", dataset.descriptions))
    unique_labels = np.unique(stratify_labels)
    
    val_indices = []
    train_indices = []
    train_idx_per_label = {}  # {label: list of train indices} for stratified sampling
    # Optional tracking of validation sample counts per label (group)
    val_counts_per_label = {}
    
    for label in unique_labels:
        label_idx = np.where(stratify_labels == label)[0]
        rng.shuffle(label_idx)
        
        # Number of validation samples for this label:
        if val_samples_per_dataset and val_samples_per_dataset > 0:
            # Fixed number per label (capped by available samples)
            n_val_label = min(int(val_samples_per_dataset), len(label_idx))
        else:
            # Fractional split: take approximately the same fraction
            # (1 - train_val_split) of its samples, with at least 1 when possible.
            n_val_label = max(1, int(round((1.0 - train_val_split) * len(label_idx))))
            n_val_label = min(n_val_label, len(label_idx))
        val_idx_label = label_idx[:n_val_label]
        train_idx_label = label_idx[n_val_label:]
        
        val_indices.extend(val_idx_label.tolist())
        train_indices.extend(train_idx_label.tolist())
        train_idx_per_label[label] = train_idx_label.tolist()
        # Track how many validation samples we took for this label
        val_counts_per_label[label] = len(val_idx_label)
    
    # If we ended up with zero training samples for some very small descriptions, put at least one into train
    if len(train_indices) == 0 and len(val_indices) > 0:
        # Move one sample back to train
        train_indices.append(val_indices.pop())
    
    # Shuffle final index lists
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    
    # Sample train_size according to train_sampling_mode
    train_sampling_mode = getattr(cfg.training, "train_sampling_mode", "random")
    if train_sampling_mode == "stratified_uniform":
        # Uniform across labels: each label gets ~train_size/num_labels samples
        labels_with_train = {
            d: idx for d, idx in train_idx_per_label.items() if len(idx) > 0
        }
        num_labels = len(labels_with_train)
        if num_labels > 0:
            target_per_label = cfg.training.train_size / num_labels
            sampled_per_label = []
            for label, idx_list in labels_with_train.items():
                n_available = len(idx_list)
                n_take = min(int(np.ceil(target_per_label)), n_available)
                if n_take > 0:
                    chosen = rng.choice(idx_list, size=n_take, replace=False)
                    sampled_per_label.extend(chosen.tolist())
            # If we overshot (due to ceil), trim; if undershot, we keep what we have
            rng.shuffle(sampled_per_label)
            train_indices = sampled_per_label[: cfg.training.train_size]
            if is_rank0:
                print(
                    f"Stratified uniform sampling: ~{target_per_label:.1f} samples/label from "
                    f"{num_labels} labels, total={len(train_indices)}"
                )
        else:
            # Fallback: no label has train samples (edge case), use train_indices as-is
            train_indices = train_indices[: cfg.training.train_size]
    else:
        # random: original behavior - shuffle and take first train_size
        train_indices = train_indices[: cfg.training.train_size]
    print(f"Effective train size: {len(train_indices)}")
    print(f"Effective val indices: {len(val_indices)}")
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    if is_rank0:
        print(f"\nTrain samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}\n")
        # Optionally save validation samples per description (group) to CSV
        if getattr(cfg.training, "print_val_samples_per_desc", False):
            sorted_items = sorted(val_counts_per_label.items(), key=lambda x: -x[1])
            csv_path = output_dir / "val_samples_per_label.csv"
            with open(csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["stratify_label", "val_count"])
                for idx, (label, count) in enumerate(sorted_items, start=1):
                    writer.writerow([label, count])
            print(f"Validation samples per label saved to: {csv_path}")
    
    # Create dataloaders
    train_sampler = None
    val_sampler = None
    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=False)

    
    dl_kwargs = dict(num_workers=cfg.training.num_workers, pin_memory=True)
    if cfg.training.num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **dl_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        sampler=val_sampler,
        **dl_kwargs,
    )
    
    # Initialize model
    model = EmbeddingAlignmentModel(
        ts_emb_dim=cfg.model.ts_emb_dim,
        desc_emb_dim=cfg.model.desc_emb_dim,
        projection_dim=cfg.model.projection_dim,
        temperature=cfg.model.temperature,
        loss_function=cfg.model.loss_function,
        desc_match_mode=cfg.model.get("desc_match_mode", "exact"),
        desc_similarity_threshold=cfg.model.get("desc_similarity_threshold", 0.95),
        per_series_loss=cfg.model.get("per_series_loss", False),
        preserve_temporal_identity=cfg.model.get("preserve_temporal_identity", False),
        identity_penalty_weight=cfg.model.get("identity_penalty_weight", 1.0),
        series_level_alignment=cfg.model.get("series_level_alignment", False),
        align_to_performance=cfg.model.get("align_to_performance", False),
        performance_dim=cfg.model.get("performance_dim", 32),
    ).to(device)
    
    # Multi-GPU support (apply before torch.compile)
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_data_parallel = False
    if cfg.parallelization.multi_gpu == "dp" and num_gpus > 1:
        if is_rank0:
            print(f"✅ Using DataParallel on {num_gpus} GPUs")
        model = nn.DataParallel(model)
        use_data_parallel = True
        # Update device to cuda:0 for DataParallel
        device = torch.device("cuda:0")
    elif cfg.parallelization.multi_gpu == "ddp" and not use_ddp:
        if is_rank0:
            print("⚠️  DistributedDataParallel (DDP) requires torchrun. Use: torchrun --nproc-per-node=N train.py")
            print("   Falling back to single GPU training.")
    elif use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)


    if cfg.parallelization.torch_compile and not use_ddp:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            if is_rank0:
                print("✅ Model compiled with torch.compile")
        except Exception as e:
            if is_rank0:
                print(f"⚠️  torch.compile failed: {e}. Continuing without compilation.")
    elif cfg.parallelization.torch_compile and use_ddp and is_rank0:
        print("⚠️  torch.compile is disabled for DDP in this script.")
    
    if is_rank0:
        print(f"\nModel architecture:")
        print(model)
        print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        if num_gpus > 1 and cfg.parallelization.multi_gpu == "dp":
            print(f"Using {num_gpus} GPUs with DataParallel\n")
        elif use_ddp:
            print(f"Using {world_size} GPUs with DistributedDataParallel\n")
        else:
            print()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.optimizer.weight_decay,
        betas=tuple(cfg.training.optimizer.betas),
    )
    
    # Learning rate scheduler
    # Extract scheduler kwargs (exclude 'name' key)
    scheduler_kwargs = {}
    if hasattr(cfg.training.scheduler, 'keys'):
        for key in cfg.training.scheduler.keys():
            if key != 'name':
                scheduler_kwargs[key] = cfg.training.scheduler[key]
    
    scheduler = get_lr_scheduler(
        optimizer,
        cfg.training.num_epochs,
        scheduler_name=cfg.training.scheduler.name,
        warmup_epochs=cfg.training.warmup_epochs,
        scheduler_kwargs=scheduler_kwargs,
    )
    
    # Initialize wandb if enabled
    wandb_enabled = cfg.get('wandb', {}).get('enabled', False)
    if wandb_enabled and is_rank0:
        if not WANDB_AVAILABLE:
            print("⚠️  wandb is enabled in config but not installed!")
            print("   Install with: pip install wandb")
            print("   Or set wandb.enabled: false in config.yaml")
        else:
            try:
                # Get wandb config with defaults
                wandb_config = cfg.get('wandb', {})
                entity_name = wandb_config.get('entity', None)  # Team/organization name
                project_name = wandb_config.get('project', 'embedding-alignment')
                wandb_tags = wandb_config.get('tags', [])
                
                print(f"\n{'='*60}")
                print("Initializing wandb...")
                if entity_name:
                    print(f"  Entity/Team: {entity_name}")
                print(f"  Project: {project_name}")
                print(f"  Run name: {run_name}")
                print(f"  Tags: {wandb_tags}")
                print(f"{'='*60}\n")
                
                # Build wandb.init arguments
                init_kwargs = {
                    'project': project_name,
                    'name': run_name,
                    'config': OmegaConf.to_container(cfg, resolve=True),
                    'dir': str(output_dir),
                }
                
                # Add entity if specified
                if entity_name:
                    init_kwargs['entity'] = entity_name
                
                # Add tags if provided
                if wandb_tags:
                    init_kwargs['tags'] = wandb_tags
                
                wandb.init(**init_kwargs)
                print(f"✅ wandb initialized successfully!")
                print(f"   View at: {wandb.run.url}\n")
            except Exception as e:
                print(f"❌ Failed to initialize wandb: {e}")
                print("   Continuing without wandb logging...\n")
                wandb_enabled = False
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=cfg.training.num_epochs,
        grad_clip_norm=cfg.training.grad_clip_norm,
        checkpoint_dir=str(checkpoint_dir),
        mixed_precision=cfg.parallelization.mixed_precision,
        use_wandb=WANDB_AVAILABLE and wandb_enabled,
        is_rank0=is_rank0,
    )
    
    # Train
    trainer.train()
    
    # Finish wandb run
    if WANDB_AVAILABLE and wandb_enabled and is_rank0:
        try:
            wandb.finish()
            print("✅ wandb run finished")
        except Exception as e:
            print(f"⚠️  Error finishing wandb run: {e}")
    
    # Print final results
    if is_rank0:
        print(f"\n{'='*80}")
        print("Training completed!")
        print(f"{'='*80}\n")
        print(f"Best validation loss: {trainer.best_val_loss:.4f}")
        print(f"Final validation Top-1 accuracy: {trainer.val_top1_accs[-1]*100:.2f}%")
        print(f"Final validation Top-5 accuracy: {trainer.val_top5_accs[-1]*100:.2f}%")
        print(f"\nModels saved to:")
        print(f"  - Best: {checkpoint_dir}/best_model.pt")
        print(f"  - Final: {checkpoint_dir}/final_model.pt\n")
    
    # Visualizations (optional)
    if is_rank0 and (
        cfg.visualization.plot_training
        or cfg.visualization.plot_tsne
        or cfg.visualization.plot_retrieval
        or getattr(cfg.visualization, "plot_two_dataset_alignment", False)
    ):
        print(f"{'='*80}")
        print("Generating visualizations...")
        print(f"{'='*80}\n")
        
        # Load best model for visualizations
        checkpoint = torch.load(checkpoint_dir / "best_model.pt", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        if cfg.visualization.plot_training:
            plot_training_curves(
                trainer.train_losses,
                trainer.val_losses,
                trainer.val_top1_accs,
                trainer.val_top5_accs,
                plots_dir / "training_curves.png",
            )
        
        if cfg.visualization.plot_tsne:
            visualize_alignment_tsne(
                model,
                val_loader,
                device,
                plots_dir / "alignment_tsne.png",
                n_samples=1000,
            )
        
        if cfg.visualization.plot_retrieval:
            visualize_sample_retrieval(
                model,
                val_dataset,
                device,
                plots_dir / "sample_retrieval.png",
                Path(cfg.data.parquet_file),
                n_samples=10,
            )

        if getattr(cfg.visualization, "plot_two_dataset_alignment", False):
            visualize_two_dataset_pre_post_alignment(
                dataset,
                model,
                device,
                plots_dir,
                align_to_performance=cfg.model.get("align_to_performance", False),
                random_seed=cfg.seed,
                reduction=getattr(cfg.visualization, "two_dataset_reduction", "pca"),
            )

        print(f"\n✅ All visualizations saved to: {plots_dir}/")

    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

