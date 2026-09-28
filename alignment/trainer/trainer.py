"""Training and evaluation functions."""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from typing import Tuple, Optional
import math
import time

# Import wandb if available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _reduce_sums(values: list[float], device: torch.device) -> list[float]:
    if not (dist.is_available() and dist.is_initialized()):
        return values
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().tolist()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
    mixed_precision: str = "none",
    scaler: Optional[GradScaler] = None,
) -> Tuple[float, dict]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    total_duplicates = 0
    total_unique_descs = 0
    
    # Determine if we should use autocast
    use_autocast = mixed_precision in ["fp16", "bf16"]
    dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
    
    for batch in dataloader:
        # Handle batch formats: 3 (legacy), 4 (desc+idx), 5 (desc+series_ids+idx), 6 (desc+series_ids+idx+perf_emb), or 7 (+label)
        perf_emb = None
        labels = None
        if len(batch) == 3:
            ts_emb, desc_emb, _ = batch
            descriptions = None
            series_ids = None
        elif len(batch) == 4:
            ts_emb, desc_emb, descriptions, _ = batch
            descriptions = list(descriptions) if descriptions is not None else None
            series_ids = None
        elif len(batch) == 5:
            ts_emb, desc_emb, descriptions, series_ids, _ = batch
            descriptions = list(descriptions) if descriptions is not None else None
            series_ids = list(series_ids) if series_ids is not None else None
        elif len(batch) == 6:
            ts_emb, desc_emb, descriptions, series_ids, _, perf_emb = batch
            descriptions = list(descriptions) if descriptions is not None else None
            series_ids = list(series_ids) if series_ids is not None else None
        else:
            ts_emb, desc_emb, descriptions, series_ids, _, perf_emb, labels = batch
            descriptions = list(descriptions) if descriptions is not None else None
            series_ids = list(series_ids) if series_ids is not None else None
        
        ts_emb = ts_emb.to(device)
        desc_emb = desc_emb.to(device)
        if perf_emb is not None:
            perf_emb = perf_emb.to(device)
        if labels is not None:
            labels = labels.to(device)
        
        # Forward pass with mixed precision if enabled
        optimizer.zero_grad()
        
        model_ref = _unwrap_model(model)
        
        # Series-level alignment: Mean-pool TS segments and descriptions by series_id
        if getattr(model_ref, "series_level_alignment", False) and series_ids is not None:
            # Group by series_id
            unique_series = list(dict.fromkeys(series_ids))
            
            pooled_ts_emb = []
            pooled_desc_emb = []
            pooled_descriptions = []
            pooled_series_ids = []
            pooled_perf_emb = [] if perf_emb is not None else None
            pooled_labels = [] if labels is not None else None
            
            for s_id in unique_series:
                indices = [i for i, sid in enumerate(series_ids) if sid == s_id]
                idx_tensor = torch.tensor(indices, device=device)
                
                # Mean pool multiple segments belonging to this series
                pooled_ts_emb.append(ts_emb[idx_tensor].mean(dim=0))
                
                # Description embeddings are identical across a series, but we mean anyway
                pooled_desc_emb.append(desc_emb[idx_tensor].mean(dim=0))
                
                # Descriptions and series id are identical
                pooled_descriptions.append(descriptions[indices[0]] if descriptions is not None else "")
                pooled_series_ids.append(s_id)

                if pooled_perf_emb is not None:
                    pooled_perf_emb.append(perf_emb[idx_tensor].mean(dim=0))
                if pooled_labels is not None:
                    pooled_labels.append((labels[idx_tensor] > 0).any().float())
                
            ts_emb = torch.stack(pooled_ts_emb)
            desc_emb = torch.stack(pooled_desc_emb)
            descriptions = pooled_descriptions if descriptions is not None else None
            series_ids = pooled_series_ids
            if pooled_perf_emb is not None:
                perf_emb = torch.stack(pooled_perf_emb)
            if pooled_labels is not None:
                labels = torch.stack(pooled_labels)

        # Decide whether to target descriptions or performance vectors
        target_emb = perf_emb if getattr(model_ref, "align_to_performance", False) and perf_emb is not None else desc_emb

        if use_autocast:
            with autocast('cuda',dtype=dtype):
                ts_proj, target_proj = model(ts_emb, target_emb)
                loss, stats = model_ref.compute_loss(
                    ts_proj=ts_proj, 
                    desc_proj=target_proj, 
                    descriptions=descriptions, 
                    desc_emb=target_emb, 
                    series_ids=series_ids, 
                    ts_emb=ts_emb,
                    labels=labels
                )
            
            # Backward pass with gradient scaling for fp16
            if mixed_precision == "fp16" and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                # bf16 doesn't need gradient scaling
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()
        else:
            # Standard precision training
            ts_proj, target_proj = model(ts_emb, target_emb)
            loss, stats = model_ref.compute_loss(
                ts_proj=ts_proj, 
                desc_proj=target_proj, 
                descriptions=descriptions, 
                desc_emb=target_emb, 
                series_ids=series_ids, 
                ts_emb=ts_emb,
                labels=labels
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
        
        # Track statistics
        total_duplicates += stats.get("num_duplicates", 0)
        total_unique_descs += stats.get("num_unique_descs", len(ts_emb))
        
        total_loss += loss.item()
        num_batches += 1
    
    total_loss, num_batches, total_duplicates, total_unique_descs = _reduce_sums(
        [total_loss, num_batches, total_duplicates, total_unique_descs],
        device,
    )
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    epoch_stats = {
        "avg_duplicates_per_batch": total_duplicates / num_batches if num_batches > 0 else 0,
        "avg_unique_descs_per_batch": total_unique_descs / num_batches if num_batches > 0 else 0,
    }
    return avg_loss, epoch_stats


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    mixed_precision: str = "none",
) -> Tuple[float, float, float, dict]:
    """Evaluate model and compute accuracy metrics.
    For multi-positive: uses recall@k (any positive in top-k) instead of diagonal.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0
    total_duplicates = 0
    total_unique_descs = 0
    
    # Determine if we should use autocast
    use_autocast = mixed_precision in ["fp16", "bf16"]
    dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
    
    with torch.no_grad():
        for batch in dataloader:
            # Handle batch formats: 3 (legacy), 4 (desc+idx), 5 (desc+series_ids+idx), 6 (desc+series_ids+idx+perf_emb), or 7 (+label)
            perf_emb = None
            labels = None
            if len(batch) == 3:
                ts_emb, desc_emb, _ = batch
                descriptions = None
                series_ids = None
            elif len(batch) == 4:
                ts_emb, desc_emb, descriptions, _ = batch
                descriptions = list(descriptions) if descriptions is not None else None
                series_ids = None
            elif len(batch) == 5:
                ts_emb, desc_emb, descriptions, series_ids, _ = batch
                descriptions = list(descriptions) if descriptions is not None else None
                series_ids = list(series_ids) if series_ids is not None else None
            elif len(batch) == 6:
                ts_emb, desc_emb, descriptions, series_ids, _, perf_emb = batch
                descriptions = list(descriptions) if descriptions is not None else None
                series_ids = list(series_ids) if series_ids is not None else None
            else:
                ts_emb, desc_emb, descriptions, series_ids, _, perf_emb, labels = batch
                descriptions = list(descriptions) if descriptions is not None else None
                series_ids = list(series_ids) if series_ids is not None else None
            
            ts_emb = ts_emb.to(device)
            desc_emb = desc_emb.to(device)
            if perf_emb is not None:
                perf_emb = perf_emb.to(device)
            if labels is not None:
                labels = labels.to(device)
            
            # Forward pass with mixed precision if enabled
            model_ref = _unwrap_model(model)
            
            # Series-level alignment: Mean-pool TS segments and descriptions by series_id
            if getattr(model_ref, "series_level_alignment", False) and series_ids is not None:
                # Group by series_id
                unique_series = list(dict.fromkeys(series_ids))
                
                pooled_ts_emb = []
                pooled_desc_emb = []
                pooled_descriptions = []
                pooled_series_ids = []
                pooled_perf_emb = [] if perf_emb is not None else None
                pooled_labels = [] if labels is not None else None
                
                for s_id in unique_series:
                    indices = [i for i, sid in enumerate(series_ids) if sid == s_id]
                    idx_tensor = torch.tensor(indices, device=device)
                    
                    pooled_ts_emb.append(ts_emb[idx_tensor].mean(dim=0))
                    pooled_desc_emb.append(desc_emb[idx_tensor].mean(dim=0))
                    pooled_descriptions.append(descriptions[indices[0]] if descriptions is not None else "")
                    pooled_series_ids.append(s_id)
                    
                    if pooled_perf_emb is not None:
                        pooled_perf_emb.append(perf_emb[idx_tensor].mean(dim=0))
                    if pooled_labels is not None:
                        pooled_labels.append((labels[idx_tensor] > 0).any().float())
                
                ts_emb = torch.stack(pooled_ts_emb)
                desc_emb = torch.stack(pooled_desc_emb)
                descriptions = pooled_descriptions if descriptions is not None else None
                series_ids = pooled_series_ids
                if pooled_perf_emb is not None:
                    perf_emb = torch.stack(pooled_perf_emb)
                if pooled_labels is not None:
                    labels = torch.stack(pooled_labels)
                
            target_emb = perf_emb if getattr(model_ref, "align_to_performance", False) and perf_emb is not None else desc_emb

            if use_autocast:
                with autocast('cuda', dtype=dtype):
                    ts_proj, target_proj = model(ts_emb, target_emb)
                    loss, stats = model_ref.compute_loss(
                        ts_proj=ts_proj, 
                        desc_proj=target_proj, 
                        descriptions=descriptions, 
                        desc_emb=target_emb, 
                        series_ids=series_ids, 
                        ts_emb=ts_emb,
                        labels=labels
                    )
            else:
                ts_proj, target_proj = model(ts_emb, target_emb)
                loss, stats = model_ref.compute_loss(
                    ts_proj=ts_proj, 
                    desc_proj=target_proj, 
                    descriptions=descriptions, 
                    desc_emb=target_emb, 
                    series_ids=series_ids, 
                    ts_emb=ts_emb,
                    labels=labels
                )
            
            # Track statistics
            total_duplicates += stats.get("num_duplicates", 0)
            total_unique_descs += stats.get("num_unique_descs", len(ts_emb))
            
            total_loss += loss.item()
            num_batches += 1
            
            # Compute accuracy
            batch_size = ts_emb.size(0)
            similarity_matrix = torch.matmul(ts_proj, target_proj.t())
            
            # Multi-positive: use recall@k (any positive in top-k). Otherwise diagonal.
            use_multi_positive = (
                hasattr(model_ref, "_build_match_mask")
                and ((descriptions is not None and len(descriptions) == batch_size)
                     or (hasattr(model_ref, "desc_match_mode") and model_ref.desc_match_mode == "embedding_similarity" and desc_emb is not None))
            )
            if use_multi_positive:
                positive_mask, _, _ = model_ref._build_match_mask(
                    descriptions, desc_emb, batch_size, device
                )
                # For each ts_i: correct if any positive is in top-k
                _, top1_pred = similarity_matrix.max(dim=1)
                batch_top1 = positive_mask[torch.arange(batch_size, device=device), top1_pred].sum().item()
                _, top5_pred = similarity_matrix.topk(5, dim=1)
                batch_top5 = (
                    positive_mask[torch.arange(batch_size, device=device).unsqueeze(1), top5_pred]
                    .any(dim=1).sum().item()
                )
            else:
                # Diagonal (1:1 pairing)
                _, top1_pred = similarity_matrix.max(dim=1)
                batch_top1 = (top1_pred == torch.arange(batch_size, device=device)).sum().item()
                _, top5_pred = similarity_matrix.topk(5, dim=1)
                labels = torch.arange(batch_size, device=device).unsqueeze(1)
                batch_top5 = (top5_pred == labels).any(dim=1).sum().item()
            
            correct_top1 += batch_top1
            correct_top5 += batch_top5
            total_samples += batch_size
    
    total_loss, num_batches, correct_top1, correct_top5, total_samples, total_duplicates, total_unique_descs = _reduce_sums(
        [total_loss, num_batches, correct_top1, correct_top5, total_samples, total_duplicates, total_unique_descs],
        device,
    )
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    top1_acc = correct_top1 / total_samples if total_samples > 0 else 0.0
    top5_acc = correct_top5 / total_samples if total_samples > 0 else 0.0
    
    eval_stats = {
        "avg_duplicates_per_batch": total_duplicates / num_batches if num_batches > 0 else 0,
        "avg_unique_descs_per_batch": total_unique_descs / num_batches if num_batches > 0 else 0,
        # Diagnostic: how much top-5 adds over top-1 in the current validation setting.
        "top5_minus_top1": top5_acc - top1_acc,
    }
    
    return avg_loss, top1_acc, top5_acc, eval_stats


def get_lr_scheduler(
    optimizer,
    num_epochs: int,
    scheduler_name: str = "warmup_cosine",
    warmup_epochs: int = 0,
    scheduler_kwargs: dict = None,
):
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        num_epochs: Total number of epochs
        scheduler_name: Type of scheduler ('warmup_cosine', 'cosine', 'step', 'exponential', 'plateau', 'constant')
        warmup_epochs: Number of warmup epochs (only used for warmup_cosine)
        scheduler_kwargs: Additional kwargs for specific schedulers
    
    Returns:
        Learning rate scheduler
    """
    if scheduler_kwargs is None:
        scheduler_kwargs = {}
    
    if scheduler_name == "warmup_cosine":
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine annealing after warmup
                progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
                return 0.5 * (1 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_name == "cosine":
        T_max = scheduler_kwargs.get("T_max", num_epochs)
        eta_min = scheduler_kwargs.get("eta_min", 0.0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )
    
    elif scheduler_name == "step":
        step_size = scheduler_kwargs.get("step_size", 30)
        gamma = scheduler_kwargs.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    
    elif scheduler_name == "exponential":
        gamma = scheduler_kwargs.get("gamma", 0.95)
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    elif scheduler_name == "plateau":
        mode = scheduler_kwargs.get("mode", "min")
        factor = scheduler_kwargs.get("factor", 0.5)
        patience = scheduler_kwargs.get("patience", 10)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience
        )
    
    elif scheduler_name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    
    else:
        raise ValueError(f"Unknown scheduler name: {scheduler_name}")


class Trainer:
    """Main trainer class."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        num_epochs: int,
        grad_clip_norm: float,
        checkpoint_dir: str,
        mixed_precision: str = "none",
        use_wandb: bool = False,
        is_rank0: bool = True,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.grad_clip_norm = grad_clip_norm
        self.checkpoint_dir = checkpoint_dir
        self.mixed_precision = mixed_precision
        self.use_wandb = use_wandb and WANDB_AVAILABLE and is_rank0
        self.is_rank0 = is_rank0
        
        # Initialize gradient scaler for fp16 mixed precision
        self.scaler = GradScaler() if mixed_precision == "fp16" else None
        
        # Check if scheduler is ReduceLROnPlateau (requires metric in step())
        self.is_plateau_scheduler = isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        )
        
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.val_top1_accs = []
        self.val_top5_accs = []
    
    def train(self):
        """Run training loop."""
        if self.is_rank0:
            print(f"\n{'='*80}")
            print("Starting training...")
            print(f"{'='*80}\n")
        
        # Print header for epoch summaries
        if self.is_rank0:
            print(f"{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Top-1 Acc':<12} {'Top-5 Acc':<12} {'LR':<12} {'Status':<10}")
            print("-" * 80)
        
        # Overall training progress bar
        epoch_iter = range(self.num_epochs)
        if self.is_rank0:
            epoch_iter = tqdm(
                epoch_iter,
                desc="Training",
                ncols=120,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}',
            )
        
        for epoch in epoch_iter:
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)
            # Train
            train_loss, train_stats = train_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.device,
                self.grad_clip_norm,
                self.mixed_precision,
                self.scaler,
            )
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_top1_acc, val_top5_acc, val_stats = evaluate(
                self.model,
                self.val_loader,
                self.device,
                self.mixed_precision,
            )
            self.val_losses.append(val_loss)
            self.val_top1_accs.append(val_top1_acc)
            self.val_top5_accs.append(val_top5_acc)
            
            # Print duplicate statistics on first epoch or if duplicates detected
            if self.is_rank0 and (epoch == 0 or train_stats.get("avg_duplicates_per_batch", 0) > 0):
                print(f"  Duplicates: train={train_stats.get('avg_duplicates_per_batch', 0):.1f}/batch, "
                      f"val={val_stats.get('avg_duplicates_per_batch', 0):.1f}/batch")
                print(f"  Unique descs: train={train_stats.get('avg_unique_descs_per_batch', 0):.1f}/batch, "
                      f"val={val_stats.get('avg_unique_descs_per_batch', 0):.1f}/batch")
            
            # Update learning rate
            # ReduceLROnPlateau requires a metric value
            if self.is_plateau_scheduler:
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best and self.is_rank0:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, val_top1_acc, val_top5_acc, is_best=True)
                status = "BEST"
            else:
                status = ""
            
            # Log to wandb
            if self.use_wandb:
                log_dict = {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/top1_acc": val_top1_acc,
                    "val/top5_acc": val_top5_acc,
                    "val/top5_minus_top1": val_stats.get("top5_minus_top1", val_top5_acc - val_top1_acc),
                    "learning_rate": self.scheduler.get_last_lr()[0],
                }
                # Add duplicate stats if available
                if train_stats.get("avg_duplicates_per_batch", 0) > 0:
                    log_dict["train/avg_duplicates_per_batch"] = train_stats.get("avg_duplicates_per_batch", 0)
                    log_dict["val/avg_duplicates_per_batch"] = val_stats.get("avg_duplicates_per_batch", 0)
                if train_stats.get("avg_unique_descs_per_batch", 0) > 0:
                    log_dict["train/avg_unique_descs_per_batch"] = train_stats.get("avg_unique_descs_per_batch", 0)
                    log_dict["val/avg_unique_descs_per_batch"] = val_stats.get("avg_unique_descs_per_batch", 0)
                
                wandb.log(log_dict)
            
            # Update overall progress bar with current metrics
            if self.is_rank0:
                epoch_iter.set_postfix({
                    "train_loss": f"{train_loss:.4f}",
                    "val_loss": f"{val_loss:.4f}",
                    "top1": f"{val_top1_acc*100:.1f}%",
                    "top5": f"{val_top5_acc*100:.1f}%",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                    "status": status,
                }, refresh=True)
            
            # Print epoch summary line
            summary_line = f"{epoch+1:6d} {train_loss:11.4f} {val_loss:11.4f} {val_top1_acc*100:11.2f}% {val_top5_acc*100:11.2f}% {self.scheduler.get_last_lr()[0]:11.2e} {status:<10}"
            if self.is_rank0:
                print(summary_line)
        
        # Close progress bar
        if self.is_rank0 and hasattr(epoch_iter, "close"):
            epoch_iter.close()
        
        # Save final model
        if self.is_rank0:
            self._save_checkpoint(
                self.num_epochs - 1,
                self.val_losses[-1],
                self.val_top1_accs[-1],
                self.val_top5_accs[-1],
                is_best=False,
                is_final=True,
            )
    
    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        val_top1_acc: float,
        val_top5_acc: float,
        is_best: bool = False,
        is_final: bool = False,
    ):
        """Save model checkpoint."""
        from pathlib import Path
        
        checkpoint_dir = Path(self.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle DataParallel: save underlying model, not the wrapper
        model_state_dict = self.model.state_dict()
        if isinstance(self.model, nn.DataParallel):
            model_state_dict = self.model.module.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_top1_acc': val_top1_acc,
            'val_top5_acc': val_top5_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_top1_accs': self.val_top1_accs,
            'val_top5_accs': self.val_top5_accs,
        }
        
        if is_best:
            torch.save(checkpoint, checkpoint_dir / "best_model.pt")
        elif is_final:
            torch.save(checkpoint, checkpoint_dir / "final_model.pt")

