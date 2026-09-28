"""Embedding alignment model for time series and description embeddings."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class EmbeddingAlignmentModel(nn.Module):
    """Model to align time series embeddings with description embeddings."""
    
    def __init__(
        self,
        ts_emb_dim: int = 768,
        desc_emb_dim: int = 768,
        projection_dim: int = 256,
        temperature: float = 0.1,
        loss_function: str = "contrastive",
        desc_match_mode: str = "exact",
        desc_similarity_threshold: float = 0.95,
        per_series_loss: bool = False,
        preserve_temporal_identity: bool = False,
        identity_penalty_weight: float = 1.0,
        series_level_alignment: bool = False,
        align_to_performance: bool = False,
        performance_dim: int = 32,
    ):
        """
        Args:
            ts_emb_dim: Dimension of time series embeddings
            desc_emb_dim: Dimension of description embeddings
            projection_dim: Dimension of projected embeddings
            temperature: Temperature for contrastive loss
            loss_function: Loss function type - "contrastive", "distance", "cosine", or "multi_positive"
            desc_match_mode: How to match descriptions for positive pairs - "exact" (string equality)
                or "embedding_similarity" (cosine sim of raw desc embeddings >= threshold)
            desc_similarity_threshold: Cosine similarity threshold for embedding_similarity mode (0-1)
            per_series_loss: If True, only compare with segments from different series (exclude same series from loss)
        """
        super().__init__()
        self.temperature = temperature
        self.loss_function = loss_function
        self.desc_match_mode = desc_match_mode
        self.desc_similarity_threshold = desc_similarity_threshold
        self.per_series_loss = per_series_loss
        self.preserve_temporal_identity = preserve_temporal_identity
        self.identity_penalty_weight = identity_penalty_weight
        self.series_level_alignment = series_level_alignment
        self.align_to_performance = align_to_performance
        self.performance_dim = performance_dim
        
        if loss_function not in ["contrastive", "distance", "cosine", "multi_positive"]:
            raise ValueError(f"loss_function must be 'contrastive', 'distance', 'cosine', or 'multi_positive', got '{loss_function}'")
        if desc_match_mode not in ["exact", "embedding_similarity"]:
            raise ValueError(f"desc_match_mode must be 'exact' or 'embedding_similarity', got '{desc_match_mode}'")
        
        # Identity projection layer for preserving temporal features
        if self.preserve_temporal_identity:
            self.ts_reconstruct = nn.Sequential(
                nn.Linear(projection_dim, projection_dim * 2),
                nn.ReLU(),
                nn.Linear(projection_dim * 2, ts_emb_dim),
            )
            
        # Projection heads for both embeddings
        self.ts_projection = nn.Sequential(
            nn.Linear(ts_emb_dim, projection_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim * 2, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        
        self.desc_projection = nn.Sequential(
            nn.Linear(desc_emb_dim, projection_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim * 2, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        
        if self.align_to_performance:
            self.perf_projection = nn.Sequential(
                nn.Linear(performance_dim, projection_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(projection_dim * 2, projection_dim),
                nn.LayerNorm(projection_dim),
            )
    
    def forward(self, ts_emb: torch.Tensor, target_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ts_emb: Time series embeddings [batch_size, ts_emb_dim]
            target_emb: Either Description embeddings or Performance embeddings [batch_size, desc_emb_dim] or [batch_size, perf_dim]
        
        Returns:
            ts_proj: Projected time series embeddings [batch_size, projection_dim]
            target_proj: Projected target embeddings [batch_size, projection_dim]
        """
        ts_proj = self.ts_projection(ts_emb)
        
        if self.align_to_performance:
            target_proj = self.perf_projection(target_emb)
        else:
            target_proj = self.desc_projection(target_emb)
        
        # L2 normalize for cosine similarity
        ts_proj = F.normalize(ts_proj, p=2, dim=1)
        target_proj = F.normalize(target_proj, p=2, dim=1)
        
        return ts_proj, target_proj
    
    def _build_match_mask(
        self,
        descriptions: Optional[list[str]],
        desc_emb: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Build the match matrix for positive/duplicate pairs.
        Returns (match_matrix, num_duplicates, num_unique_descs).
        """
        if self.desc_match_mode == "embedding_similarity" and desc_emb is not None:
            # Use cosine similarity of raw 768-dim description embeddings
            desc_emb_norm = F.normalize(desc_emb, p=2, dim=1)
            sim_matrix = torch.matmul(desc_emb_norm, desc_emb_norm.t())
            match_matrix = sim_matrix >= self.desc_similarity_threshold
            # Filter empty descriptions if provided
            if descriptions is not None and len(descriptions) == batch_size:
                desc_array = np.array(descriptions)
                non_empty_mask = torch.from_numpy(desc_array != "").to(device)
                match_matrix = match_matrix & (non_empty_mask[:, None] & non_empty_mask[None, :])
            elif getattr(self, "align_to_performance", False):
                # If we are aligning to performance, we don't have descriptions, but we want all valid ones
                pass
            
            num_duplicates = int(
                (match_matrix.sum().item() - batch_size) // 2
            )  # exclude diagonal, count pairs once
            num_unique = batch_size  # approximate when using similarity
            if descriptions is not None:
                num_unique = len(set([d for d in descriptions if d != ""]))
            elif getattr(self, "align_to_performance", False):
                # Count unique rows in the embedding tensor
                unique_tensor = torch.unique(desc_emb, dim=0)
                num_unique = len(unique_tensor)
        else:
            # Exact string match (original behavior)
            if descriptions is None or len(descriptions) != batch_size:
                return (
                    torch.eye(batch_size, dtype=torch.bool, device=device),
                    0,
                    batch_size,
                )
            desc_array = np.array(descriptions)
            non_empty_mask = desc_array != ""
            desc_expanded_i = desc_array[:, None]
            desc_expanded_j = desc_array[None, :]
            match_matrix = (
                (desc_expanded_i == desc_expanded_j)
                & non_empty_mask[:, None]
                & non_empty_mask[None, :]
            )
            match_matrix = torch.from_numpy(match_matrix).to(device)
            num_duplicates = int(np.sum(np.triu(match_matrix.cpu().numpy(), k=1)))
            num_unique = len(set([d for d in descriptions if d != ""]))
        return match_matrix, num_duplicates, num_unique
    
    def _build_exclude_same_series_mask(
        self, series_ids: list, batch_size: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        """Build mask [batch_size, batch_size]: True = exclude (same series)."""
        if not self.per_series_loss or series_ids is None or len(series_ids) != batch_size:
            return None
        arr = np.array(series_ids)
        non_empty = arr != ""
        same = (arr[:, None] == arr[None, :]) & non_empty[:, None] & non_empty[None, :]
        return torch.from_numpy(same).to(device)

    def compute_contrastive_loss(
        self,
        ts_proj: torch.Tensor,
        desc_proj: torch.Tensor,
        descriptions: list[str] = None,
        desc_emb: torch.Tensor = None,
        series_ids: list = None,
        labels: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute InfoNCE contrastive loss with duplicate description handling.
        
        Args:
            ts_proj: Projected time series embeddings [batch_size, projection_dim]
            desc_proj: Projected description embeddings [batch_size, projection_dim]
            descriptions: List of description strings for duplicate detection [batch_size]
            desc_emb: Raw description embeddings [batch_size, 768] for embedding_similarity mode
        
        Returns:
            loss: Contrastive loss scalar
            stats: Dictionary with statistics (num_duplicates, num_unique_descs)
        """
        batch_size = ts_proj.size(0)
        stats = {"num_duplicates": 0, "num_unique_descs": batch_size}
        
        # Compute similarity matrix (cosine similarity = dot product after normalization)
        # [batch_size, batch_size]
        similarity_matrix = torch.matmul(ts_proj, desc_proj.t()) / self.temperature

        # Exclude same-series pairs from comparison (per_series_loss)
        exclude_mask = self._build_exclude_same_series_mask(
            series_ids, batch_size, ts_proj.device
        )
        if exclude_mask is not None:
            similarity_matrix = similarity_matrix.clone()
            similarity_matrix[exclude_mask] = -1e9
        
        # Positive pairs are on the diagonal (i-th TS matches i-th description)
            target_labels = torch.arange(batch_size, device=ts_proj.device)
            
            # Loss from TS -> Desc perspective
            loss_ts_to_desc_per_sample = F.cross_entropy(masked_similarity, target_labels, reduction='none')
            
            # Loss from Desc -> TS perspective (symmetric)
            loss_desc_to_ts_per_sample = F.cross_entropy(masked_similarity.t(), target_labels, reduction='none')
            
            if getattr(self, "align_anomalies_only", False) and labels is not None:
                anomaly_mask = labels > 0.0
                loss_ts_to_desc_per_sample = loss_ts_to_desc_per_sample * anomaly_mask
                loss_desc_to_ts_per_sample = loss_desc_to_ts_per_sample * anomaly_mask
                if anomaly_mask.sum() > 0:
                    loss_ts_to_desc = loss_ts_to_desc_per_sample.sum() / anomaly_mask.sum()
                    loss_desc_to_ts = loss_desc_to_ts_per_sample.sum() / anomaly_mask.sum()
                else:
                    loss_ts_to_desc = torch.tensor(0.0, device=ts_proj.device)
                    loss_desc_to_ts = torch.tensor(0.0, device=ts_proj.device)
            else:
                loss_ts_to_desc = loss_ts_to_desc_per_sample.mean()
                loss_desc_to_ts = loss_desc_to_ts_per_sample.mean()
        else:
            # Standard loss without duplicate handling
            target_labels = torch.arange(batch_size, device=ts_proj.device)
            loss_ts_to_desc_per_sample = F.cross_entropy(similarity_matrix, target_labels, reduction='none')
            loss_desc_to_ts_per_sample = F.cross_entropy(similarity_matrix.t(), target_labels, reduction='none')
            
            if getattr(self, "align_anomalies_only", False) and labels is not None:
                anomaly_mask = labels > 0.0
                loss_ts_to_desc_per_sample = loss_ts_to_desc_per_sample * anomaly_mask
                loss_desc_to_ts_per_sample = loss_desc_to_ts_per_sample * anomaly_mask
                if anomaly_mask.sum() > 0:
                    loss_ts_to_desc = loss_ts_to_desc_per_sample.sum() / anomaly_mask.sum()
                    loss_desc_to_ts = loss_desc_to_ts_per_sample.sum() / anomaly_mask.sum()
                else:
                    loss_ts_to_desc = torch.tensor(0.0, device=ts_proj.device)
                    loss_desc_to_ts = torch.tensor(0.0, device=ts_proj.device)
            else:
                loss_ts_to_desc = loss_ts_to_desc_per_sample.mean()
                loss_desc_to_ts = loss_desc_to_ts_per_sample.mean()
        
        # Average both directions
        loss = (loss_ts_to_desc + loss_desc_to_ts) / 2.0
        
        return loss, stats
    
    def compute_distance_loss(
        self,
        ts_proj: torch.Tensor,
        desc_proj: torch.Tensor,
        descriptions: list[str] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute simple distance loss that minimizes distance between matching pairs.
        This avoids the problem of contrastive loss where similar descriptions
        are forced apart.
        
        Args:
            ts_proj: Projected time series embeddings [batch_size, projection_dim]
            desc_proj: Projected description embeddings [batch_size, projection_dim]
            descriptions: List of description strings (unused for distance loss, kept for API compatibility)
        
        Returns:
            loss: Distance loss scalar (mean squared distance between matching pairs)
            stats: Dictionary with statistics (empty for distance loss)
        """
        batch_size = ts_proj.size(0)
        stats = {"num_duplicates": 0, "num_unique_descs": batch_size}
        
        # Compute distance between matching pairs (diagonal)
        # Since embeddings are L2 normalized, distance = 2 - 2 * cosine_similarity
        # Or we can use squared L2 distance directly
        # For normalized vectors: ||a - b||^2 = 2 - 2 * (a · b)
        # We want to minimize this, so we compute mean squared distance
        matching_pairs_dist = torch.sum((ts_proj - desc_proj) ** 2, dim=1)
        loss = matching_pairs_dist.mean()
        
        return loss, stats
    
    def compute_cosine_loss(
        self,
        ts_proj: torch.Tensor,
        desc_proj: torch.Tensor,
        descriptions: list[str] = None,
        labels: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute cosine similarity loss that maximizes cosine similarity between matching pairs.
        This directly optimizes for high similarity without using a similarity matrix,
        avoiding the contrastive loss problem with duplicate descriptions.
        
        Args:
            ts_proj: Projected time series embeddings [batch_size, projection_dim]
            desc_proj: Projected description embeddings [batch_size, projection_dim]
            descriptions: List of description strings (unused for cosine loss, kept for API compatibility)
        
        Returns:
            loss: Cosine loss scalar (1 - mean cosine similarity between matching pairs)
            stats: Dictionary with statistics (empty for cosine loss)
        """
        batch_size = ts_proj.size(0)
        stats = {"num_duplicates": 0, "num_unique_descs": batch_size}
        
        # Compute cosine similarity for matching pairs (diagonal)
        # Since embeddings are L2 normalized, cosine similarity = dot product
        # We want to maximize cosine similarity, so we minimize (1 - cosine_similarity)
        cosine_similarities = torch.sum(ts_proj * desc_proj, dim=1)  # [batch_size]
        losses = 1.0 - cosine_similarities
        
        if getattr(self, "align_anomalies_only", False) and labels is not None:
            anomaly_mask = labels > 0.0
            losses = losses * anomaly_mask
            if anomaly_mask.sum() > 0:
                loss = losses.sum() / anomaly_mask.sum()
            else:
                loss = torch.tensor(0.0, device=ts_proj.device)
        else:
            loss = losses.mean()
        
        return loss, stats
    
    def compute_multi_positive_loss(
        self,
        ts_proj: torch.Tensor,
        desc_proj: torch.Tensor,
        descriptions: list[str] = None,
        desc_emb: torch.Tensor = None,
        series_ids: list = None,
        labels: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute multi-positive contrastive loss that handles multiple positive pairs per anchor.
        This naturally handles duplicate descriptions by treating all matching descriptions
        as positives for each time series.
        
        Args:
            ts_proj: Projected time series embeddings [batch_size, projection_dim]
            desc_proj: Projected description embeddings [batch_size, projection_dim]
            descriptions: List of description strings for identifying positive pairs [batch_size]
            desc_emb: Raw description embeddings [batch_size, 768] for embedding_similarity mode
        
        Returns:
            loss: Multi-positive contrastive loss scalar
            stats: Dictionary with statistics (num_duplicates, num_unique_descs)
        """
        batch_size = ts_proj.size(0)
        stats = {"num_duplicates": 0, "num_unique_descs": batch_size}
        
        # Compute similarity matrix (cosine similarity = dot product after normalization)
        # [batch_size, batch_size]
        similarity_matrix = torch.matmul(ts_proj, desc_proj.t()) / self.temperature

        # Exclude same-series pairs from comparison (per_series_loss)
        exclude_mask = self._build_exclude_same_series_mask(
            series_ids, batch_size, ts_proj.device
        )
        if exclude_mask is not None:
            similarity_matrix = similarity_matrix.clone()
            similarity_matrix[exclude_mask] = -1e9
        
        use_match_mask = (
            (descriptions is not None and len(descriptions) == batch_size)
            or (self.desc_match_mode == "embedding_similarity" and desc_emb is not None)
        )
        if use_match_mask:
            positive_mask, num_duplicates, num_unique = self._build_match_mask(
                descriptions, desc_emb, batch_size, ts_proj.device
            )
            stats["num_duplicates"] = num_duplicates
            stats["num_unique_descs"] = num_unique
            # Exclude same-series from positives when per_series_loss
            if exclude_mask is not None:
                positive_mask = positive_mask & ~exclude_mask
            
            # Compute multi-positive loss from TS -> Desc perspective (vectorized)
            # For each TS i, sum over all positive descriptions j where descriptions[i] == descriptions[j]
            exp_sim = torch.exp(similarity_matrix)  # [batch_size, batch_size]
            
            # Vectorized computation: sum over columns (desc dimension) for each TS anchor
            # pos_sum[i] = sum of exp(similarities) for all positive descriptions j for TS i
            pos_sum_ts_to_desc = (exp_sim * positive_mask).sum(dim=1)  # [batch_size]
            # all_sum[i] = sum of exp(similarities) for all descriptions j for TS i
            all_sum_ts_to_desc = exp_sim.sum(dim=1)  # [batch_size]
            
            # Handle cases where there are no positives (set loss to 0 for those samples)
            has_positives_ts_to_desc = pos_sum_ts_to_desc > 0
            loss_per_sample_ts_to_desc = torch.zeros(batch_size, device=ts_proj.device)
            loss_per_sample_ts_to_desc[has_positives_ts_to_desc] = -torch.log(
                pos_sum_ts_to_desc[has_positives_ts_to_desc] / all_sum_ts_to_desc[has_positives_ts_to_desc]
            )
            
            if getattr(self, "align_anomalies_only", False) and labels is not None:
                anomaly_mask = labels > 0.0
                loss_per_sample_ts_to_desc = loss_per_sample_ts_to_desc * anomaly_mask
                if anomaly_mask.sum() > 0:
                    loss_ts_to_desc = loss_per_sample_ts_to_desc.sum() / anomaly_mask.sum()
                else:
                    loss_ts_to_desc = torch.tensor(0.0, device=ts_proj.device)
            else:
                loss_ts_to_desc = loss_per_sample_ts_to_desc.mean()
            
            # Compute multi-positive loss from Desc -> TS perspective (symmetric, vectorized)
            # For each Desc j, sum over all positive TS i where descriptions[i] == descriptions[j]
            # pos_sum[j] = sum of exp(similarities) for all positive TS i for Desc j
            pos_sum_desc_to_ts = (exp_sim * positive_mask).sum(dim=0)  # [batch_size]
            # all_sum[j] = sum of exp(similarities) for all TS i for Desc j
            all_sum_desc_to_ts = exp_sim.sum(dim=0)  # [batch_size]
            
            # Handle cases where there are no positives (set loss to 0 for those samples)
            has_positives_desc_to_ts = pos_sum_desc_to_ts > 0
            loss_per_sample_desc_to_ts = torch.zeros(batch_size, device=ts_proj.device)
            loss_per_sample_desc_to_ts[has_positives_desc_to_ts] = -torch.log(
                pos_sum_desc_to_ts[has_positives_desc_to_ts] / all_sum_desc_to_ts[has_positives_desc_to_ts]
            )
            
            if getattr(self, "align_anomalies_only", False) and labels is not None:
                anomaly_mask = labels > 0.0
                loss_per_sample_desc_to_ts = loss_per_sample_desc_to_ts * anomaly_mask
                if anomaly_mask.sum() > 0:
                    loss_desc_to_ts = loss_per_sample_desc_to_ts.sum() / anomaly_mask.sum()
                else:
                    loss_desc_to_ts = torch.tensor(0.0, device=ts_proj.device)
            else:
                loss_desc_to_ts = loss_per_sample_desc_to_ts.mean()
        else:
            # Fallback to standard contrastive loss if no descriptions provided
            target_labels = torch.arange(batch_size, device=ts_proj.device)
            loss_ts_to_desc_per_sample = F.cross_entropy(similarity_matrix, target_labels, reduction='none')
            loss_desc_to_ts_per_sample = F.cross_entropy(similarity_matrix.t(), target_labels, reduction='none')
            
            if getattr(self, "align_anomalies_only", False) and labels is not None:
                anomaly_mask = labels > 0.0
                loss_ts_to_desc_per_sample = loss_ts_to_desc_per_sample * anomaly_mask
                loss_desc_to_ts_per_sample = loss_desc_to_ts_per_sample * anomaly_mask
                if anomaly_mask.sum() > 0:
                    loss_ts_to_desc = loss_ts_to_desc_per_sample.sum() / anomaly_mask.sum()
                    loss_desc_to_ts = loss_desc_to_ts_per_sample.sum() / anomaly_mask.sum()
                else:
                    loss_ts_to_desc = torch.tensor(0.0, device=ts_proj.device)
                    loss_desc_to_ts = torch.tensor(0.0, device=ts_proj.device)
            else:
                loss_ts_to_desc = loss_ts_to_desc_per_sample.mean()
                loss_desc_to_ts = loss_desc_to_ts_per_sample.mean()
        
        # Average both directions
        loss = (loss_ts_to_desc + loss_desc_to_ts) / 2.0
        
        return loss, stats
    
    def compute_loss(
        self,
        ts_proj: torch.Tensor,
        desc_proj: torch.Tensor,
        descriptions: list[str] = None,
        desc_emb: torch.Tensor = None,
        series_ids: list = None,
        ts_emb: torch.Tensor = None,
        labels: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Unified loss computation that dispatches to the appropriate loss function.
        
        Args:
            ts_proj: Projected time series embeddings [batch_size, projection_dim]
            desc_proj: Projected description embeddings [batch_size, projection_dim]
            descriptions: List of description strings for duplicate detection [batch_size]
            desc_emb: Raw description embeddings (or performance embeddings) [batch_size, 768 or 32] for embedding_similarity mode
            series_ids: List of series identifiers [batch_size]. When per_series_loss=True, excludes same-series from comparison.
            ts_emb: Original time series embeddings [batch_size, 768] for reconstruction penalty
            labels: Normal (0) or Anomaly (1) segment label tensor [batch_size]
        
        Returns:
            loss: Loss scalar
            stats: Dictionary with statistics
        """
        if self.loss_function == "contrastive":
            base_loss, stats = self.compute_contrastive_loss(ts_proj, desc_proj, descriptions, desc_emb, series_ids, labels=labels)
        elif self.loss_function == "distance":
            base_loss, stats = self.compute_distance_loss(ts_proj, desc_proj, descriptions)
        elif self.loss_function == "cosine":
            base_loss, stats = self.compute_cosine_loss(ts_proj, desc_proj, descriptions, labels=labels)
        elif self.loss_function == "multi_positive":
            base_loss, stats = self.compute_multi_positive_loss(ts_proj, desc_proj, descriptions, desc_emb, series_ids, labels=labels)
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")
            
        if self.preserve_temporal_identity and ts_emb is not None:
            # Reconstruct the original 768-D representation from the 256-D projection
            ts_reconstructed = self.ts_reconstruct(ts_proj)
            identity_loss = F.mse_loss(ts_reconstructed, ts_emb)
            stats["alignment_loss"] = base_loss.item()
            stats["identity_loss"] = identity_loss.item()
            return base_loss + (self.identity_penalty_weight * identity_loss), stats
        
        return base_loss, stats

