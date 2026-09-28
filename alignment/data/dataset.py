"""Dataset for time series and description embedding pairs."""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False


def _list_array_to_numpy(arr) -> np.ndarray:
    """Convert PyArrow ListArray to numpy (N, emb_dim) in one shot."""
    if hasattr(arr, 'combine_chunks'):  # ChunkedArray from multi-row-group parquet
        arr = arr.combine_chunks()
    flat = np.asarray(arr.values, dtype=np.float32)
    n = len(arr)
    emb_dim = flat.size // n
    return flat.reshape(n, emb_dim)


class EmbeddingAlignmentDataset(Dataset):
    """Dataset for time series and description embedding pairs."""
    
    def __init__(
        self,
        parquet_path: Path,
        ts_emb_col: str,
        desc_emb_col: str,
        desc_text_col: str = "description",
        series_col: str = None,
        val_stratify_col: str = None,
        use_fast_load: bool = True,
        align_to_performance: bool = False,
        performance_file: str = None,
    ):
        """
        Args:
            parquet_path: Path to parquet file
            ts_emb_col: Column name for time series embeddings
            desc_emb_col: Column name for description embeddings
            desc_text_col: Column name for description text (for duplicate detection)
            series_col: Column name for series identifier (e.g. dataset_name). If set, used for per_series_loss.
            val_stratify_col: Column name for stratification.
            label_col: Column name for segment anomaly label.
            use_fast_load: If True, use PyArrow + vectorized conversion (much faster for large files)
        """
        print(f"Loading dataset from: {parquet_path}")
        required_cols = [ts_emb_col, desc_emb_col]
        if desc_text_col:
            required_cols.append(desc_text_col)
        if series_col:
            required_cols.append(series_col)
        if val_stratify_col:
            required_cols.append(val_stratify_col)
        
        # We try to load label_col if available in kwargs or hardcoded to fallback
        self.label_col = "label"  # default fallback if not explicitly passed
        if "label_col" in locals() and locals()["label_col"] is not None:
             self.label_col = locals()["label_col"]
        required_cols.append(self.label_col)
            
        # Deduplicate required_cols to prevent loading duplicate columns if 
        # e.g., desc_text_col and val_stratify_col are the same.
        required_cols = list(dict.fromkeys(required_cols))
        
        # Robust path: pandas + shape validation
        df = pd.read_parquet(parquet_path, columns=required_cols)
        
        # Load performance dictionary if requested
        self.align_to_performance = align_to_performance
        self.perf_dict = {}
        if self.align_to_performance and performance_file:
            try:
                perf_df = pd.read_csv(performance_file)
                # Filter out the 'file' column and other metadata columns that aren't models
                exclude_cols = {"file", "anomaly_len", "anomaly_ratio", "avg_anomaly_len", 
                                "num_anomaly", "point_anomaly", "seq_anomaly", "ts_len"}
                perf_cols = [c for c in perf_df.columns if c not in exclude_cols]
                for _, row in perf_df.iterrows():
                    # Filename typically e.g. "001_UCR_id_1.csv"
                    # dataset_name typically e.g. "001_UCR_id_1"
                    filename = str(row["file"]).strip()
                    dataset_name = filename.removesuffix(".csv")
                    
                    # Store as a float array, filling NaNs with -1.0
                    vals = row[perf_cols].apply(pd.to_numeric, errors="coerce").fillna(-1.0).values
                    self.perf_dict[dataset_name] = vals.astype(np.float32)
                print(f"Loaded performance data for {len(self.perf_dict)} datasets with {len(perf_cols)} dimensions.")
            except Exception as e:
                print(f"Warning: Failed to load performance file {performance_file}: {e}")

        n = len(df)
        ts_vals = df[ts_emb_col].values
        desc_vals = df[desc_emb_col].values

        good_ts = []
        good_desc = []
        good_perf = []
        good_idx = []
        ts_dim = None
        desc_dim = None

        for i, (ts_v, desc_v) in enumerate(zip(ts_vals, desc_vals)):
            if ts_v is None or desc_v is None:
                continue
            try:
                ts_arr = np.asarray(ts_v, dtype=np.float32).reshape(-1)
                desc_arr = np.asarray(desc_v, dtype=np.float32).reshape(-1)
            except Exception:
                continue
            if ts_dim is None:
                ts_dim = ts_arr.size
            if desc_dim is None:
                desc_dim = desc_arr.size
            if ts_arr.size != ts_dim or desc_arr.size != desc_dim:
                # Skip rows with inconsistent embedding sizes
                continue
            
            # Check for performance embedding if alignment is enabled
            perf_arr = None
            if self.align_to_performance:
                ds_name = df[series_col].iloc[i] if series_col in df.columns else ""
                if str(ds_name) in self.perf_dict:
                    perf_arr = self.perf_dict[str(ds_name)]
                else:
                    # Skip rows that don't have performance data when alignment is requested
                    continue

            good_ts.append(ts_arr)
            good_desc.append(desc_arr)
            if perf_arr is not None:
                good_perf.append(perf_arr)
            good_idx.append(i)

        if not good_ts or not good_desc:
            raise ValueError(
                f"No valid embedding rows found in {parquet_path} "
                f"for columns {ts_emb_col}, {desc_emb_col}"
            )

        self.ts_embeddings = np.stack(good_ts)
        self.desc_embeddings = np.stack(good_desc)
        if hasattr(self, "align_to_performance") and self.align_to_performance and good_perf:
            self.performance_embeddings = np.stack(good_perf)
        else:
            # Create a dummy array if not used to avoid breaking downstream
            self.performance_embeddings = np.zeros((len(self.ts_embeddings), 32), dtype=np.float32)

        if desc_text_col and desc_text_col in df.columns:
            desc_series = df[desc_text_col].fillna("").astype(str).values
            self.descriptions = desc_series[good_idx]
        else:
            self.descriptions = np.array([""] * len(self.ts_embeddings), dtype=object)

        if val_stratify_col and val_stratify_col in df.columns:
            stratify_series = df[val_stratify_col].fillna("").astype(str).values
            self.stratify_labels = stratify_series[good_idx]
        else:
            # Fallback to descriptions if stratify column is not provided or not found
            self.stratify_labels = self.descriptions

        if series_col and series_col in df.columns:
            series_series = df[series_col].fillna("").astype(str).values
            self.series_ids = series_series[good_idx]
        else:
            self.series_ids = np.array([""] * len(self.ts_embeddings), dtype=object)
            
        if self.label_col and self.label_col in df.columns:
            # The label column is usually an int or list of ints. We just care if it's > 0 anywhere.
            # Assuming it might be an array of points per segment, we sum it or max it to get 1/0
            labels_series = df[self.label_col].values[good_idx]
            self.labels = []
            for lbl in labels_series:
                if lbl is None:
                    self.labels.append(0)
                elif isinstance(lbl, (list, np.ndarray)):
                    self.labels.append(1 if np.sum(lbl) > 0 else 0)
                else:
                    self.labels.append(1 if float(lbl) > 0 else 0)
            self.labels = np.array(self.labels, dtype=np.float32)
        else:
            self.labels = np.zeros(len(self.ts_embeddings), dtype=np.float32)
        
        self.series_col = series_col
        print(f"Loaded {len(self.ts_embeddings)} samples")
        print(f"TS embedding shape: {self.ts_embeddings.shape}")
        print(f"Description embedding shape: {self.desc_embeddings.shape}")
    
    def __len__(self):
        return len(self.ts_embeddings)
    
    def __getitem__(self, idx):
        ts_emb = torch.tensor(self.ts_embeddings[idx], dtype=torch.float32)
        desc_emb = torch.tensor(self.desc_embeddings[idx], dtype=torch.float32)
        desc_text = self.descriptions[idx] if self.descriptions[idx] is not None else ""
        series_id = self.series_ids[idx] if hasattr(self, "series_ids") and self.series_ids[idx] is not None else ""
        perf_emb = torch.tensor(self.performance_embeddings[idx], dtype=torch.float32) if hasattr(self, "performance_embeddings") else torch.zeros(32, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32) if hasattr(self, "labels") else torch.tensor(0.0, dtype=torch.float32)
        return ts_emb, desc_emb, desc_text, series_id, idx, perf_emb, label

