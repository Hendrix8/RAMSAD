import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

def normalize_ts_key(name: str) -> str:
    return str(name).strip().removesuffix(".csv")

def main():
    parser = argparse.ArgumentParser(description="Evaluate selected models against true test performance across ALL OOD domains.")
    parser.add_argument("--root-dir", type=str, required=True, help="Directory containing ood_*/ folders (e.g. outputs/full_pipeline_ood_all_...)")
    parser.add_argument("--k-max", type=int, default=10, help="Maximum k value to evaluate")
    parser.add_argument("--perf-col", type=str, default="file", help="Column in test-csv identifying the series")
    parser.add_argument("--out-csv", type=str, default="ood_all_k_sweep_evaluation.csv", help="Name of output summary CSV")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)

    if not root_dir.exists():
        print(f"Error: root-dir {root_dir} does not exist.")
        sys.exit(1)

    domains = [
        "environment", "facility", "finance", "humanactivity", "medical",
        "sensor", "synthetic", "traffic", "webservice"
    ]

    # Resolve RAMSAD_DATA_ROOT if present
    data_root_env = os.environ.get("RAMSAD_DATA_ROOT", "data/processed_data")
    data_root = Path(data_root_env)

    results = []

    for k in range(1, args.k_max + 1):
        achieved_scores = []
        oracle_scores = []
        
        for domain in domains:
            sel_path = root_dir / f"ood_{domain}" / "eval_k" / f"k{k}" / f"model_selection_k{k}.csv"
            test_csv_path = data_root / "ood" / "Datasets" / "OOD" / f"test_perf_OOD_{domain}.csv"
            
            if not sel_path.exists() or not test_csv_path.exists():
                continue
            
            df_sel = pd.read_csv(sel_path)
            df_test = pd.read_csv(test_csv_path)
            df_test[args.perf_col] = df_test[args.perf_col].astype(str).apply(normalize_ts_key)
            
            # Identify numeric columns for Oracle
            skip_cols = {args.perf_col, "anomaly_len", "anomaly_ratio", "avg_anomaly_len", "num_anomaly", "point_anomaly", "seq_anomaly", "ts_len"}
            model_cols = []
            for c in df_test.columns:
                if c in skip_cols: continue
                if pd.api.types.is_numeric_dtype(df_test[c]):
                    model_cols.append(c)
                else:
                    s = pd.to_numeric(df_test[c], errors="coerce")
                    if s.notna().sum() > len(df_test) * 0.5:
                        model_cols.append(c)
                        
            df_test.set_index(args.perf_col, inplace=True)
            for c in model_cols:
                df_test[c] = pd.to_numeric(df_test[c], errors="coerce")
                
            for _, row in df_sel.iterrows():
                ts = normalize_ts_key(row["time_series"])
                model = str(row["selected_model"]).strip()
                
                if ts not in df_test.index:
                    continue
                    
                test_row = df_test.loc[ts]
                
                # Achieved score
                if model in test_row and pd.notna(test_row[model]):
                    achieved_scores.append(test_row[model])
                else:
                    achieved_scores.append(0.0)
                    
                # Oracle score
                oracle_score = test_row[model_cols].max()
                if pd.notna(oracle_score):
                    oracle_scores.append(oracle_score)

        if len(achieved_scores) == 0:
            continue
            
        mean_achieved = np.mean(achieved_scores)
        mean_oracle = np.mean(oracle_scores)
        
        results.append({
            "k": k,
            "Mean_VUS_PR": mean_achieved,
            "Oracle_VUS_PR": mean_oracle,
            "N_series": len(achieved_scores)
        })

    if not results:
        print(f"No evaluation results found across domains in {root_dir}")
        sys.exit(0)
        
    df_res = pd.DataFrame(results)
    out_path = root_dir / args.out_csv
    df_res.to_csv(out_path, index=False)
    
    print("\n" + "="*80)
    print("OOD ALL-Domains K-Sweep Evaluation Results")
    print("="*80)
    print(df_res.to_string(index=False))
    print(f"\nSaved evaluation to {out_path}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
