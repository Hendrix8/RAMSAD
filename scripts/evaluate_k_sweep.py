import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def normalize_ts_key(name: str) -> str:
    return str(name).strip().removesuffix(".csv")

def main():
    parser = argparse.ArgumentParser(description="Evaluate selected models against true test performance.")
    parser.add_argument("--k-dir", type=str, required=True, help="Directory containing k1, k2, etc. folders")
    parser.add_argument("--test-csv", type=str, required=True, help="Path to true performance CSV (e.g. data/raw/VUS/test.csv)")
    parser.add_argument("--perf-col", type=str, default="file", help="Column in test-csv identifying the series")
    parser.add_argument("--out-csv", type=str, default="k_sweep_evaluation.csv", help="Name of output summary CSV")
    args = parser.parse_args()

    k_dir = Path(args.k_dir)
    test_csv_path = Path(args.test_csv)

    if not k_dir.exists():
        print(f"Error: k-dir {k_dir} does not exist.")
        sys.exit(1)
    
    if not test_csv_path.exists():
        print(f"Error: test-csv {test_csv_path} does not exist.")
        sys.exit(1)

    df_test = pd.read_csv(test_csv_path)
    df_test[args.perf_col] = df_test[args.perf_col].astype(str).apply(normalize_ts_key)
    
    # Identify numeric columns (models) for Oracle
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

    # Set index for easy lookup
    df_test.set_index(args.perf_col, inplace=True)
    # Ensure numeric types
    for c in model_cols:
        df_test[c] = pd.to_numeric(df_test[c], errors="coerce")

    # Find k values
    k_vals = []
    for p in k_dir.glob("k*"):
        if p.is_dir() and p.name[1:].isdigit():
            k_vals.append(int(p.name[1:]))
    k_vals.sort()

    if not k_vals:
        print(f"No 'k*' folders found in {k_dir}")
        sys.exit(0)

    results = []

    for k in k_vals:
        sel_path = k_dir / f"k{k}" / f"model_selection_k{k}.csv"
        if not sel_path.exists():
            continue
        
        df_sel = pd.read_csv(sel_path)
        
        achieved_scores = []
        oracle_scores = []

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
                

                
        mean_achieved = np.mean(achieved_scores) if achieved_scores else 0.0
        mean_oracle = np.mean(oracle_scores) if oracle_scores else 0.0

        
        results.append({
            "k": k,
            "Mean_VUS_PR": mean_achieved,
            "Oracle_VUS_PR": mean_oracle,
            "N_series": len(achieved_scores)
        })

    df_res = pd.DataFrame(results)
    
    out_path = k_dir.parent / args.out_csv
    df_res.to_csv(out_path, index=False)
    
    print("\n" + "="*80)
    print("K-Sweep Evaluation Results")
    print("="*80)
    print(df_res.to_string(index=False))
    print(f"\nSaved evaluation to {out_path}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
