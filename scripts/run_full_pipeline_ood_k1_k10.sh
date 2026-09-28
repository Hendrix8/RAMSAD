#!/usr/bin/env bash
# End-to-end: Python venv, install deps, segment generation, MPNet desc embeddings,
# Chronos embeddings, alignment training (OOD train split, target domain held out), aligned retrieval
# with top_k=K_MAX, then model selection for every k in 1..K_MAX using the same
# retrieval CSV.
#
# Usage (from anywhere):
#   bash /path/to/ramsad/scripts/run_full_pipeline_ood_k1_k10.sh
#   INSTALL_CUDA124=1 bash scripts/run_full_pipeline_ood_k1_k10.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO"

: "${PYTHON_CMD:=python3}"
: "${INSTALL_CUDA124:=0}"
: "${INSTALL_CUDA118:=0}"
: "${SKIP_VENV:=0}"
: "${SKIP_INSTALL:=0}"
: "${SKIP_GENERATE_SEGMENTS:=0}"
: "${FORCE_SEGMENTS:=0}"
: "${SKIP_EMBED_DESC:=0}"
: "${SKIP_CHRONOS:=0}"
: "${SKIP_ALIGNMENT:=0}"
: "${SKIP_RETRIEVE:=0}"
: "${SKIP_EVAL:=0}"
RAMSAD_SPLIT=ood
: "${RAMSAD_OOD_DOMAIN:=all}"
: "${OOD_DOMAINS:=${RAMSAD_OOD_DOMAIN}}"
: "${ALIGN_DESC_COL:=all-mpnet-base-v2_desc}"
: "${K_MAX:=10}"
: "${RET_CSV_OVERRIDE:=}"
: "${TRAIN_CSV:=data/raw/VUS/train.csv}"
: "${TEST_CSV:=data/raw/VUS/test.csv}"
: "${RAW_ROOT:=}"
: "${DESCRIPTIONS_CSV:=data/raw/descriptions.csv}"
: "${RAMSAD_DATA_ROOT:=$REPO/data/processed_data}"
export RAMSAD_DATA_ROOT

# Auto-detect CUDA capability if neither INSTALL_CUDA124 nor INSTALL_CUDA118 is forced
if [[ "${INSTALL_CUDA124}" == "0" && "${INSTALL_CUDA118}" == "0" ]]; then
  if command -v nvidia-smi &> /dev/null; then
    CUDA_VER=$(nvidia-smi | grep -i "CUDA Version" | sed -E 's/.*CUDA Version: ([0-9]+)\.([0-9]+).*/\1 \2/' || true)
    if [[ -n "$CUDA_VER" ]]; then
      read -r MAJOR MINOR <<< "$CUDA_VER"
      if [[ "$MAJOR" -lt 12 || ( "$MAJOR" -eq 12 && "$MINOR" -lt 1 ) ]]; then
        echo "Detected CUDA $MAJOR.$MINOR: Auto-selecting cu118 PyTorch wheels."
        INSTALL_CUDA118=1
      elif [[ "$MAJOR" -ge 12 && "$MINOR" -ge 4 ]]; then
        echo "Detected CUDA $MAJOR.$MINOR: Auto-selecting cu124 PyTorch wheels."
        INSTALL_CUDA124=1
      else
        echo "Detected CUDA $MAJOR.$MINOR: Using default PyTorch wheels (cu121)."
      fi
    fi
  fi
fi

VENV="$REPO/.venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
UV="$VENV/bin/uv"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -n "${RAMSAD_OUTPUTS_ROOT:-}" ]]; then
  : "${RUN_ROOT:=$RAMSAD_OUTPUTS_ROOT}"
else
  : "${RUN_ROOT:=$REPO/outputs/full_pipeline_ood_$STAMP}"
fi
mkdir -p "$RUN_ROOT"

banner() {
  echo ""
  echo "================================================================================"
  echo "$1"
  echo "================================================================================"
}

if [[ "${SKIP_VENV}" != "1" ]]; then
  if [[ ! -x "$PY" ]]; then
    banner "Creating venv: $VENV"
    "$PYTHON_CMD" -m venv "$VENV"
  fi
else
  if [[ ! -x "$PY" ]]; then
    echo "ERROR: SKIP_VENV=1 but $PY missing" >&2
    exit 1
  fi
fi

if [[ "${SKIP_INSTALL}" != "1" ]]; then
  banner "Installing package (editable)"
  "$PIP" install -U pip wheel setuptools uv
  if [[ "${INSTALL_CUDA124}" == "1" ]]; then
    "$UV" pip install -e ".[dev,alignment,cuda124]" \
      --extra-index-url https://download.pytorch.org/whl/cu124
  elif [[ "${INSTALL_CUDA118}" == "1" ]]; then
    "$UV" pip install -e ".[dev,alignment]" \
      --extra-index-url https://download.pytorch.org/whl/cu118
  else
    "$UV" pip install -e ".[dev,alignment]"
  fi
fi

if [[ "$RAMSAD_OOD_DOMAIN" == "all" ]]; then
  DOMAINS=(environment facility finance humanactivity medical sensor synthetic traffic webservice)
  RUN_ROOT="${RAMSAD_OUTPUTS_ROOT:-$REPO/outputs}/full_pipeline_ood_all_$(date +%Y%m%dT%H%M%SZ)"
  mkdir -p "$RUN_ROOT"
  
  banner "Pre-generating ID segments & embeddings for all domains (OOD split preparation)"
  
  if [[ "${SKIP_GENERATE_SEGMENTS}" != "1" ]]; then
    SEG_CMD=("$PY" -m ramsad.generate_segments --id --train-csv "$TRAIN_CSV" --test-csv "$TEST_CSV" --data-root "$RAMSAD_DATA_ROOT")
    if [[ -n "${RAW_ROOT}" ]]; then
      SEG_CMD+=(--raw-root "$RAW_ROOT")
    fi
    if [[ -n "${DESCRIPTIONS_CSV}" ]]; then
      SEG_CMD+=(--descriptions-csv "$DESCRIPTIONS_CSV")
    fi
    if [[ "${FORCE_SEGMENTS}" == "1" ]]; then
      SEG_CMD+=(--force)
    fi
    "${SEG_CMD[@]}"
  fi
  
  if [[ "${SKIP_EMBED_DESC}" != "1" ]]; then
    banner "Description embeddings for ID (MPNet)"
    "$PY" -m ramsad.run_embed_desc data=id hydra.run.dir="$RUN_ROOT/hydra_embed_desc_id"
  fi
  
  if [[ "${SKIP_CHRONOS}" != "1" ]]; then
    banner "Chronos embeddings for ID"
    "$PY" -m ramsad.run_embed experiment=paper_id_single hydra.run.dir="$RUN_ROOT/hydra_chronos_id"
  fi
  
  banner "Splitting ID parquets into all OOD domains"
  "$PY" "$SCRIPT_DIR/build_ood_splits_from_id.py" \
    --id-root "$RAMSAD_DATA_ROOT/id" \
    --out-root "$RAMSAD_DATA_ROOT/ood" \
    --domains "$(IFS=,; echo "${DOMAINS[*]}")" \
    --train-csv "$TRAIN_CSV" \
    --test-csv "$TEST_CSV"
  
  for domain in "${DOMAINS[@]}"; do
    banner "Running pipeline for domain: $domain"
    SKIP_GENERATE_SEGMENTS=1 SKIP_EMBED_DESC=1 SKIP_CHRONOS=1 SKIP_VENV=1 SKIP_INSTALL=1 SKIP_INDIVIDUAL_EVAL_PRINT=1 \
    RAMSAD_OOD_DOMAIN="$domain" RAMSAD_OUTPUTS_ROOT="$RUN_ROOT/ood_${domain}" bash "$SCRIPT_DIR/$(basename "$0")"
  done
  
  if [[ "${SKIP_EVAL}" != "1" ]]; then
    banner "Evaluating Aggregated Selection Accuracy over all domains (k=1..${K_MAX})"
    "$PY" "$SCRIPT_DIR/evaluate_all_ood_k_sweep.py" \
      --root-dir "$RUN_ROOT" \
      --k-max "${K_MAX}"
  fi
  
  exit 0
fi

if [[ "${RAMSAD_SPLIT}" == "ood" ]]; then
  EXP=(experiment=paper_ood_single "data.ood_domain=${RAMSAD_OOD_DOMAIN}")
  EMBED_DESC_EXTRA=(data=ood "data.ood_domain=${RAMSAD_OOD_DOMAIN}")
else
  EXP=(experiment=paper_id_single)
  EMBED_DESC_EXTRA=()
fi

banner "Ramsad OOD full pipeline — outputs under: $RUN_ROOT"

if [[ "${SKIP_GENERATE_SEGMENTS}" != "1" ]]; then
  banner "Segment generation (${RAMSAD_SPLIT})"
  SEG_CMD=("$PY" -m ramsad.generate_segments)
  if [[ "${RAMSAD_SPLIT}" == "ood" ]]; then
    SEG_CMD+=(--ood --domains "$OOD_DOMAINS")
  else
    SEG_CMD+=(--id)
  fi
  SEG_CMD+=(--train-csv "$TRAIN_CSV" --test-csv "$TEST_CSV" --data-root "$RAMSAD_DATA_ROOT")
  if [[ -n "${RAW_ROOT}" ]]; then
    SEG_CMD+=(--raw-root "$RAW_ROOT")
  fi
  if [[ -n "${DESCRIPTIONS_CSV}" ]]; then
    SEG_CMD+=(--descriptions-csv "$DESCRIPTIONS_CSV")
  fi
  if [[ "${FORCE_SEGMENTS}" == "1" ]]; then
    SEG_CMD+=(--force)
  fi
  "${SEG_CMD[@]}"
else
  echo "[skip] segment generation"
fi

if [[ "${SKIP_EMBED_DESC}" != "1" ]]; then
  banner "Description embeddings (MPNet)"
  "$PY" -m ramsad.run_embed_desc "${EMBED_DESC_EXTRA[@]}" \
    hydra.run.dir="$RUN_ROOT/hydra_embed_desc"
else
  echo "[skip] embed_desc"
fi

if [[ "${SKIP_CHRONOS}" != "1" ]]; then
  banner "Chronos embeddings"
  "$PY" -m ramsad.run_embed "${EXP[@]}" hydra.run.dir="$RUN_ROOT/hydra_chronos"
else
  echo "[skip] chronos embed"
fi

if [[ "${SKIP_ALIGNMENT}" != "1" ]]; then
  # Leave-domain-out: train alignment on the OOD train split only (target domain excluded).
  ALIGN_TRAIN_SEGMENTS="$RAMSAD_DATA_ROOT/ood/OOD_${RAMSAD_OOD_DOMAIN}/train_segments_OOD_${RAMSAD_OOD_DOMAIN}.parquet"
  ALIGN_TRAIN_PERF="$RAMSAD_DATA_ROOT/ood/Datasets/OOD/train_perf_OOD_${RAMSAD_OOD_DOMAIN}.csv"
  banner "Alignment training (OOD train split, target domain '${RAMSAD_OOD_DOMAIN}' held out)"
  # Trains from repo root; writes e.g. outputs/alignment/run-* under cwd or RAMSAD_OUTPUTS_ROOT
  (cd "$REPO" && "$PY" alignment/train.py \
    "data.desc_emb_col=${ALIGN_DESC_COL}" \
    "data.parquet_file=${ALIGN_TRAIN_SEGMENTS}" \
    "data.performance_file=${ALIGN_TRAIN_PERF}")
else
  echo "[skip] alignment train"
fi

if [[ "${SKIP_RETRIEVE}" != "1" ]]; then
  banner "Retrieval (mode=alignment, top_k=${K_MAX})"
  "$PY" -m ramsad.run_retrieve "${EXP[@]}" \
    retrieve.mode=alignment \
    "retrieve.top_k=${K_MAX}" \
    hydra.run.dir="$RUN_ROOT/retrieve"
else
  echo "[skip] retrieve"
fi

if [[ -n "${RET_CSV_OVERRIDE:-}" ]]; then
  RET_CSV_ABS="$("$PY" -c "from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())" "${RET_CSV_OVERRIDE}")"
else
  RET_CSV="$RUN_ROOT/retrieve/retrieval_topk.csv"
  RET_CSV_ABS="$("$PY" -c "from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())" "$RET_CSV")"
fi
if [[ ! -f "$RET_CSV_ABS" ]]; then
  echo "ERROR: missing retrieval CSV: $RET_CSV_ABS" >&2
  exit 1
fi

if [[ "${SKIP_EVAL}" != "1" ]]; then
  banner "Model selection for k=1..${K_MAX} (shared retrieval CSV)"
  mkdir -p "$RUN_ROOT/eval_k"
  for k in $(seq 1 "${K_MAX}"); do
    echo "--- select k=${k} ---"
    "$PY" -m ramsad.run_select "${EXP[@]}" \
      "select.k=${k}" \
      "select.retrieval_csv=${RET_CSV_ABS}" \
      "select.output_csv_name=model_selection_k${k}.csv" \
      hydra.run.dir="$RUN_ROOT/eval_k/k${k}"
  done

  if [[ "${SKIP_INDIVIDUAL_EVAL_PRINT:-0}" != "1" ]]; then
    banner "Evaluating Selection Accuracy over k=1..${K_MAX}"
    "$PY" "$SCRIPT_DIR/evaluate_k_sweep.py" \
      --k-dir "$RUN_ROOT/eval_k" \
      --test-csv "$RAMSAD_DATA_ROOT/ood/Datasets/OOD/test_perf_OOD_${RAMSAD_OOD_DOMAIN}.csv"
  fi
else
  echo "[skip] eval k-sweep"
fi

cat > "$RUN_ROOT/SUMMARY.txt" << EOF
Run root: $RUN_ROOT
split: ${RAMSAD_SPLIT} (ood_domain=${RAMSAD_OOD_DOMAIN})
retrieval CSV: $RET_CSV_ABS
selection outputs: $RUN_ROOT/eval_k/k*/model_selection_k*.csv
alignment checkpoints: see alignment training log (Output directory:) or \${RAMSAD_OUTPUTS_ROOT:-$REPO/outputs}/alignment/
EOF

banner "Done. See $RUN_ROOT/SUMMARY.txt"
