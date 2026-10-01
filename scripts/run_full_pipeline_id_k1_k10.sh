#!/usr/bin/env bash
# End-to-end: Python venv, install deps, segment generation, MPNet desc embeddings,
# Chronos embeddings, alignment training (ID segments by default), aligned retrieval
# with top_k=K_MAX, then model selection for every k in 1..K_MAX using the same
# retrieval CSV.
#
# Usage (from anywhere):
#   bash /path/to/ramsad/scripts/run_full_pipeline_id_k1_k10.sh
#   INSTALL_CUDA124=1 bash scripts/run_full_pipeline_id_k1_k10.sh
#
# Environment (optional):
#   PYTHON_CMD        default python3
#   INSTALL_CUDA124   set to 1 to pass PyTorch CUDA 12.4 index (see README)
#   SKIP_VENV         set to 1 to skip creating .venv
#   SKIP_INSTALL      set to 1 to skip pip install
#   SKIP_GENERATE_SEGMENTS set to 1
#   FORCE_SEGMENTS    set to 1 to overwrite existing segment/series parquets
#   TRAIN_CSV         default data/splits/id/train.csv
#   TEST_CSV          default data/splits/id/test.csv
#   RAW_ROOT          optional raw AD root with Train/ and Test/ folders
#   DESCRIPTIONS_CSV  optional source,description CSV
#   SKIP_EMBED_DESC   set to 1
#   SKIP_CHRONOS      set to 1
#   SKIP_ALIGNMENT    set to 1 (use existing outputs/alignment/run-*)
#   SKIP_RETRIEVE     set to 1
#   RET_CSV_OVERRIDE  path to existing retrieval_topk.csv if SKIP_RETRIEVE=1 (required for eval)
#   SKIP_EVAL         set to 1 (skip k=1..K_MAX selection)
#   ALIGN_DESC_COL    data.desc_emb_col for alignment (default all-mpnet-base-v2_desc)
#   K_MAX             default 10 (retrieval top_k and max k for selection)
#   SELECT_N          number of detectors to select per series (default 1; N>1 = ensemble)
#   RUN_ROOT          output folder; default outputs/full_pipeline_<UTC stamp>

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
RAMSAD_SPLIT=id
: "${RAMSAD_OOD_DOMAIN:=sensor}"
: "${OOD_DOMAINS:=${RAMSAD_OOD_DOMAIN}}"
: "${ALIGN_DESC_COL:=all-mpnet-base-v2_desc}"
: "${K_MAX:=10}"
: "${SELECT_N:=1}"
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
: "${RUN_ROOT:=$REPO/outputs/full_pipeline_id_$STAMP}"
mkdir -p "$RUN_ROOT"

banner() {
  echo ""
  echo "================================================================================"
  echo "$1"
  echo "================================================================================"
}

if [[ "${RAMSAD_SPLIT}" == "ood" ]]; then
  EXP=(experiment=paper_ood_single "data.ood_domain=${RAMSAD_OOD_DOMAIN}")
  EMBED_DESC_EXTRA=(data=ood "data.ood_domain=${RAMSAD_OOD_DOMAIN}")
else
  EXP=(experiment=paper_id_single)
  EMBED_DESC_EXTRA=()
fi

banner "Ramsad ID full pipeline — outputs under: $RUN_ROOT"

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
  banner "Alignment training (default ID parquets in alignment/config)"
  # Trains from repo root; writes e.g. outputs/alignment/run-* under cwd or RAMSAD_OUTPUTS_ROOT
  (cd "$REPO" && "$PY" alignment/train.py "data.desc_emb_col=${ALIGN_DESC_COL}")
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
      "select.n=${SELECT_N}" \
      "select.retrieval_csv=${RET_CSV_ABS}" \
      "select.output_csv_name=model_selection_k${k}.csv" \
      hydra.run.dir="$RUN_ROOT/eval_k/k${k}"
  done

  banner "Evaluating Selection Accuracy over k=1..${K_MAX}"
  "$PY" "$SCRIPT_DIR/evaluate_k_sweep.py" \
    --k-dir "$RUN_ROOT/eval_k" \
    --test-csv "$TEST_CSV"
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
