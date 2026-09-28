#!/usr/bin/env bash

# Simple launcher for DDP training of the embedding alignment model.
# Uses all visible GPUs on the current node.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect number of GPUs (falls back to 1 if nvidia-smi is not available)
if command -v nvidia-smi >/dev/null 2>&1; then
  NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
  echo "nvidia-smi not found; defaulting to 1 process."
  NUM_GPUS=1
fi

if [ "$NUM_GPUS" -lt 1 ]; then
  echo "No GPUs detected, running single-process on CPU."
  python train.py "$@"
  exit 0
fi

echo "Launching DDP training on $NUM_GPUS GPUs..."

torchrun --nproc-per-node="$NUM_GPUS" train.py "$@"

