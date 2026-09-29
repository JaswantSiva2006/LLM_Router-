#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASET_PATH="${DATASET_PATH:-$REPO_ROOT/router_train_data.jsonl}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-2916}"

python -m src.router_training.kaggle_preflight \
  --data "$DATASET_PATH" \
  --expected-samples "$EXPECTED_SAMPLES" \
  --model "$MODEL_NAME"

torchrun --standalone --nproc_per_node=2 -m src.router_training.ddp_smoke

echo "KAGGLE ROUTER TRAINING READY: YES"
