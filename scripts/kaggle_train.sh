#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASET_PATH="${DATASET_PATH:-$REPO_ROOT/router_train_data.jsonl}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
MODEL_REVISION="${MODEL_REVISION:-989aa7980e4cf806f80c7fef2b1adb7bc71aa306}"
EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-2916}"
OUTPUT_DIR="${OUTPUT_DIR:-/kaggle/working/router_training_v2}"

DATASET_PATH="$DATASET_PATH" \
MODEL_NAME="$MODEL_NAME" \
MODEL_REVISION="$MODEL_REVISION" \
EXPECTED_SAMPLES="$EXPECTED_SAMPLES" \
bash scripts/kaggle_router_preflight.sh

mkdir -p "$OUTPUT_DIR/checkpoints"

torchrun --standalone --nproc_per_node=2 -m src.router_training.train \
  --data "$DATASET_PATH" \
  --model "$MODEL_NAME" \
  --model-revision "$MODEL_REVISION" \
  --epochs 25 \
  --microbatch 1 \
  --gradient-accumulation 8 \
  --precision fp16 \
  --val-fraction 0.10 \
  --num-workers 2 \
  --allow-small-dataset \
  --output "$OUTPUT_DIR" \
  "$@"
