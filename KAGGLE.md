# Kaggle execution guide

The production target is a Kaggle notebook with **2 × NVIDIA T4 (16 GB each)**.
The runner uses two independent spawned processes: physical GPU 0 is visible
only to worker 0, and physical GPU 1 only to worker 1. It does not use DDP or
model parallelism.

## 1. Open the repository

```bash
cd /kaggle/working/Kaggle_DRLLM
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

If the repository is attached under a different directory, change the first
command accordingly. Confirm that Kaggle enabled two GPUs:

```bash
nvidia-smi -L
```

## 2. Lightweight checks

```bash
python -m pytest -q
python -m src.grading.gsm8k_verifier --self-test
```

## 3. Validate routed Qwen on the Kaggle T4

This loads one model and performs the Phase 2 identity/hook/determinism checks:

```bash
python -m src.model.validate_routed_qwen \
  --model Qwen/Qwen2.5-1.5B-Instruct
```

## 4. Run a one-question MCTS smoke test

```bash
python -m src.search.single \
  --index 0 \
  --num-simulations 50 \
  --early-stop none \
  --output runs/smoke_index_00000.jsonl
```

Inspect the one-record output before starting the dual-GPU pilot.

## 5. Run only the 100-question pilot

```bash
python -m src.run_generation \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset openai/gsm8k \
  --split train \
  --start 0 \
  --count 100 \
  --num-simulations 50 \
  --early-stop none \
  --gpus 0,1 \
  --seed 42 \
  --output runs/pilot_0000_0099
```

If the notebook is interrupted, rerun the **exact same command**. Each worker
repairs only an incomplete trailing JSONL fragment, reads completed question
IDs, and never recomputes those IDs. Do not change configuration while reusing
the same output directory.

## 6. Audit the pilot

```bash
python -m src.audit.run runs/pilot_0000_0099
python -m src.audit.run runs/pilot_0000_0099 --json \
  > runs/pilot_0000_0099/audit.json
```

## 7. Export examples for human review

```bash
python -m src.export.audit_examples \
  --run runs/pilot_0000_0099 \
  --n 20 \
  > runs/pilot_0000_0099/audit_examples.txt
```

Inspect `audit.json`, `audit_examples.txt`, both worker checkpoints, and
`merged.jsonl` before authorizing any larger run.

## 8. Pin the validated model revision

All subsequent runs must use the exact immutable model revision recorded by
the pilot:

```bash
MODEL_REVISION=$(python -c 'import json; print(json.load(open("runs/pilot_0000_0099/run_manifest.json"))["model_revision"])')
echo "$MODEL_REVISION"
```

Do not continue unless the pilot audit and human review are satisfactory.

## 9. Search the remaining GSM8K training questions

The official `main/train` split has indices 0–7472. The pilot already covers
0–99, so the non-overlapping remainder is:

```bash
python -m src.run_generation \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --revision "$MODEL_REVISION" \
  --dataset openai/gsm8k \
  --split train \
  --start 100 \
  --count 7373 \
  --num-simulations 50 \
  --early-stop none \
  --gpus 0,1 \
  --seed 42 \
  --output runs/full_0100_7472
```

Rerun this exact command after an interruption. Completed IDs are not
recomputed.

## 10. Create and search the unresolved-ID list

```bash
python -m src.unresolved \
  --runs runs/ \
  --output artifacts/unresolved_ids.txt
```

If the file is nonempty, run the requested 100-simulation second pass:

```bash
python -m src.run_generation \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --revision "$MODEL_REVISION" \
  --dataset openai/gsm8k \
  --split train \
  --ids-file artifacts/unresolved_ids.txt \
  --num-simulations 100 \
  --early-stop none \
  --gpus 0,1 \
  --seed 42 \
  --output runs/retry_unresolved_100sim
```

## 11. Replay and finalize exactly 4,000 samples

The finalizer first checks that at least 4,000 unique eligible samples exist.
It then independently replays selected routes and refuses to publish a partial
or weakened dataset.

```bash
python -m src.finalize \
  --runs runs/ \
  --target 4000 \
  --output artifacts/gsm8k_mcts_4k.jsonl
```

An interrupted finalization can be resumed with the same command; completed
replays are read from `artifacts/gsm8k_mcts_4k_replay.jsonl`.

Verify the artifacts:

```bash
gzip -t artifacts/gsm8k_mcts_4k_provenance.jsonl.gz
python - <<'PY'
import json
from pathlib import Path

canonical = Path("artifacts/gsm8k_mcts_4k.jsonl")
rows = [json.loads(line) for line in canonical.open(encoding="utf-8")]
summary = json.load(open("artifacts/gsm8k_mcts_4k_summary.json", encoding="utf-8"))
assert len(rows) == 4000
assert summary["rows"] == 4000
assert summary["unique_ids"] == 4000
assert summary["unique_questions"] == 4000
assert summary["replay_failures"] == 0
print(json.dumps(summary, indent=2, sort_keys=True))
PY
```

## Router training

Run these steps in order after the repository and, for steps 7 onward, the
final dataset have been attached to the Kaggle notebook. None of these commands
is launched automatically by the repository.

### 1. Check the GPUs

```bash
nvidia-smi
```

Confirm that Kaggle exposes two NVIDIA T4 16 GB GPUs.

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

### 3. Verify the Transformers version

```bash
python -c 'import transformers; print(transformers.__version__)'
```

It must match the version pinned in `requirements.txt` and `pyproject.toml`.

### 4. Run the test suite

```bash
python -m pytest -q
```

### 5. Validate the routed router model

```bash
python -m src.router_training.validate_model \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --precision fp16
```

### 6. Smoke train on 8 examples

```bash
python -m src.router_training.smoke_train \
  --data tests/fixtures/router_training_small.jsonl \
  --allow-small-dataset \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --samples 8 \
  --precision fp16
```

Do not continue unless the smoke test reports decreasing loss, changed router
weights, and unchanged Qwen weights.

### 7. Verify the final 4k dataset

```bash
python -m src.router_training.verify_readiness \
  --data artifacts/gsm8k_mcts_4k.jsonl \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --precision fp16
```

Continue only when this prints `ROUTER TRAINING READY: YES`.

### 8. Run two-GPU diagnostic 3600/400 training

```bash
torchrun \
  --standalone \
  --nproc_per_node=2 \
  -m src.router_training.train \
  --data artifacts/gsm8k_mcts_4k.jsonl \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --epochs 25 \
  --microbatch 1 \
  --gradient-accumulation 8 \
  --precision fp16 \
  --val-fraction 0.10 \
  --output /kaggle/working/runs/router_train_diagnostic
```

### 9. Inspect metrics

```bash
tail -n 5 /kaggle/working/runs/router_train_diagnostic/metrics.jsonl
ls -lh /kaggle/working/runs/router_train_diagnostic/
```

Inspect validation loss, macro F1, class proportions, executed-layer means,
exact-path accuracy, and per-layer accuracy before considering final training.

### 10. Optionally run final all-4000 training

```bash
torchrun \
  --standalone \
  --nproc_per_node=2 \
  -m src.router_training.train \
  --data artifacts/gsm8k_mcts_4k.jsonl \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --epochs 25 \
  --microbatch 1 \
  --gradient-accumulation 8 \
  --precision fp16 \
  --val-fraction 0 \
  --output /kaggle/working/runs/router_train_final
```
