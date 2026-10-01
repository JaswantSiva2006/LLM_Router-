# Kaggle router training

This launcher trains the existing Dr.LLM routers on a deterministic 90/10 split
of 2,916 examples with two
NVIDIA T4 GPUs. Each rank owns one frozen Qwen2.5-1.5B model, one router copy,
and one dataset shard. The global effective batch size is `2 x 1 x 8 = 16`.
Router inputs use Qwen's user chat template with the assistant generation marker,
matching MCTS generation. The gold answer is never included. Model revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306` is pinned throughout.

## Fresh notebook

```bash
!nvidia-smi
!git clone https://github.com/JaswantSiva2006/LLM_Router-.git
%cd LLM_Router-
!python -m pip install -r requirements.txt
!test -f router_train_data.jsonl || cp "$(find /kaggle/input -name router_train_data.jsonl -print -quit)" router_train_data.jsonl
!bash scripts/kaggle_train.sh
```

`kaggle_train.sh` runs `kaggle_router_preflight.sh` before launching DDP. To run
the preflight by itself, use `!bash scripts/kaggle_router_preflight.sh`.

The final artifacts are written beneath `/kaggle/working/router_training_v2/`.
Save that directory as notebook output before the Kaggle session ends.

## Resume in the same working directory

```bash
!bash scripts/kaggle_train.sh --resume latest
```

## Resume in a fresh notebook

Attach the previous notebook output as a Kaggle input, restore it, then resume:

```bash
!mkdir -p /kaggle/working/router_training_v2
!cp -a /kaggle/input/YOUR_PREVIOUS_OUTPUT/router_training_v2/. /kaggle/working/router_training_v2/
!bash scripts/kaggle_train.sh --resume latest
```

T4 GPUs do not support BF16, so this launcher uses FP16 autocast and GradScaler.
The preflight loads one model on GPU 0 for a smoke update before checking the
two-rank NCCL launch; training itself then starts with one model replica per GPU.
