# Kaggle router training

This launcher trains the existing Dr.LLM routers on 2,916 examples with two
NVIDIA T4 GPUs. Each rank owns one frozen Qwen2.5-1.5B model, one router copy,
and one dataset shard. The global effective batch size is `2 x 1 x 8 = 16`.

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

The final artifacts are written beneath `/kaggle/working/router_training/`.
Save that directory as notebook output before the Kaggle session ends.

## Resume in the same working directory

```bash
!bash scripts/kaggle_train.sh --resume latest
```

## Resume in a fresh notebook

Attach the previous notebook output as a Kaggle input, restore it, then resume:

```bash
!mkdir -p /kaggle/working/router_training
!cp -a /kaggle/input/YOUR_PREVIOUS_OUTPUT/router_training/. /kaggle/working/router_training/
!bash scripts/kaggle_train.sh --resume latest
```

T4 GPUs do not support BF16, so this launcher uses FP16 autocast and GradScaler.
The preflight loads one model on GPU 0 for a smoke update before checking the
two-rank NCCL launch; training itself then starts with one model replica per GPU.
