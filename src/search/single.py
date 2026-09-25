"""Run length-aware MCTS for one GSM8K training question."""

from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

from src.data.gsm8k import iter_gsm8k
from src.model.routed_qwen import enable_qwen2_routing, generate_with_route
from src.search.mcts import EarlyStopMode, MCTSConfig, MCTSSearch


def _get_record(index: int):
    if index < 0:
        raise ValueError("index must be non-negative")
    records = list(islice(iter_gsm8k(streaming=True), index, index + 1))
    if not records:
        raise IndexError(f"GSM8K train index {index} does not exist")
    return records[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--num-simulations", type=int, default=50)
    parser.add_argument("--early-stop", choices=[mode.value for mode in EarlyStopMode], default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        parser.error("CUDA is required for Qwen route evaluation")

    record = _get_record(args.index)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = Qwen2ForCausalLM.from_pretrained(args.model, dtype=torch.float16).to("cuda")
    if model.config.num_hidden_layers != 28:
        raise ValueError(f"expected a 28-layer Qwen2 model, got {model.config.num_hidden_layers}")
    enable_qwen2_routing(model)
    config = MCTSConfig(
        num_simulations=args.num_simulations,
        seed=args.seed,
        early_stop=args.early_stop,
    )
    search = MCTSSearch(config)

    def evaluate(path: list[int]) -> str:
        return generate_with_route(
            model,
            tokenizer,
            record.generation_prompt,
            path,
            max_new_tokens=args.max_new_tokens,
        )

    result = search.run(
        question_id=record.dataset_id,
        question=record.question,
        gold_answer=record.gold_answer,
        evaluate_route=evaluate,
    )
    output = args.output or Path("data/mcts_raw") / f"gsm8k_train_{args.index:05d}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_dict(), ensure_ascii=False) + "\n", encoding="utf-8")
    selected = result.selected_pi_star
    print(f"question_id: {record.dataset_id}")
    print(f"evaluated: {result.metadata['unique_paths_evaluated']} unique paths")
    print(f"baseline: {result.baseline.verifier_status}")
    print(f"pi*: {list(selected.execution_path) if selected else None}")
    print(f"output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

