import pytest

from src.data.gsm8k import PROMPT_TEMPLATE, adapt_record, iter_gsm8k, parse_gold_answer


def test_record_has_stable_identity_and_provenance():
    item = {"question": "What is 600 + 600?", "answer": "Add them.\n600 + 600 = 1200\n#### 1,200"}
    record = adapt_record(item, 7)
    assert record.dataset_id == "openai/gsm8k:main:train:00007"
    assert record.index == 7
    assert record.question == item["question"]
    assert record.gold_reasoning == "Add them.\n600 + 600 = 1200"
    assert record.gold_answer == "1200"
    assert record.generation_prompt == PROMPT_TEMPLATE.format(question=item["question"])
    assert "ONLY return the final result" in record.generation_prompt
    assert r"\boxed{...}" in record.generation_prompt


def test_only_final_marker_is_canonical_answer():
    reasoning, answer = parse_gold_answer("mention #### 2 inline\nwork\n#### -0.75")
    assert reasoning == "mention #### 2 inline\nwork"
    assert answer == "-3/4"


def test_missing_or_bad_final_marker_rejected():
    with pytest.raises(ValueError):
        parse_gold_answer("No marker here")
    with pytest.raises(ValueError):
        parse_gold_answer("work\n#### many")


def test_generation_loader_rejects_non_train_split_before_loading():
    with pytest.raises(ValueError, match="only permits.*train"):
        next(iter_gsm8k("test"))
