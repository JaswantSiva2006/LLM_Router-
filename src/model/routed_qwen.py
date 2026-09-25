"""Weight-preserving arbitrary execution paths for Hugging Face Qwen2 models.

The adaptation changes only the loop selecting decoder blocks. Parameters and
state-dict keys remain untouched. Repeated-layer generation intentionally
disables the KV cache: a physical Qwen2 layer owns one cache slot, so executing
it twice at a position would otherwise append that position twice.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from typing import Any

import torch
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Model
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

QWEN25_15B_NUM_LAYERS = 28


class LayerAction(IntEnum):
    SKIP = 0
    EXECUTE = 1
    REPEAT = 2


def _validate_num_layers(num_layers: int) -> None:
    if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError("num_layers must be a positive integer")


def labels_to_path(
    labels: Sequence[int | LayerAction], num_layers: int = QWEN25_15B_NUM_LAYERS
) -> list[int]:
    """Convert one SKIP/EXECUTE/REPEAT label per layer to an ordered path."""
    _validate_num_layers(num_layers)
    if isinstance(labels, (str, bytes)) or not isinstance(labels, Sequence):
        raise TypeError("labels must be a sequence")
    if len(labels) != num_layers:
        raise ValueError(f"expected {num_layers} labels, got {len(labels)}")
    path: list[int] = []
    for layer_id, label in enumerate(labels):
        if isinstance(label, bool) or not isinstance(label, (int, LayerAction)):
            raise TypeError(f"label {layer_id} must be an integer LayerAction")
        try:
            count = int(LayerAction(label))
        except ValueError as exc:
            raise ValueError(f"invalid label at layer {layer_id}: {label!r}") from exc
        path.extend([layer_id] * count)
    return path


def validate_path(
    path: Sequence[int], num_layers: int = QWEN25_15B_NUM_LAYERS
) -> list[int]:
    """Validate and return a defensive list copy of a legal execution path.

    Legal paths are nondecreasing, contain only layer IDs ``0..L-1``, execute
    each layer at most twice, and have length at most ``2L``. The empty path is
    legal and represents skipping every decoder block.
    """
    _validate_num_layers(num_layers)
    if isinstance(path, (str, bytes)) or not isinstance(path, Sequence):
        raise TypeError("path must be a sequence of integer layer IDs")
    if len(path) > 2 * num_layers:
        raise ValueError(f"path length {len(path)} exceeds 2L={2 * num_layers}")
    normalized: list[int] = []
    previous = -1
    counts = [0] * num_layers
    for position, layer_id in enumerate(path):
        if isinstance(layer_id, bool) or not isinstance(layer_id, int):
            raise TypeError(f"path entry {position} must be an integer layer ID")
        if not 0 <= layer_id < num_layers:
            raise ValueError(f"layer ID {layer_id} is outside 0..{num_layers - 1}")
        if layer_id < previous:
            raise ValueError("layer path must preserve original nondecreasing order")
        counts[layer_id] += 1
        if counts[layer_id] > 2:
            raise ValueError(f"layer {layer_id} appears more than twice")
        normalized.append(layer_id)
        previous = layer_id
    return normalized


def path_to_labels(
    path: Sequence[int], num_layers: int = QWEN25_15B_NUM_LAYERS
) -> list[LayerAction]:
    """Convert a validated path to one action label per original layer."""
    normalized = validate_path(path, num_layers)
    counts = [0] * num_layers
    for layer_id in normalized:
        counts[layer_id] += 1
    return [LayerAction(count) for count in counts]


class RoutedQwen2Model(Qwen2Model):
    """Qwen2 base model whose decoder loop follows ``layer_indices``."""

    def __init__(self, config):
        super().__init__(config)
        self.layer_indices = list(range(config.num_hidden_layers))

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        route = validate_path(self.layer_indices, self.config.num_hidden_layers)
        has_repeats = len(route) != len(set(route))
        if has_repeats and (use_cache or past_key_values is not None):
            raise ValueError(
                "KV caching is unsafe for repeated physical layers; pass use_cache=False "
                "or use generate_with_route(), which does so automatically"
            )
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer_id in route:
            decoder_layer = self.layers[layer_id]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[layer_id]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


def enable_qwen2_routing(model: Qwen2ForCausalLM) -> Qwen2ForCausalLM:
    """Enable routing in-place without replacing or copying any parameter.

    Changing the Python class of the existing base-model object is intentional:
    the routed subclass adds no modules or parameters, so parameter identity,
    storage pointers, values, and state-dict keys are all preserved exactly.
    """
    if not isinstance(model, Qwen2ForCausalLM):
        raise TypeError("model must be a transformers Qwen2ForCausalLM")
    if not isinstance(model.model, RoutedQwen2Model):
        if type(model.model) is not Qwen2Model:
            raise TypeError(f"unsupported Qwen2 base model subclass: {type(model.model).__name__}")
        model.model.__class__ = RoutedQwen2Model
        model.model.layer_indices = list(range(model.config.num_hidden_layers))
    model.requires_grad_(False)
    model.eval()
    return model


def get_active_route(model: Qwen2ForCausalLM) -> list[int]:
    if not isinstance(model.model, RoutedQwen2Model):
        raise TypeError("routing is not enabled; call enable_qwen2_routing(model) first")
    return validate_path(model.model.layer_indices, model.config.num_hidden_layers)


def generate_with_route(
    model: Qwen2ForCausalLM,
    tokenizer: Any,
    question: str,
    path: Sequence[int] | None = None,
    *,
    max_new_tokens: int = 24,
) -> str:
    """Greedily generate from a Qwen chat prompt and always restore the route."""
    if not isinstance(model.model, RoutedQwen2Model):
        raise TypeError("routing is not enabled; call enable_qwen2_routing(model) first")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    active = (
        list(range(model.config.num_hidden_layers))
        if path is None
        else validate_path(path, model.config.num_hidden_layers)
    )
    original = list(model.model.layer_indices)
    try:
        model.model.layer_indices = active
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(rendered, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        input_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                use_cache=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(output_ids[0, input_length:], skip_special_tokens=True).strip()
    finally:
        model.model.layer_indices = original

