"""Teacher-forced Qwen2 router model for supervised Dr.LLM training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from src.model.routed_qwen import _create_mask

NUM_LAYERS = 28
QWEN_HIDDEN_SIZE = 1536
ROUTER_HIDDEN_SIZE = 128
NUM_ACTIONS = 3
DEFAULT_WINDOWS = 8


class LayerRouter(nn.Module):
    """One 1536 -> 128 -> 3 layer-action classifier."""

    def __init__(self, hidden_size: int = QWEN_HIDDEN_SIZE) -> None:
        super().__init__()
        self.input = nn.Linear(hidden_size, ROUTER_HIDDEN_SIZE)
        self.activation = nn.GELU(approximate="tanh")
        self.output = nn.Linear(ROUTER_HIDDEN_SIZE, NUM_ACTIONS)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.zeros_(self.input.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.output(self.activation(self.input(hidden_states)))


def windowed_router_logits(
    hidden_states: Tensor,
    attention_mask: Tensor,
    router: nn.Module,
    *,
    num_windows: int = DEFAULT_WINDOWS,
) -> Tensor:
    """Pool valid tokens into equal windows, then average window logits.

    ``hidden_states`` is ``[B,T,d]`` and the returned tensor is ``[B,3]``.
    Padding may be on either side; valid tokens retain their original order.
    """
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [B, T, d]")
    if attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError("attention_mask must have shape [B, T]")
    if isinstance(num_windows, bool) or not isinstance(num_windows, int) or num_windows <= 0:
        raise ValueError("num_windows must be a positive integer")

    results = []
    for sample, mask in zip(hidden_states, attention_mask):
        valid = sample[mask.to(dtype=torch.bool)]
        token_count = valid.shape[0]
        if token_count == 0:
            raise ValueError("each sample must contain at least one non-padding token")
        effective_windows = min(num_windows, token_count)
        window_size = token_count // effective_windows
        used = valid[: effective_windows * window_size]
        means = used.reshape(effective_windows, window_size, sample.shape[-1]).mean(dim=1)
        # Average logits, deliberately not softmax probabilities.
        results.append(router(means).mean(dim=0))
    return torch.stack(results)


@dataclass
class RouterTrainingOutput:
    router_logits: Tensor
    logits: Tensor
    last_hidden_state: Tensor


class TeacherForcedRouterQwen(nn.Module):
    """Frozen Qwen2 with one trainable router at each transformer layer."""

    def __init__(self, base_model: Qwen2ForCausalLM, *, num_windows: int = DEFAULT_WINDOWS) -> None:
        super().__init__()
        if not isinstance(base_model, Qwen2ForCausalLM):
            raise TypeError("base_model must be a Qwen2ForCausalLM")
        if base_model.config.num_hidden_layers != NUM_LAYERS:
            raise ValueError(f"expected a {NUM_LAYERS}-layer Qwen2 model")
        if num_windows <= 0:
            raise ValueError("num_windows must be positive")

        self.base_model = base_model
        self.num_windows = num_windows
        self.base_model.requires_grad_(False)
        self.routers = nn.ModuleList(
            LayerRouter(base_model.config.hidden_size) for _ in range(NUM_LAYERS)
        )
        base_parameter = next(base_model.parameters())
        self.routers.to(device=base_parameter.device, dtype=base_parameter.dtype)

    @property
    def router_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.routers.parameters())

    def _forward_one(
        self,
        input_ids: Tensor | None,
        inputs_embeds: Tensor | None,
        attention_mask: Tensor,
        labels: Tensor,
        position_ids: Tensor | None,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        qwen = self.base_model.model
        if inputs_embeds is None:
            inputs_embeds = qwen.embed_tokens(input_ids)
        sequence_length = inputs_embeds.shape[1]
        cache_position = torch.arange(sequence_length, device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        mask_kwargs = {
            "config": qwen.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        masks = {"full_attention": _create_mask(create_causal_mask, mask_kwargs)}
        if qwen.has_sliding_layers:
            masks["sliding_attention"] = _create_mask(
                create_sliding_window_causal_mask, mask_kwargs
            )

        hidden_states = inputs_embeds
        position_embeddings = qwen.rotary_emb(hidden_states, position_ids)
        all_router_logits = []
        for layer_index, (layer, router) in enumerate(zip(qwen.layers, self.routers)):
            all_router_logits.append(
                windowed_router_logits(
                    hidden_states, attention_mask, router, num_windows=self.num_windows
                )
            )
            action = int(labels[0, layer_index].item())
            if action not in (0, 1, 2):
                raise ValueError(f"invalid router label at layer {layer_index}: {action}")
            for _ in range(action):
                hidden_states = layer(
                    hidden_states,
                    attention_mask=masks[
                        getattr(layer, "attention_type", qwen.config.layer_types[layer_index])
                    ],
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=cache_position,
                    **kwargs,
                )
        return qwen.norm(hidden_states), torch.stack(all_router_logits, dim=1)

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        router_labels: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> RouterTrainingOutput:
        if use_cache:
            raise ValueError("teacher-forced router training requires use_cache=False")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids or inputs_embeds")
        source = input_ids if input_ids is not None else inputs_embeds
        batch_size, sequence_length = source.shape[:2]
        device = source.device
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long, device=device)
        if router_labels is None or router_labels.shape != (batch_size, NUM_LAYERS):
            actual = None if router_labels is None else tuple(router_labels.shape)
            raise ValueError(f"router_labels must have shape [B, {NUM_LAYERS}]; got {actual}")
        if router_labels.dtype != torch.long:
            raise TypeError("router_labels must have dtype torch.long")

        hidden_batches, router_batches = [], []
        for sample_index in range(batch_size):
            sample_ids = input_ids[sample_index : sample_index + 1] if input_ids is not None else None
            sample_embeds = (
                inputs_embeds[sample_index : sample_index + 1]
                if inputs_embeds is not None
                else None
            )
            sample_positions = (
                position_ids[sample_index : sample_index + 1]
                if position_ids is not None and position_ids.shape[0] == batch_size
                else position_ids
            )
            hidden, router_logits = self._forward_one(
                sample_ids,
                sample_embeds,
                attention_mask[sample_index : sample_index + 1],
                router_labels[sample_index : sample_index + 1],
                sample_positions,
                **kwargs,
            )
            hidden_batches.append(hidden)
            router_batches.append(router_logits)

        last_hidden_state = torch.cat(hidden_batches, dim=0)
        return RouterTrainingOutput(
            router_logits=torch.cat(router_batches, dim=0),
            logits=self.base_model.lm_head(last_hidden_state),
            last_hidden_state=last_hidden_state,
        )
