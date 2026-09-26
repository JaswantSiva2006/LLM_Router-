"""Class-balanced focal loss for 28-layer Dr.LLM router decisions."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def focal_loss(
    logits: Tensor, targets: Tensor, alpha: Tensor | list[float] | tuple[float, ...], gamma: float = 2.0
) -> Tensor:
    """Return mean ``-alpha_y * (1-p_y)^gamma * log(p_y)``."""
    if logits.ndim != 3 or logits.shape[-1] != 3:
        raise ValueError(f"logits must have shape [B, 28, 3]; got {tuple(logits.shape)}")
    if logits.shape[1] != 28:
        raise ValueError(f"logits must have 28 layer positions; got {logits.shape[1]}")
    if targets.shape != logits.shape[:2]:
        raise ValueError(
            f"targets must have shape {tuple(logits.shape[:2])}; got {tuple(targets.shape)}"
        )
    if targets.dtype not in (torch.int64, torch.long):
        raise TypeError("targets must have dtype torch.long")
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    if targets.numel() and (targets.min().item() < 0 or targets.max().item() > 2):
        raise ValueError("targets must contain only labels 0, 1, or 2")

    weights = torch.as_tensor(alpha, dtype=logits.dtype, device=logits.device)
    if weights.shape != (3,):
        raise ValueError(f"alpha must contain three class weights; got shape {tuple(weights.shape)}")

    log_probabilities = torch.log_softmax(logits, dim=-1)
    log_p_y = log_probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    # -expm1(log(p)) computes 1-p accurately when p is close to one.
    one_minus_p_y = -torch.expm1(log_p_y)
    alpha_y = weights[targets]
    return (-alpha_y * one_minus_p_y.pow(gamma) * log_p_y).mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: Tensor | list[float] | tuple[float, ...], gamma: float = 2.0) -> None:
        super().__init__()
        weights = torch.as_tensor(alpha, dtype=torch.float32)
        if weights.shape != (3,):
            raise ValueError("alpha must contain three class weights")
        self.register_buffer("alpha", weights)
        self.gamma = gamma

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return focal_loss(logits, targets, self.alpha, self.gamma)
