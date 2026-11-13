from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.60, gamma: float = 1.5) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
        pt = torch.clamp(pt, 1e-8, 1.0)
        alpha_t = torch.where(
            target == 1,
            torch.full_like(pt, self.alpha),
            torch.full_like(pt, 1.0 - self.alpha),
        )
        return (alpha_t * (1.0 - pt).pow(self.gamma) * ce).mean()


def false_positive_penalty(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    negative = target == 0
    if not bool(negative.any()):
        return logits.new_zeros(())
    prob_pos = torch.softmax(logits.float(), dim=1)[:, 1]
    return (prob_pos[negative] ** 2).mean()
