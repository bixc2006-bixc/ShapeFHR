from __future__ import annotations

import torch
import torch.nn as nn


class MLPStatClassifier(nn.Module):
    def __init__(self, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(14, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
