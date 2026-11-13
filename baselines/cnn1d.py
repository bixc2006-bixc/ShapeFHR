from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, padding: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class CNN1D(nn.Module):
    def __init__(self, dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock1D(1, 32, kernel_size=7, padding=3),
            nn.MaxPool1d(kernel_size=4),
            ConvBlock1D(32, 64, kernel_size=5, padding=2),
            nn.MaxPool1d(kernel_size=4),
            ConvBlock1D(64, 128, kernel_size=5, padding=2),
            nn.MaxPool1d(kernel_size=4),
            ConvBlock1D(128, 128, kernel_size=3, padding=1),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.classifier(x)
