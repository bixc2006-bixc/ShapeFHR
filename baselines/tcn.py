from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.25):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)

        self.downsample = None
        if in_ch != out_ch:
            self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.gelu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.gelu(out)
        out = self.dropout(out)

        residual = x if self.downsample is None else self.downsample(x)
        out = out + residual
        out = F.gelu(out)
        return out


class TCNClassifier(nn.Module):
    def __init__(self, dropout: float = 0.25):
        super().__init__()
        self.stem = nn.Conv1d(1, 64, kernel_size=3, padding=1)

        self.block1 = TemporalBlock(64, 64, dilation=1, dropout=dropout)
        self.block2 = TemporalBlock(64, 64, dilation=2, dropout=dropout)
        self.block3 = TemporalBlock(64, 128, dilation=4, dropout=dropout)
        self.block4 = TemporalBlock(128, 128, dilation=8, dropout=dropout)
        self.block5 = TemporalBlock(128, 128, dilation=16, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        avg_pool = torch.mean(x, dim=2)
        max_pool = torch.max(x, dim=2).values
        feat = torch.cat([avg_pool, max_pool], dim=1)
        logits = self.classifier(feat)
        return logits
