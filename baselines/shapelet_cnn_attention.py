from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShapeletCNNAttentionClassifier(nn.Module):
    def __init__(self, num_shapelets: int, dropout: float = 0.25, concat_stat: bool = False, stat_dim: int = 14):
        super().__init__()
        self.num_shapelets = num_shapelets
        self.concat_stat = concat_stat
        self.stat_dim = stat_dim

        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(dropout)

        self.attn_fc1 = nn.Linear(128, 64)
        self.attn_fc2 = nn.Linear(64, 1)

        feat_dim = 128 * 3
        if concat_stat:
            feat_dim += stat_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor, stat: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.dropout(x)

        # [B, 128, K] -> [B, K, 128]
        x = x.transpose(1, 2)

        attn = self.attn_fc1(x)
        attn = torch.tanh(attn)
        attn = self.attn_fc2(attn).squeeze(-1)
        attn = torch.softmax(attn, dim=1).unsqueeze(-1)

        z_attn = torch.sum(x * attn, dim=1)
        z_mean = torch.mean(x, dim=1)
        z_max = torch.max(x, dim=1).values

        feat = torch.cat([z_attn, z_mean, z_max], dim=1)
        if self.concat_stat and stat is not None:
            feat = torch.cat([feat, stat], dim=1)

        logits = self.classifier(feat)
        return logits
