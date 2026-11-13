from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        weights = torch.softmax(self.attn(x).squeeze(-1), dim=1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class TransformerOnly(nn.Module):
    def __init__(
        self,
        seq_len: int = 4800,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(1, d_model, kernel_size=8, stride=8, padding=0)
        num_patches = seq_len // 8
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attn_pool = AttentionPooling(d_model)
        self.norm = nn.LayerNorm(d_model * 3)
        self.fc1 = nn.Linear(d_model * 3, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x.transpose(1, 2)  # [B, T, C]
        if x.size(1) <= self.pos_embed.size(1):
            x = x + self.pos_embed[:, : x.size(1), :]
        else:
            pos = nn.functional.interpolate(
                self.pos_embed.transpose(1, 2),
                size=x.size(1),
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            x = x + pos

        x = self.encoder(x)

        attn_pool = self.attn_pool(x)
        mean_pool = x.mean(dim=1)
        max_pool, _ = x.max(dim=1)
        pooled = torch.cat([attn_pool, mean_pool, max_pool], dim=1)

        pooled = self.norm(pooled)
        pooled = self.fc1(pooled)
        pooled = self.act(pooled)
        pooled = self.drop(pooled)
        return self.fc2(pooled)
