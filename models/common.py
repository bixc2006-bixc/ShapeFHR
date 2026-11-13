from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class AttnMeanMaxPool1D(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = max(1, d_model // 2)
        self.score = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.softmax(self.score(x).squeeze(-1), dim=1).unsqueeze(-1)
        z_attn = (x * weight).sum(dim=1)
        z_mean = x.mean(dim=1)
        z_max = x.max(dim=1).values
        return self.proj(torch.cat([z_attn, z_mean, z_max], dim=-1))


class MultiScaleConvStem1D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_sizes: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        branch_ch = out_ch // 3
        channels = [branch_ch, branch_ch, out_ch - 2 * branch_ch]
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_ch,
                        branch_out,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(branch_out),
                    nn.GELU(),
                )
                for kernel_size, branch_out in zip(kernel_sizes, channels)
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))


class GenericTransformerBranch(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        d_gen: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        kernel_sizes: Sequence[int],
        downsample_stride: int,
    ) -> None:
        super().__init__()
        self.stem = MultiScaleConvStem1D(
            in_ch=1,
            out_ch=d_gen,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
        )
        self.pool_down = nn.AvgPool1d(
            kernel_size=downsample_stride,
            stride=downsample_stride,
        )
        num_tokens = seq_len // downsample_stride
        self.pos_embed = nn.Parameter(torch.empty(1, num_tokens, d_gen))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_gen,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool = AttnMeanMaxPool1D(d_gen, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        z = self.pool_down(z).transpose(1, 2)
        if z.size(1) != self.pos_embed.size(1):
            raise RuntimeError(
                f"generic token length mismatch: got {z.size(1)}, expected {self.pos_embed.size(1)}"
            )
        z = self.encoder(z + self.pos_embed)
        return self.pool(z)
