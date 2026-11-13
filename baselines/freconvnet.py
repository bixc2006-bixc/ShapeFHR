from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyAdaptiveBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight_all = nn.Parameter(torch.randn(dim, 2) * 0.02)
        self.weight_high = nn.Parameter(torch.randn(dim, 2) * 0.02)
        self.weight_low = nn.Parameter(torch.randn(dim, 2) * 0.02)
        self.threshold = nn.Parameter(torch.rand(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_time = x.transpose(1, 2).float()
        n = x_time.size(1)
        x_fft = torch.fft.rfft(x_time, dim=1, norm="ortho")

        energy = x_fft.abs().pow(2).sum(dim=-1)
        median_energy = energy.reshape(energy.size(0), -1).median(
            dim=1,
            keepdim=True,
        )[0]
        normalized_energy = energy / (median_energy + 1e-6)

        hard_mask = (normalized_energy > self.threshold).float()
        high_mask = (hard_mask - self.threshold).detach() + self.threshold
        low_mask = 1.0 - high_mask

        high_mask = high_mask.unsqueeze(-1)
        low_mask = low_mask.unsqueeze(-1)

        w_all = torch.view_as_complex(self.weight_all.contiguous())
        w_high = torch.view_as_complex(self.weight_high.contiguous())
        w_low = torch.view_as_complex(self.weight_low.contiguous())

        x_fft = (
            x_fft * high_mask * w_high
            + x_fft * low_mask * w_low
        ) * w_all

        out = torch.fft.irfft(x_fft, n=n, dim=1, norm="ortho")
        return out.to(dtype).transpose(1, 2)


class FreConvBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        ffn_ratio: int,
        nvars: int,
        dropout: float,
    ):
        super().__init__()
        d_ff = int(d_model * ffn_ratio)

        self.fab = FrequencyAdaptiveBlock(d_model)
        self.fab_norm = nn.BatchNorm1d(d_model)

        self.ffn_var_1 = nn.Conv1d(
            nvars * d_model,
            nvars * d_ff,
            kernel_size=1,
            groups=nvars,
        )
        self.ffn_var_2 = nn.Conv1d(
            nvars * d_ff,
            nvars * d_model,
            kernel_size=1,
            groups=nvars,
        )

        self.ffn_feat_1 = nn.Conv1d(
            d_model * nvars,
            d_ff * nvars,
            kernel_size=1,
            groups=d_model,
        )
        self.ffn_feat_2 = nn.Conv1d(
            d_ff * nvars,
            d_model * nvars,
            kernel_size=1,
            groups=d_model,
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        b, m, d, n = x.shape

        x = x.reshape(b * m, d, n)
        x = self.fab(x)
        x = self.fab_norm(x)
        x = x.reshape(b, m, d, n)

        x = x.reshape(b, m * d, n)
        x = self.ffn_var_1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.ffn_var_2(x)
        x = self.dropout(x)
        x = x.reshape(b, m, d, n)

        x = x.permute(0, 2, 1, 3).reshape(b, d * m, n)
        x = self.ffn_feat_1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.ffn_feat_2(x)
        x = self.dropout(x)
        x = x.reshape(b, d, m, n).permute(0, 2, 1, 3)

        x = x + residual
        x = x.reshape(b * m, d, n)
        x = self.norm(x)
        return x.reshape(b, m, d, n)


class FreConvStage(nn.Module):
    def __init__(
        self,
        d_model: int,
        ffn_ratio: int,
        nvars: int,
        num_blocks: int,
        dropout: float,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                FreConvBlock(
                    d_model=d_model,
                    ffn_ratio=ffn_ratio,
                    nvars=nvars,
                    dropout=dropout,
                )
                for _ in range(int(num_blocks))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class FreConvNet(nn.Module):
    def __init__(
        self,
        seq_len: int,
        in_channels: int,
        num_classes: int,
        patch_size: int = 16,
        patch_stride: int = 8,
        downsample_ratio: int = 2,
        ffn_ratio: int = 2,
        num_blocks: Sequence[int] = (2, 2),
        dims: Sequence[int] = (64, 128),
        dropout: float = 0.10,
    ):
        super().__init__()

        if len(num_blocks) != len(dims):
            raise ValueError("num_blocks and dims must have the same length")
        if in_channels < 1:
            raise ValueError("in_channels must be >= 1")

        self.seq_len = int(seq_len)
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.downsample_ratio = int(downsample_ratio)
        self.ffn_ratio = int(ffn_ratio)
        self.num_blocks = tuple(int(x) for x in num_blocks)
        self.dims = tuple(int(x) for x in dims)
        self.dropout_p = float(dropout)

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv1d(
                    1,
                    self.dims[0],
                    kernel_size=self.patch_size,
                    stride=self.patch_stride,
                ),
                nn.BatchNorm1d(self.dims[0]),
            )
        )

        for i in range(len(self.dims) - 1):
            self.downsample_layers.append(
                nn.Sequential(
                    nn.BatchNorm1d(self.dims[i]),
                    nn.Conv1d(
                        self.dims[i],
                        self.dims[i + 1],
                        kernel_size=self.downsample_ratio,
                        stride=self.downsample_ratio,
                    ),
                )
            )

        self.stages = nn.ModuleList(
            [
                FreConvStage(
                    d_model=self.dims[i],
                    ffn_ratio=self.ffn_ratio,
                    nvars=self.in_channels,
                    num_blocks=self.num_blocks[i],
                    dropout=self.dropout_p,
                )
                for i in range(len(self.dims))
            ]
        )

        self.head_dropout = nn.Dropout(self.dropout_p)
        self.classifier = nn.Linear(self.dims[-1], self.num_classes)

    @staticmethod
    def _pad_last(x: torch.Tensor, pad_len: int) -> torch.Tensor:
        if pad_len <= 0:
            return x
        return torch.cat([x, x[:, :, -1:].expand(-1, -1, pad_len)], dim=-1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,C,L], got {tuple(x.shape)}")
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {x.size(1)}"
            )

        b, m, _ = x.shape
        x = x.unsqueeze(2)

        for stage_idx, stage in enumerate(self.stages):
            b, m, d, n = x.shape
            x = x.reshape(b * m, d, n)

            layer = self.downsample_layers[stage_idx]
            if stage_idx == 0:
                if n < self.patch_size:
                    x = self._pad_last(x, self.patch_size - n)
                remainder = (x.size(-1) - self.patch_size) % self.patch_stride
                if remainder != 0:
                    x = self._pad_last(x, self.patch_stride - remainder)
            else:
                remainder = x.size(-1) % self.downsample_ratio
                if remainder != 0:
                    x = self._pad_last(x, self.downsample_ratio - remainder)

            x = layer(x)
            _, d_new, n_new = x.shape
            x = x.reshape(b, m, d_new, n_new)
            x = stage(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = F.gelu(x)
        x = x.permute(0, 2, 1, 3)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.head_dropout(x)
        return self.classifier(x)
