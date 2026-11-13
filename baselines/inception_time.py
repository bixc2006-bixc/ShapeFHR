from __future__ import annotations

import torch
import torch.nn as nn


class InceptionModule1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_sizes=(9, 19, 39)):
        super().__init__()
        bottleneck_ch = out_ch
        self.bottleneck = nn.Conv1d(in_ch, bottleneck_ch, kernel_size=1, bias=False)

        self.conv_list = nn.ModuleList(
            [
                nn.Conv1d(bottleneck_ch, out_ch, kernel_size=k, padding=k // 2, bias=False)
                for k in kernel_sizes
            ]
        )

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm1d(out_ch * (len(kernel_sizes) + 1))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bottleneck = self.bottleneck(x)
        conv_outs = [conv(x_bottleneck) for conv in self.conv_list]
        pool_out = self.conv_pool(self.maxpool(x))
        x = torch.cat(conv_outs + [pool_out], dim=1)
        x = self.bn(x)
        return self.act(x)


class InceptionTime(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 32, depth: int = 6, dropout: float = 0.25):
        super().__init__()
        modules = []
        channels = in_ch
        self.residual_layers = nn.ModuleList()
        res_in_ch = in_ch

        for i in range(depth):
            modules.append(InceptionModule1D(channels, out_ch))
            channels = out_ch * 4
            if (i + 1) % 3 == 0:
                self.residual_layers.append(nn.Conv1d(res_in_ch, channels, kernel_size=1, bias=False))
                res_in_ch = channels
            else:
                self.residual_layers.append(nn.Identity())

        self.modules_list = nn.ModuleList(modules)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_input = x
        for i, module in enumerate(self.modules_list):
            if i % 3 == 0:
                residual_input = x
            x = module(x)
            if (i + 1) % 3 == 0:
                res_layer = self.residual_layers[i]
                residual = res_layer(residual_input)
                x = x + residual
        x = self.pool(x).squeeze(-1)
        x = self.drop(x)
        return self.classifier(x)
