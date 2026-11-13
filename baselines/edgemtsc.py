from __future__ import annotations

from typing import Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def get_conv1d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
    bias: bool,
):
    return nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


def fuse_bn(conv: nn.Conv1d, bn: nn.BatchNorm1d):
    running_mean = bn.running_mean
    kernel = conv.weight.to(bn.running_mean.device)
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


def convert_dilated_to_nondilated(kernel: torch.Tensor, dilate_rate: int):
    identity_kernel = torch.ones((1, 1, 1), device=kernel.device)
    if kernel.size(1) == 1:
        return F.conv_transpose1d(
            kernel,
            identity_kernel,
            stride=dilate_rate,
        )
    slices = []
    for i in range(kernel.size(1)):
        slices.append(
            F.conv_transpose1d(
                kernel[:, i : i + 1, :],
                identity_kernel,
                stride=dilate_rate,
            )
        )
    return torch.cat(slices, dim=1)


def dilated_into_large_kernel(
    large_kernel: torch.Tensor,
    dilated_kernel: torch.Tensor,
    dilated_r: int,
    device: torch.device,
):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(
        dilated_kernel,
        dilated_r,
    )
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    pad = torch.zeros(
        equivalent_kernel.size(0),
        equivalent_kernel.size(1),
        rows_to_pad,
        device=device,
    )
    return torch.cat((pad, equivalent_kernel, pad), dim=-1)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        super().__init__()
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                stride=1,
                bias=True,
            )
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class SKB(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.conv1 = weight_norm(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                groups=in_channels,
            )
        )
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv2 = weight_norm(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=1,
                padding=0,
                bias=False,
                groups=in_channels,
            )
        )
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.bn3 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.model = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            return self.relu(
                self.bn1(self.conv1(x))
                + self.bn2(self.conv2(x))
                + self.bn3(x)
            )
        return self.relu(self.model(x))

    def fuse_conv(self, conv, bn):
        if conv is not None:
            kernel = conv.weight
            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            eps = bn.eps
        else:
            kernel = np.zeros(
                (self.in_channels, 1, 3),
                dtype=np.float32,
            )
            for i in range(self.in_channels):
                kernel[i, 0, 1] = 1
            kernel = torch.from_numpy(kernel).to(bn.weight.device)
            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    @staticmethod
    def pad_(weight: torch.Tensor):
        pad_zeros = torch.zeros_like(weight)
        return torch.cat((pad_zeros, weight, pad_zeros), dim=-1)

    def merge(self, device: torch.device):
        k3, b3 = self.fuse_conv(self.conv1, self.bn1)
        k1, b1 = self.fuse_conv(self.conv2, self.bn2)
        ki, bi = self.fuse_conv(None, self.bn3)
        k1 = self.pad_(k1)
        k = k3 + k1 + ki
        b = b3 + b1 + bi
        conv = nn.Conv1d(
            self.in_channels,
            self.in_channels,
            kernel_size=3,
            padding=1,
            bias=True,
            groups=self.in_channels,
        ).to(device)
        conv.eval()
        conv.weight.data = k
        conv.bias.data = b
        self.model = conv
        del self.conv1, self.conv2, self.bn1, self.bn2, self.bn3


class IMP_SKB(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        seq_len: int,
    ):
        super().__init__()
        self.up_sample = ConvBlock(in_channels, hidden_channels)
        self.depth = SKB(hidden_channels)
        self.input_adj = weight_norm(
            nn.Linear(hidden_channels, hidden_channels, bias=False)
        )
        self.bn1 = nn.BatchNorm1d(seq_len)
        self.actv1 = nn.ReLU()
        self.actv2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_sample(x)
        x_t = x.transpose(-1, -2)
        x = self.actv1(
            self.bn1(self.input_adj(x_t) + x_t).transpose(-1, -2)
        )
        return self.actv2(self.depth(x))

    def merge(self, device: torch.device):
        self.depth.merge(device)


class LKB(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int):
        super().__init__()
        table = {
            47: ([5, 23, 3, 3, 3], [1, 2, 21, 19, 17]),
            17: ([5, 9, 3, 3, 3], [1, 2, 4, 5, 7]),
            15: ([5, 7, 3, 3, 3], [1, 2, 3, 5, 7]),
            13: ([5, 7, 3, 3, 3], [1, 2, 3, 4, 5]),
            11: ([5, 5, 3, 3, 3], [1, 2, 3, 4, 5]),
            9: ([5, 5, 3, 3], [1, 2, 3, 4]),
            7: ([5, 3, 3], [1, 2, 3]),
            5: ([3, 3], [1, 2]),
        }
        if kernel_size not in table:
            raise ValueError(f"Unsupported widthwise_kernel_size={kernel_size}")

        self.kernel_sizes, self.dilates = table[kernel_size]
        self.model = None
        self.origin_bn = nn.BatchNorm1d(in_channels)
        self.lk_origin = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            dilation=1,
            groups=in_channels,
            bias=False,
        )

        for k, r in zip(self.kernel_sizes, self.dilates):
            setattr(
                self,
                f"dil_conv_k{k}_{r}",
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=k,
                    stride=1,
                    padding=(r * (k - 1) + 1) // 2,
                    dilation=r,
                    groups=in_channels,
                    bias=False,
                ),
            )
            setattr(
                self,
                f"dil_bn_k{k}_{r}",
                nn.BatchNorm1d(in_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model is not None:
            return self.model(x)

        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = getattr(self, f"dil_conv_k{k}_{r}")
            bn = getattr(self, f"dil_bn_k{k}_{r}")
            out = out + bn(conv(x))
        return out + x

    def merge_dilated_branches(self, device: torch.device):
        if self.model is not None:
            return

        origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = getattr(self, f"dil_conv_k{k}_{r}")
            bn = getattr(self, f"dil_bn_k{k}_{r}")
            branch_k, branch_b = fuse_bn(conv, bn)
            equivalent_k = dilated_into_large_kernel(
                origin_k,
                branch_k,
                r,
                device,
            )
            origin_k += equivalent_k
            origin_b += branch_b

        merged_conv = get_conv1d(
            origin_k.size(0),
            origin_k.size(0),
            origin_k.size(2),
            stride=1,
            padding=origin_k.size(2) // 2,
            dilation=1,
            groups=origin_k.size(0),
            bias=True,
        )
        merged_conv.eval()
        identity = torch.zeros_like(origin_k)
        identity[:, :, origin_k.size(2) // 2] = 1
        merged_conv.weight.data = origin_k + identity
        merged_conv.bias.data = origin_b
        self.model = merged_conv.to(device)

        del self.origin_bn, self.lk_origin
        for k, r in zip(self.kernel_sizes, self.dilates):
            delattr(self, f"dil_conv_k{k}_{r}")
            delattr(self, f"dil_bn_k{k}_{r}")


class IMP_LKB(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        widthwise_kernel_size: int,
        seq_len: int,
    ):
        super().__init__()
        self.up_sample = ConvBlock(in_channels, hidden_channels)
        self.width = LKB(hidden_channels, widthwise_kernel_size)
        self.input_adj = weight_norm(
            nn.Linear(hidden_channels, hidden_channels, bias=False)
        )
        self.bn1 = nn.BatchNorm1d(seq_len)
        self.actv1 = nn.ReLU()
        self.actv2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_sample(x)
        x_t = x.transpose(-1, -2)
        x = self.actv1(
            self.bn1(self.input_adj(x_t) + x_t).transpose(-1, -2)
        )
        out = self.width(x)
        return self.actv2(out + x)

    def merge(self, device: torch.device):
        self.width.merge_dilated_branches(device)


class EdgeMTSC(nn.Module):
    def __init__(
        self,
        seq_len: int,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.2,
        widthwise_kernel_size: int = 13,
        hidden_channels: Sequence[int] = (256, 64),
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.dropout = float(dropout)
        self.widthwise_kernel_size = int(widthwise_kernel_size)
        self.hidden_channels = [self.in_channels] + list(hidden_channels)

        self.b0 = IMP_SKB(
            self.in_channels,
            int(hidden_channels[0]),
            self.seq_len,
        )
        self.b1 = IMP_LKB(
            int(hidden_channels[0]),
            int(hidden_channels[1]),
            self.widthwise_kernel_size,
            self.seq_len,
        )

        self.lin_cls = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            weight_norm(
                nn.Linear(
                    int(hidden_channels[-1]),
                    self.num_classes,
                )
            ),
        )

        self.conv_cls = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            ConvBlock(
                int(hidden_channels[-1]),
                self.num_classes,
                kernel_size=1,
            ),
            nn.Dropout(self.dropout),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.b0(x)
        out = self.b1(out)
        return (self.conv_cls(out) + self.lin_cls(out)) / 2.0

    def merge(self, device: torch.device):
        self.b0.merge(device)
        self.b1.merge(device)
