from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

from types import SimpleNamespace
from data.shapelets import ShapeletEntry


class ConvBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class OriginalShapeFormer(nn.Module):
    """Original ShapeFormer baseline adapted to single-channel FHR."""

    def __init__(self, cfg: SimpleNamespace, shapelets: Sequence[ShapeletEntry]) -> None:
        super().__init__()
        if cfg.match_distance not in {"cid", "euclidean"}:
            raise ValueError("match_distance must be 'cid' or 'euclidean'")
        if not shapelets:
            raise ValueError("OriginalShapeFormer requires at least one shapelet")

        self.seq_len = cfg.seq_len
        self.match_distance = cfg.match_distance
        self.match_window = min(max(int(cfg.shapelet_match_window), 1), cfg.seq_len)

        self.shapelets = nn.ParameterList()
        self.input_projectors = nn.ModuleList()
        self.shapelet_projectors = nn.ModuleList()
        start_indices: list[int] = []
        end_indices: list[int] = []
        search_starts: list[int] = []
        search_ends: list[int] = []
        variables: list[int] = []

        for item in shapelets:
            values = torch.as_tensor(item.values, dtype=torch.float32).reshape(-1)
            if values.numel() > cfg.seq_len:
                values = values[: cfg.seq_len]
            length = int(values.numel())
            self.shapelets.append(nn.Parameter(values.clone()))
            self.input_projectors.append(nn.Linear(length, cfg.d_spe))
            self.shapelet_projectors.append(nn.Linear(length, cfg.d_spe))

            start = max(0, min(item.start_idx or 0, cfg.seq_len - 1))
            end = item.end_idx - 1 if item.end_idx is not None and item.end_idx > start else start + length - 1
            end = max(0, min(end, cfg.seq_len - 1))
            start_indices.append(start)
            end_indices.append(end)
            variables.append(max(0, min(int(item.variable), cfg.in_channels - 1)))

            has_position = not (start == 0 and end <= length)
            if has_position:
                search_start = max(0, start - self.match_window)
                search_end = min(cfg.seq_len, end + self.match_window + 1)
                if search_end - search_start < length:
                    search_start, search_end = 0, cfg.seq_len
            else:
                search_start, search_end = 0, cfg.seq_len
            search_starts.append(search_start)
            search_ends.append(search_end)

        self.register_buffer("shapelet_start_idx", torch.tensor(start_indices, dtype=torch.long), persistent=False)
        self.register_buffer("shapelet_end_idx", torch.tensor(end_indices, dtype=torch.long), persistent=False)
        self.register_buffer("shapelet_search_start", torch.tensor(search_starts, dtype=torch.long), persistent=False)
        self.register_buffer("shapelet_search_end", torch.tensor(search_ends, dtype=torch.long), persistent=False)
        self.register_buffer("shapelet_variable", torch.tensor(variables, dtype=torch.long), persistent=False)

        self.start_embedding = nn.Embedding(cfg.seq_len, cfg.d_spe)
        self.end_embedding = nn.Embedding(cfg.seq_len, cfg.d_spe)
        self.variable_embedding = nn.Embedding(cfg.in_channels, cfg.d_spe)
        specific_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_spe,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.specific_encoder = nn.TransformerEncoder(specific_layer, num_layers=cfg.num_layers)

        self.generic_conv = nn.Sequential(
            ConvBlock1d(cfg.in_channels, cfg.d_gen, cfg.generic_kernel_size, stride=cfg.generic_stride),
            ConvBlock1d(cfg.d_gen, cfg.d_gen, cfg.generic_kernel_size, stride=1),
        )
        max_generic_tokens = math.ceil(cfg.seq_len / max(cfg.generic_stride, 1)) + 4
        self.generic_pos_embedding = nn.Parameter(torch.zeros(1, max_generic_tokens, cfg.d_gen))
        generic_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_gen,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.generic_encoder = nn.TransformerEncoder(generic_layer, num_layers=cfg.num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_spe + cfg.d_gen),
            nn.Linear(cfg.d_spe + cfg.d_gen, cfg.d_spe),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_spe, cfg.num_classes),
        )

    def shapelet_parameters(self) -> list[nn.Parameter]:
        return list(self.shapelets.parameters())

    @staticmethod
    def _cid_distance(subseq: torch.Tensor, shapelet: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        euclidean = torch.sqrt(torch.sum((subseq - shapelet.view(1, 1, -1)) ** 2, dim=-1) + eps)
        if shapelet.numel() <= 1:
            return euclidean
        subseq_complexity = torch.sqrt(torch.sum(torch.diff(subseq, dim=-1) ** 2, dim=-1) + eps)
        shapelet_complexity = torch.sqrt(torch.sum(torch.diff(shapelet) ** 2) + eps)
        return euclidean * (
            torch.maximum(subseq_complexity, shapelet_complexity)
            / torch.clamp(torch.minimum(subseq_complexity, shapelet_complexity), min=eps)
        )

    def _best_fit_subsequence(self, x: torch.Tensor, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        shapelet = self.shapelets[idx]
        length = int(shapelet.numel())
        variable = int(self.shapelet_variable[idx].item())
        search_start = int(self.shapelet_search_start[idx].item())
        search_end = int(self.shapelet_search_end[idx].item())
        series = x[:, variable, search_start:search_end]  # [B, search_len]
        if series.shape[-1] < length:
            series = x[:, variable, :]
            search_start = 0

        subseq = series.unfold(dimension=-1, size=length, step=1)  # [B, num_windows, length]
        # The discrete selection does not contribute a gradient.
        with torch.no_grad():
            best_dist: torch.Tensor | None = None
            best_start: torch.Tensor | None = None
            for chunk_start in range(0, subseq.shape[1], self.match_window):
                chunk_end = min(chunk_start + self.match_window, subseq.shape[1])
                chunk = subseq[:, chunk_start:chunk_end, :].contiguous()
                if self.match_distance == "cid":
                    dist = self._cid_distance(chunk, shapelet)
                else:
                    dist = torch.sum((chunk - shapelet.view(1, 1, -1)) ** 2, dim=-1)
                chunk_dist, local_start = torch.min(dist, dim=-1)
                local_start = local_start + chunk_start
                if best_dist is None:
                    best_dist = chunk_dist
                    best_start = local_start
                else:
                    improved = chunk_dist < best_dist
                    best_dist = torch.where(improved, chunk_dist, best_dist)
                    best_start = torch.where(improved, local_start, best_start)

        if best_start is None:
            raise RuntimeError("No valid subsequence windows for shapelet matching")
        batch_idx = torch.arange(series.shape[0], device=series.device)
        best = subseq[batch_idx, best_start, :]  # [B, length]
        return best, best_start + search_start

    def _specific_tokens(self, x: torch.Tensor) -> torch.Tensor:
        tokens: list[torch.Tensor] = []
        for idx, shapelet in enumerate(self.shapelets):
            best, best_start = self._best_fit_subsequence(x, idx)
            input_emb = self.input_projectors[idx](best)
            shapelet_emb = self.shapelet_projectors[idx](shapelet.view(1, -1))
            token = input_emb - shapelet_emb  # U_i = P_I(I_i) - P_S(S_i), [B, d_spe]

            static_start = int(self.shapelet_start_idx[idx].item())
            static_end = int(self.shapelet_end_idx[idx].item())
            if static_start == 0 and static_end <= int(shapelet.numel()):
                start_pos = torch.clamp(best_start, 0, self.seq_len - 1)
                end_pos = torch.clamp(best_start + int(shapelet.numel()) - 1, 0, self.seq_len - 1)
                token = token + self.start_embedding(start_pos) + self.end_embedding(end_pos)
            else:
                token = token + self.start_embedding(self.shapelet_start_idx[idx]).view(1, -1)
                token = token + self.end_embedding(self.shapelet_end_idx[idx]).view(1, -1)
            token = token + self.variable_embedding(self.shapelet_variable[idx]).view(1, -1)
            tokens.append(token)
        return torch.stack(tokens, dim=1)  # [B, num_shapelets, d_spe]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 4800]
        specific_tokens = self._specific_tokens(x)  # [B, K, d_spe]
        specific_out = self.specific_encoder(specific_tokens)
        specific_repr = specific_out[:, 0, :]  # [B, d_spe]

        generic_tokens = self.generic_conv(x).transpose(1, 2)  # [B, T, d_gen]
        generic_tokens = generic_tokens + self.generic_pos_embedding[:, : generic_tokens.shape[1], :]
        generic_out = self.generic_encoder(generic_tokens)
        generic_repr = generic_out.mean(dim=1)  # [B, d_gen]

        return self.head(torch.cat([specific_repr, generic_repr], dim=-1))  # [B, 2]
