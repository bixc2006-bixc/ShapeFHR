from __future__ import annotations

import torch
import torch.nn as nn

from config import GLOBAL_SIGNAL_STRIDE, LOCAL_LATENT_RADIUS
from data.shapelets import ShapeletEntry
from models.common import AttnMeanMaxPool1D, GenericTransformerBranch
from models.signal_match import best_signal_window, candidate_starts


class ShapeletSpecificBranch(nn.Module):
    """Global vector-to-vector shapelet matching branch.

    For every shapelet:
      shapelet -> shared Conv1D -> global average pooling -> e_j [C]
      FHR      -> shared Conv1D -> h_1...h_T [C]
      q_t = <e_j, h_t>; t* = argmax(q_t)
      delta_j = h_t* - e_j
      token_j = MLP(delta_j)

    The resulting shapelet tokens are contextualized by a Transformer and
    summarized with attention/mean/max pooling.
    """

    def __init__(
        self,
        *,
        shapelets: list[ShapeletEntry],
        d_spe: int,
        shared_kernel_size: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        match_mode: str = 'global_latent',
        signal_global_stride: int | None = None,
    ) -> None:
        super().__init__()
        if not shapelets:
            raise ValueError("shapelet list is empty")

        if match_mode not in {"global_latent", "local_latent", "global_signal"}:
            raise ValueError(f"Unknown matching mode: {match_mode}")
        self.match_mode = match_mode
        self.shapelet_starts = [entry.start_idx for entry in shapelets]
        self.search_radius = LOCAL_LATENT_RADIUS
        self.signal_global_stride = GLOBAL_SIGNAL_STRIDE if signal_global_stride is None else signal_global_stride
        if self.signal_global_stride < 1:
            raise ValueError("Global signal matching stride must be positive")
        self.d_spe = int(d_spe)
        self.kernel_size = int(shared_kernel_size)
        self.shapelet_classes = [int(entry.cls) for entry in shapelets]
        self.shapelet_lengths = [int(len(entry.values)) for entry in shapelets]

        self.shapelet_params = nn.ParameterList()
        for idx, entry in enumerate(shapelets):
            values = torch.as_tensor(entry.values, dtype=torch.float32).reshape(-1)
            if values.numel() < self.kernel_size:
                raise ValueError(
                    f"shapelet[{idx}] length={values.numel()} is shorter than shared kernel={self.kernel_size}"
                )
            self.shapelet_params.append(nn.Parameter(values.clone()))
            self.register_buffer(f"shapelet_init_{idx}", values.clone(), persistent=False)

        self.shared_conv = nn.Conv1d(
            in_channels=1,
            out_channels=d_spe,
            kernel_size=self.kernel_size,
            padding=0,
            bias=True,
        )

        self.token_proj = nn.Sequential(
            nn.Linear(d_spe, d_spe),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        num_shapelets = len(self.shapelet_params)
        self.token_embed = nn.Parameter(torch.empty(1, num_shapelets, d_spe))
        nn.init.trunc_normal_(self.token_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_spe,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool = AttnMeanMaxPool1D(d_spe, dropout=dropout)

    def shapelet_deviation_loss(self) -> torch.Tensor:
        losses = []
        for i, param in enumerate(self.shapelet_params):
            init_value = getattr(self, f"shapelet_init_{i}")
            losses.append(torch.mean((param - init_value) ** 2))
        return torch.stack(losses).mean()

    def _shapelet_vector(self, shapelet: torch.Tensor) -> torch.Tensor:
        feature_map = self.shared_conv(shapelet.view(1, 1, -1))
        return feature_map.mean(dim=-1).squeeze(0)

    @staticmethod
    def _best_feature(
        fhr_features: torch.Tensor,
        shapelet_vector: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # fhr_features: [B, C, T], shapelet_vector: [C]
        similarity = torch.einsum("bct,c->bt", fhr_features, shapelet_vector)
        best_index = torch.argmax(similarity, dim=1)
        fhr_tokens = fhr_features.transpose(1, 2)  # [B, T, C]
        batch_index = torch.arange(fhr_tokens.size(0), device=fhr_tokens.device)
        best_feature = fhr_tokens[batch_index, best_index]
        best_score = similarity[batch_index, best_index]
        return best_feature, best_index, best_score

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_match_info: bool = False,
    ):
        fhr_features = self.shared_conv(x) if self.match_mode.endswith("latent") else None
        tokens: list[torch.Tensor] = []
        all_best_indices: list[torch.Tensor] = []
        all_best_scores: list[torch.Tensor] = []

        for shapelet, anchor in zip(self.shapelet_params, self.shapelet_starts):
            shapelet_vector = self._shapelet_vector(shapelet)
            length = shapelet.numel()
            if self.match_mode == "global_latent":
                best_feature, best_index, best_score = self._best_feature(fhr_features, shapelet_vector)
            elif self.match_mode == "local_latent":
                if anchor is None or not 0 <= anchor <= x.size(-1) - length:
                    raise ValueError("Local matching requires valid shapelet start_idx metadata")
                low = max(0, anchor - self.search_radius)
                high = min(fhr_features.size(-1), anchor + length + self.search_radius - self.kernel_size + 1)
                best_feature, best_index, best_score = self._best_feature(
                    fhr_features[..., low:high], shapelet_vector
                )
                best_index = best_index + low
            else:
                starts = candidate_starts(x.size(-1), length, self.signal_global_stride, x.device)
                best_index, distance = best_signal_window(x[:, 0], shapelet, starts)
                indices = best_index[:, None] + torch.arange(length, device=x.device)
                selected = x.gather(2, indices[:, None, :])
                best_feature = self.shared_conv(selected).mean(dim=-1)
                best_score = -distance
            difference = best_feature - shapelet_vector.unsqueeze(0)
            tokens.append(self.token_proj(difference))
            if return_match_info:
                all_best_indices.append(best_index)
                all_best_scores.append(best_score)

        u = torch.stack(tokens, dim=1)
        u = self.encoder(u + self.token_embed)
        representation = self.pool(u)

        if not return_match_info:
            return representation

        info = {
            "best_indices": torch.stack(all_best_indices, dim=1),
            "best_scores": torch.stack(all_best_scores, dim=1),
            "shapelet_classes": torch.as_tensor(
                self.shapelet_classes,
                dtype=torch.long,
                device=x.device,
            ),
            "shapelet_lengths": torch.as_tensor(
                self.shapelet_lengths,
                dtype=torch.long,
                device=x.device,
            ),
        }
        return representation, info


class StatisticalBranch(nn.Module):
    def __init__(self, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(14),
            nn.Linear(14, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(statistical_features(x))


def statistical_features(x: torch.Tensor) -> torch.Tensor:
    s = x.squeeze(1)
    dx = s[:, 1:] - s[:, :-1]
    k = max(1, s.size(1) // 20)

    low5 = torch.topk(s, k=k, dim=1, largest=False).values.mean(dim=1)
    high5 = torch.topk(s, k=k, dim=1, largest=True).values.mean(dim=1)

    feats = torch.stack(
        [
            s.mean(dim=1),
            s.std(dim=1, unbiased=False),
            s.min(dim=1).values,
            s.max(dim=1).values,
            low5,
            high5,
            s.max(dim=1).values - s.min(dim=1).values,
            dx.abs().mean(dim=1),
            dx.std(dim=1, unbiased=False),
            dx.abs().max(dim=1).values,
            (s < -1.0).float().mean(dim=1),
            (s < -1.5).float().mean(dim=1),
            (s > 1.0).float().mean(dim=1),
            (s > 1.5).float().mean(dim=1),
        ],
        dim=1,
    )
    return feats


__all__ = [
"ShapeletSpecificBranch",
"GenericTransformerBranch",
"StatisticalBranch",
]
