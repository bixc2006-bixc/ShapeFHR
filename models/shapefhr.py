from dataclasses import asdict
from typing import Iterable

import torch
import torch.nn as nn

from config import ModelConfig
from data.shapelets import ShapeletEntry
from models.branches import GenericTransformerBranch, ShapeletSpecificBranch, StatisticalBranch


VALID_BRANCHES = ("shapelet", "generic", "stat")


class ShapeFHR(nn.Module):
    """Three-branch ShapeFHR with direct concatenation fusion.

    Every enabled branch is instantiated and used.  Disabled branches are not
    constructed.  The active branch representations are concatenated directly
    and passed to the linear classifier; no branch gate or branch-wise scalar
    weighting is used.
    """

    def __init__(
        self,
        *,
        shapelets: list[ShapeletEntry],
        model_cfg: ModelConfig,
        dropout: float,
        enabled_branches: Iterable[str] = VALID_BRANCHES,
        match_mode: str = "global_latent",
        signal_global_stride: int | None = None,
    ) -> None:
        super().__init__()
        enabled = tuple(enabled_branches)
        if not enabled:
            raise ValueError("at least one branch must be enabled")
        if len(set(enabled)) != len(enabled):
            raise ValueError(f"duplicate branch in {enabled}")
        if any(name not in VALID_BRANCHES for name in enabled):
            raise ValueError(f"invalid enabled_branches={enabled}")

        self.enabled_branches = enabled
        self.match_mode = match_mode
        self.model_cfg = model_cfg
        self.dropout = float(dropout)

        dims = {
            "shapelet": model_cfg.d_spe,
            "generic": model_cfg.d_gen,
            "stat": model_cfg.d_stat,
        }

        if "shapelet" in enabled:
            self.specific = ShapeletSpecificBranch(
                shapelets=shapelets,
                d_spe=model_cfg.d_spe,
                shared_kernel_size=model_cfg.shared_kernel_size,
                nhead=model_cfg.nhead,
                num_layers=model_cfg.num_layers,
                dim_feedforward=model_cfg.dim_feedforward,
                dropout=dropout,
                match_mode=match_mode,
                signal_global_stride=signal_global_stride,
            )
        else:
            self.specific = None

        if "generic" in enabled:
            self.generic = GenericTransformerBranch(
                seq_len=model_cfg.seq_len,
                d_gen=model_cfg.d_gen,
                nhead=model_cfg.nhead,
                num_layers=model_cfg.num_layers,
                dim_feedforward=model_cfg.dim_feedforward,
                dropout=dropout,
                kernel_sizes=model_cfg.generic_kernels,
                downsample_stride=model_cfg.generic_downsample_stride,
            )
        else:
            self.generic = None

        if "stat" in enabled:
            self.stat = StatisticalBranch(model_cfg.d_stat, dropout)
        else:
            self.stat = None

        head_in = sum(dims[name] for name in enabled)
        self.classifier = nn.Linear(head_in, model_cfg.num_classes)

    @property
    def has_shapelet_branch(self) -> bool:
        return self.specific is not None

    def shapelet_deviation_loss(self) -> torch.Tensor:
        if self.specific is None:
            return next(self.parameters()).new_zeros(())
        return self.specific.shapelet_deviation_loss()

    def set_shapelets_trainable(self, trainable: bool) -> None:
        if self.specific is None:
            return
        for param in self.specific.shapelet_params:
            param.requires_grad = bool(trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reps: list[torch.Tensor] = []
        for name in self.enabled_branches:
            if name == "shapelet":
                reps.append(self.specific(x))
            elif name == "generic":
                reps.append(self.generic(x))
            elif name == "stat":
                reps.append(self.stat(x))

        fused = torch.cat(reps, dim=-1)
        return self.classifier(fused)

    @torch.no_grad()
    def shapelet_match_info(self, x: torch.Tensor) -> dict:
        if self.specific is None:
            raise RuntimeError("shapelet branch is disabled")
        _, info = self.specific(x, return_match_info=True)
        return info

    def export_model_settings(self) -> dict:
        settings = {
            "model_cfg": asdict(self.model_cfg),
            "dropout": self.dropout,
            "enabled_branches": list(self.enabled_branches),
            "fusion": "direct_concatenation",
            "match_mode": self.match_mode,
        }
        if self.match_mode == "global_signal" and self.specific is not None:
            settings["signal_global_stride"] = self.specific.signal_global_stride
        return settings
