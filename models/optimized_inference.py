"""Fixed-weight ShapeFHR acceleration used exclusively by the complexity experiment."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class OptimizedShapeFHR(nn.Module):
    """Collapse the global latent dot product into one scalar convolution per template."""

    def __init__(self, model):
        super().__init__()
        if model.enabled_branches != ("shapelet", "generic", "stat") or model.match_mode != "global_latent":
            raise ValueError("Optimized inference requires the full global-latent ShapeFHR model")
        self.model = model.eval()
        branch = model.specific
        conv = branch.shared_conv
        with torch.no_grad():
            vectors = torch.stack([branch._shapelet_vector(p) for p in branch.shapelet_params])
            filters = torch.einsum("kc,cil->kil", vectors, conv.weight)
            # Bound accumulated FP32 rounding at both the convolution and dot-product stages.
            weight_scale = vectors.abs() @ conv.weight.abs().flatten(1).sum(dim=1)
            bias_scale = vectors.abs() @ conv.bias.abs()
            eps = torch.finfo(vectors.dtype).eps
            tolerance_scale = 4 * (vectors.shape[1] + 2 * conv.kernel_size[0] + 4) * eps
        self.register_buffer("vectors", vectors, persistent=False)
        self.register_buffer("filters", filters, persistent=False)
        self.register_buffer("weight_tolerance", tolerance_scale * weight_scale, persistent=False)
        self.register_buffer("bias_tolerance", tolerance_scale * bias_scale, persistent=False)

    def selected_starts(self, x):
        # The convolution bias adds the same offset at every time point.
        similarities = F.conv1d(x, self.filters)
        if similarities.shape[-1] == 1:
            return similarities.argmax(dim=-1), False
        top = similarities.topk(2, dim=-1)
        gap = top.values[..., 0] - top.values[..., 1]
        max_input = x.abs().amax(dim=(1, 2))[:, None]
        tolerance = max_input * self.weight_tolerance + self.bias_tolerance
        return top.indices[..., 0], bool((gap <= tolerance).any())

    def forward(self, x):
        branch = self.model.specific
        starts, fallback = self.selected_starts(x)
        if fallback:
            specific = branch(x)
        else:
            kernel = branch.shared_conv.kernel_size[0]
            positions = starts[..., None] + torch.arange(kernel, device=x.device)
            windows = x[:, 0].gather(1, positions.flatten(1))
            n, k = starts.shape
            features = branch.shared_conv(windows.reshape(n * k, 1, kernel))
            features = features.reshape(n, k, -1)
            tokens = branch.token_proj(features - self.vectors[None])
            specific = branch.pool(branch.encoder(tokens + branch.token_embed))
        return self.model.classifier(torch.cat((specific, self.model.generic(x), self.model.stat(x)), dim=1))


@torch.no_grad()
def verify_optimization(model, optimized, length):
    generator = torch.Generator().manual_seed(8701)
    noise = torch.randn((8, 1, length), generator=generator)
    test = torch.cat((noise, torch.zeros(1, 1, length), torch.ones(1, 1, length),
                      torch.sin(torch.arange(length).reshape(1, 1, -1) / 91)), dim=0)
    errors = []
    fast_count = 0
    for i, row in enumerate(test):
        sample = row.unsqueeze(0)
        reference = model(sample)
        actual = optimized(sample)
        torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-5)
        errors.append(float((actual - reference).abs().max()))
        starts, fallback = optimized.selected_starts(sample)
        if not fallback:
            fast_count += 1
            expected_starts = model.shapelet_match_info(sample)["best_indices"]
            if not torch.equal(starts, expected_starts):
                raise RuntimeError(f"Optimized ShapeFHR changed the match indices on test case {i}")
    if fast_count == 0:
        raise RuntimeError("ShapeFHR equivalence check never exercised the fast path")
    return max(errors), len(test), fast_count
