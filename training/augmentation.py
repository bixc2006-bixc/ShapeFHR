from __future__ import annotations

import torch


def augment_batch(
    x: torch.Tensor,
    *,
    noise_std: float,
    mask_ratio: float,
    gain_std: float,
    baseline_std: float,
    max_abs_z: float,
) -> torch.Tensor:
    out = x
    batch_size, _, seq_len = out.shape

    if gain_std > 0:
        gain = 1.0 + torch.randn(
            (batch_size, 1, 1), device=out.device, dtype=out.dtype
        ) * float(gain_std)
        out = out * gain

    if baseline_std > 0:
        shift = torch.randn(
            (batch_size, 1, 1), device=out.device, dtype=out.dtype
        ) * float(baseline_std)
        out = out + shift

    if noise_std > 0:
        local_scale = out.detach().std(dim=2, keepdim=True).clamp(min=0.2, max=2.0)
        out = out + torch.randn_like(out) * float(noise_std) * local_scale

    if mask_ratio > 0:
        mask_len = max(1, int(seq_len * float(mask_ratio)))
        if mask_len < seq_len - 1:
            out = out.clone()
            starts = torch.randint(
                1,
                seq_len - mask_len,
                (batch_size,),
                device=out.device,
            )
            alpha = torch.linspace(
                0.0,
                1.0,
                steps=mask_len + 2,
                device=out.device,
                dtype=out.dtype,
            )[1:-1].view(1, 1, -1)
            for i in range(batch_size):
                start = int(starts[i].item())
                end = start + mask_len
                left = out[i : i + 1, :, start - 1 : start]
                right = out[i : i + 1, :, end : end + 1]
                out[i : i + 1, :, start:end] = left + (right - left) * alpha

    if max_abs_z > 0:
        out = out.clamp(-float(max_abs_z), float(max_abs_z))
    return out
