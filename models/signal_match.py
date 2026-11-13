from __future__ import annotations

import torch


@torch.no_grad()
def best_signal_window(x, shapelet, starts, chunk_size=64):
    """CID matching in FP32, with deterministic earliest-start tie breaking."""
    x = x.float()
    shapelet = shapelet.float()
    length = shapelet.numel()
    offsets = torch.arange(length, device=x.device)
    shape_complexity = shapelet.diff().square().sum().add(1e-6).sqrt()
    best_distance = x.new_full((x.size(0),), float("inf"))
    best_start = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    for group in starts.split(chunk_size):
        windows = x[:, group[:, None] + offsets]
        euclidean = (windows - shapelet).square().sum(dim=-1).add(1e-6).sqrt()
        complexity = windows.diff(dim=-1).square().sum(dim=-1).add(1e-6).sqrt()
        distances = euclidean * (torch.maximum(complexity, shape_complexity) /
                                 torch.minimum(complexity, shape_complexity).clamp_min(1e-6))
        distance, offset = distances.min(dim=1)
        better = distance < best_distance
        best_start = torch.where(better, group[offset], best_start)
        best_distance = torch.where(better, distance, best_distance)
    return best_start, best_distance


def candidate_starts(seq_len, length, stride, device):
    last = seq_len - length
    if last < 0:
        raise ValueError(f"Template length {length} exceeds input length {seq_len}")
    return torch.cat((torch.arange(0, last, stride, device=device),
                      torch.tensor([last], device=device)))
