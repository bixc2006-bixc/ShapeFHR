from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from config import TrainConfig


@dataclass
class LoadedDataset:
    x: np.ndarray
    y: np.ndarray
    file_names: list[str]


@dataclass
class SplitData:
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    mean: float
    std: float
    file_names: list[str]


def fix_length_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = int(x.size)

    if n == target_len:
        out = x.copy()
    elif n > target_len:
        out = x[:target_len].copy()
    elif n > 0:
        out = np.empty(target_len, dtype=np.float32)
        out[:n] = x
        finite = x[np.isfinite(x)]
        out[n:] = float(finite[-1]) if finite.size else 0.0
    else:
        out = np.zeros(target_len, dtype=np.float32)

    if not np.all(np.isfinite(out)):
        finite = out[np.isfinite(out)]
        fill = float(finite[-1]) if finite.size else 0.0
        out = np.nan_to_num(out, nan=fill, posinf=fill, neginf=fill).astype(np.float32)

    return out


def _read_label(value: np.ndarray) -> int:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return int(arr.item())
    return int(arr.reshape(-1)[0])


def load_files(
    file_names: Sequence[str],
    *,
    signal_key: str,
    label_key: str,
    seq_len: int,
) -> LoadedDataset:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    names: list[str] = []
    for name in file_names:
        path = Path(str(name))
        with np.load(path, allow_pickle=True) as item:
            signal = fix_length_1d(item[signal_key], seq_len)
            label = _read_label(item[label_key])
        xs.append(signal)
        ys.append(label)
        names.append(str(path))
    return LoadedDataset(
        x=np.stack(xs).astype(np.float32, copy=False),
        y=np.asarray(ys, dtype=np.int64),
        file_names=names,
    )


def save_split(split: SplitData, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = np.asarray(split.file_names, dtype=str)
    np.savez_compressed(
        path,
        train_indices=split.train_indices,
        val_indices=split.val_indices,
        test_indices=split.test_indices,
        train_file_names=names[split.train_indices],
        val_file_names=names[split.val_indices],
        test_file_names=names[split.test_indices],
        y_train=split.y_train,
        y_val=split.y_val,
        y_test=split.y_test,
        mean=np.asarray(split.mean),
        std=np.asarray(split.std),
    )


def _weighted_sampler(y: np.ndarray, power: float, seed: int) -> WeightedRandomSampler:
    classes, counts = np.unique(y, return_counts=True)
    count_map = {int(c): int(n) for c, n in zip(classes, counts)}
    weights = np.asarray(
        [1.0 / (float(count_map[int(label)]) ** float(power)) for label in y],
        dtype=np.float64,
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def make_loaders(split: SplitData, cfg: TrainConfig, seed: int) -> dict[str, DataLoader]:
    train_ds = TensorDataset(torch.from_numpy(split.x_train), torch.from_numpy(split.y_train))
    val_ds = TensorDataset(torch.from_numpy(split.x_val), torch.from_numpy(split.y_val))
    test_ds = TensorDataset(torch.from_numpy(split.x_test), torch.from_numpy(split.y_test))

    sampler = _weighted_sampler(split.y_train, cfg.sampler_power, seed) if cfg.use_weighted_sampler else None
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    common = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_ds,
        sampler=sampler,
        shuffle=sampler is None,
        generator=generator if sampler is None else None,
        **common,
    )
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return {"train": train_loader, "val": val_loader, "test": test_loader}


def normalize_bpm(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((np.asarray(x, dtype=np.float32) - float(mean)) / float(std))[:, None, :].astype(np.float32)
