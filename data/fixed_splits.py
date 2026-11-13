"""Read the archived paper split directly, independent of sklearn and disk ordering."""
from pathlib import Path

import numpy as np

from config import DatasetConfig
from data.dataset import LoadedDataset, SplitData, fix_length_1d

PARTS = ("train", "val", "test")


def paper_files(cfg: DatasetConfig):
    root, archive = cfg.data_path, cfg.split_manifest
    available = {part: {p.name: p for p in directory.glob("*.npz")}
                 for part, directory in (("flat", root), *((p, root / p) for p in PARTS))}
    result = {}
    names_seen = set()
    with np.load(archive, allow_pickle=False) as fixed:
        for part, count in zip(PARTS, cfg.split_counts):
            names = [Path(name.replace("\\", "/")).name for name in fixed[f"{part}_file_names"]]
            labels = fixed[f"y_{part}"]
            indices = fixed[f"{part}_indices"]
            if (labels.ndim != 1 or not np.isin(labels, (0, 1)).all()
                    or indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)):
                raise ValueError(f"{archive}: invalid {part} labels or indices")
            if len(names) != count or not len(names) == len(labels) == len(indices):
                raise ValueError(f"{archive}: invalid {part} manifest")
            paths = []
            for name in names:
                if name in names_seen:
                    raise ValueError(f"{archive}: duplicate record {name}")
                names_seen.add(name)
                path = available["flat"].get(name) or available[part].get(name)
                if path is None:
                    raise FileNotFoundError(f"{root}: paper {part} record missing: {name}")
                paths.append(path)
            result[part] = (paths, labels.astype(np.int64), indices)
    labels = np.concatenate([result[part][1] for part in PARTS])
    if tuple(np.bincount(labels, minlength=2)) != cfg.class_counts:
        raise ValueError(f"{archive}: class counts do not match the dataset protocol")
    indices = np.concatenate([result[part][2] for part in PARTS])
    if not np.array_equal(np.sort(indices), np.arange(len(labels))):
        raise ValueError(f"{archive}: split indices must partition the dataset")
    return result


def read_signal(path, signal_key, label_key, expected_label, seq_len):
    with np.load(path, allow_pickle=True) as record:
        if signal_key not in record or label_key not in record:
            raise KeyError(f"{path}: requires {signal_key!r} and {label_key!r}; found {record.files}")
        labels = np.asarray(record[label_key]).reshape(-1)
        if not labels.size or labels[0] != expected_label:
            raise ValueError(f"{path}: label does not match paper manifest ({expected_label})")
        x = np.asarray(record[signal_key], dtype=np.float32).reshape(-1)
    if not x.size:
        raise ValueError(f"{path}: empty {signal_key}")
    return fix_length_1d(x, seq_len)


def load_paper_arrays(cfg: DatasetConfig, seq_len: int) -> SplitData:
    groups = paper_files(cfg)
    split = {}
    all_paths = [""] * sum(len(paths) for paths, _, _ in groups.values())
    normalization = None
    for part, (paths, labels, indices) in groups.items():
        x = np.empty((len(paths), seq_len), dtype=np.float32)
        for i, (path, label) in enumerate(zip(paths, labels)):
            x[i] = read_signal(path, cfg.signal_key, cfg.label_key, label, seq_len)
        if part == "train":
            mean, std = float(x.mean()), float(x.std())
            if not np.isfinite(mean) or not np.isfinite(std) or std < 1e-8:
                raise ValueError(f"{cfg.key}: invalid training normalization")
            normalization = {"mean": mean, "std": std}
        x -= normalization["mean"]
        x /= normalization["std"]
        split[f"x_{part}"] = x[:, None, :]
        split[f"y_{part}"] = labels
        split[f"{part}_indices"] = indices
        for index, path in zip(indices, paths):
            all_paths[index] = str(path)
    print(f"[split] {cfg.key} {cfg.signal_key}: paper train/val/test "
          f"{tuple(len(split[f'y_{part}']) for part in PARTS)}, "
          f"mean={normalization['mean']:.6f}, std={normalization['std']:.6f}", flush=True)
    return SplitData(**split, **normalization, file_names=all_paths)


def load_private_test(cfg: DatasetConfig, seq_len: int) -> LoadedDataset:
    paths, labels, _ = paper_files(cfg)["test"]
    signals = np.stack([read_signal(path, cfg.signal_key, cfg.label_key, label, seq_len)
                        for path, label in zip(paths, labels)])
    return LoadedDataset(signals, labels, [str(path) for path in paths])
