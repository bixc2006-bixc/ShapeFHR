"""Load the official BabyBeatAnalyzer TEST partition in bpm."""
from pathlib import Path

import numpy as np

from config import REFM_EXPECTED_TEST_NORMAL, REFM_EXPECTED_TEST_ABNORMAL, SEQ_LEN
from data.dataset import LoadedDataset

BABYBEAT_FHR_MEAN = 142.63593
BABYBEAT_FHR_STD = 11.838785


def load_refm_ts(path: Path) -> LoadedDataset:
    signals, labels = [], []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip().lower() == "@data":
                break
        else:
            raise ValueError(f"{path}: missing @data marker")
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fhr, _, label = line.split(":")
            samples = np.fromstring(fhr, sep=",", dtype=np.float64)
            signals.append(samples * BABYBEAT_FHR_STD + BABYBEAT_FHR_MEAN)
            labels.append(int(label))
    ids = [f"babybeat_test_{i:04d}" for i in range(1, len(labels) + 1)]
    return _validated(signals, labels, ids)


def load_refm_test(path: Path) -> LoadedDataset:
    if path.is_file():
        return load_refm_ts(path)
    records = sorted(path.glob("*.npz"))
    if not records:
        raise FileNotFoundError(f"Set SHAPEFHR_REFM_TEST to the official TEST .ts file or a bpm NPZ directory: {path}")
    signals, labels = [], []
    for record in records:
        with np.load(record, allow_pickle=False) as data:
            signals.append(np.asarray(data["clean_fhr"]).reshape(-1))
            labels.append(np.asarray(data["label"]).item())
    return _validated(signals, labels, [record.stem for record in records])


def _validated(signals, labels, ids):
    if (labels.count(0), labels.count(1)) != (REFM_EXPECTED_TEST_NORMAL, REFM_EXPECTED_TEST_ABNORMAL):
        raise ValueError("BabyBeatAnalyzer TEST must contain 519 normal and 521 abnormal recordings")
    x = np.asarray(signals, dtype=np.float32)
    if x.shape != (len(labels), SEQ_LEN) or not np.isfinite(x).all():
        raise ValueError("Every BabyBeatAnalyzer TEST record must have 4,800 finite FHR values")
    return LoadedDataset(x, np.asarray(labels, dtype=np.int64), ids)
