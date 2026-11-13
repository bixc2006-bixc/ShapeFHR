"""Generate a class-balanced PIP/CID/information-gain bank from the fixed training split."""
import argparse
import pickle
from pathlib import Path

from config import DATASETS, MODEL, TOPK_STUDY
from data.fixed_splits import paper_files, read_signal
from data.offline_discovery import OfflineShapeletDiscovery
from data.shapelets import load_shapelets
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=DATASETS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args()
    cfg = DATASETS[args.source]
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}")
    paths, labels, _ = paper_files(cfg)["train"]
    x = np.stack([read_signal(p, cfg.signal_key, cfg.label_key, y, MODEL.seq_len)
                  for p, y in zip(paths, labels)])
    mean, std = x.mean(), x.std()
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Invalid training-set standard deviation")
    x -= mean
    x /= std
    discoverer = OfflineShapeletDiscovery(
        pip_ratio=0.1, window_size=80, max_candidates_per_series=300,
        clinical_weight=0.0, overlap_outer=0.2, overlap_inner=0.1,
        device_preference=args.device,
    )
    k = max(TOPK_STUDY)
    bank = discoverer.fit(x, labels, top_k_total=2 * k)
    if any(len(bank.get(cls, ())) < k for cls in (0, 1)):
        raise ValueError(f"Discovery produced fewer than {k} templates in a class")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as handle:
        pickle.dump(bank, handle)
    load_shapelets(args.out, k)


if __name__ == "__main__":
    main()
