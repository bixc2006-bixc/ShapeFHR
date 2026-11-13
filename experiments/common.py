from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from config import FINAL_TOPK_PER_CLASS, MODEL, MODEL_VERSION, OUTPUT_ROOT, TRAIN_CONFIGS, DatasetConfig, TrainConfig
from data.dataset import SplitData, make_loaders, save_split
from data.fixed_splits import load_paper_arrays
from data.shapelets import load_shapelets
from models.shapefhr import ShapeFHR
from training.checkpoint import load_checkpoint, save_checkpoint
from training.io import save_json
from training.seed import set_seed
from training.trainer import evaluate_model, train_model


# Each ablation instantiates exactly the branches listed here.
VARIANTS = {
    "full": ("shapelet", "generic", "stat"),
    "local_latent": ("shapelet", "generic", "stat"),
    "global_signal": ("shapelet", "generic", "stat"),
    "wo_shapelet_branch": ("generic", "stat"),
    "wo_generic_branch": ("shapelet", "stat"),
    "wo_statistical_branch": ("shapelet", "generic"),
}


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_dataset(dataset_cfg: DatasetConfig) -> SplitData:
    return load_paper_arrays(dataset_cfg, MODEL.seq_len)


def build_model(
    *,
    dataset_cfg: DatasetConfig,
    train_cfg: TrainConfig,
    topk_per_class: int,
    variant: str = "full",
    signal_global_stride: int | None = None,
) -> tuple[ShapeFHR, list]:
    if variant not in VARIANTS:
        raise KeyError(f"unknown variant {variant!r}")
    enabled_branches = VARIANTS[variant]
    shapelets = load_shapelets(dataset_cfg.shapelets_path, topk_per_class) if "shapelet" in enabled_branches else []
    model = ShapeFHR(
        shapelets=shapelets,
        model_cfg=MODEL,
        dropout=train_cfg.dropout,
        enabled_branches=enabled_branches,
        match_mode=variant if variant in {"local_latent", "global_signal"} else "global_latent",
        signal_global_stride=signal_global_stride,
    )
    return model, shapelets


def run_training(
    *,
    dataset_cfg: DatasetConfig,
    train_cfg: TrainConfig,
    split: SplitData,
    output_dir: Path,
    seed: int,
    topk_per_class: int,
    variant: str = "full",
    signal_global_stride: int | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    loaders = make_loaders(split, train_cfg, seed)
    model, shapelets = build_model(
        dataset_cfg=dataset_cfg,
        train_cfg=train_cfg,
        topk_per_class=topk_per_class,
        variant=variant,
        signal_global_stride=signal_global_stride,
    )

    dev = device()
    print(
        f"\n[run] dataset={dataset_cfg.name} seed={seed} variant={variant} "
        f"topk={topk_per_class} device={dev}"
    )
    matching_settings = {}
    if variant == "global_signal":
        stride = model.specific.signal_global_stride
        matching_settings["signal_global_stride"] = stride
        print(f"[matching] signal_global_stride={stride} samples", flush=True)
    trained = train_model(model, loaders, train_cfg, dev)
    model = trained["model"]
    threshold = float(trained["best_threshold"])

    val_metrics, y_val, val_scores = evaluate_model(
        model, loaders["val"], dev, threshold
    )
    test_metrics, y_test, test_scores = evaluate_model(
        model, loaders["test"], dev, threshold
    )

    checkpoint_path = output_dir / "best_shapefhr.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        initial_shapelets=shapelets,
        threshold=threshold,
        epoch=trained["best_epoch"],
        val_metrics=val_metrics,
        normalization={"mean": split.mean, "std": split.std},
        dataset_key=dataset_cfg.key,
        signal_key=dataset_cfg.signal_key,
        topk_per_class=topk_per_class,
        seed=seed,
        train_config=asdict(train_cfg),
    )

    np.save(output_dir / "val_scores.npy", val_scores)
    np.save(output_dir / "test_scores.npy", test_scores)
    np.save(output_dir / "val_y_true.npy", y_val)
    np.save(output_dir / "test_y_true.npy", y_test)

    result = {
        "protocol_version": MODEL_VERSION,
        "dataset_key": dataset_cfg.key,
        "dataset_name": dataset_cfg.name,
        "seed": int(seed),
        "variant": variant,
        "topk_per_class": int(topk_per_class),
        "checkpoint": str(checkpoint_path),
        "best_epoch": int(trained["best_epoch"]),
        "best_threshold": threshold,
        "normalization": {"mean": split.mean, "std": split.std},
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "train_config": asdict(train_cfg),
        "history": trained["history"],
        **matching_settings,
    }
    save_json(result, output_dir / "result.json")
    return result


def save_dataset_protocol(
    *,
    dataset_cfg: DatasetConfig,
    split: SplitData,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_split(split, output_dir / "data_split_indices.npz")
    save_json(
        {
            "dataset": asdict(dataset_cfg),
            "normalization": {"mean": split.mean, "std": split.std},
            "counts": {
                "train": int(len(split.y_train)),
                "val": int(len(split.y_val)),
                "test": int(len(split.y_test)),
            },
        },
        output_dir / "dataset_protocol.json",
    )


def main_run_dir(dataset_key: str, seed: int) -> Path:
    return OUTPUT_ROOT / "main" / dataset_key / f"seed_{seed}"


def main_checkpoint(dataset_key: str, seed: int = 42) -> Path:
    return main_run_dir(dataset_key, seed) / "best_shapefhr.pt"


def load_full_source(source: DatasetConfig, seed: int, dev: torch.device):
    model, state = load_checkpoint(main_checkpoint(source.key, seed), device=dev)
    if ((state["dataset_key"], state["signal_key"], state["seed"], state["topk_per_class"])
            != (source.key, source.signal_key, seed, FINAL_TOPK_PER_CLASS)
            or state["train_config"] != asdict(TRAIN_CONFIGS[source.key])
            or model.model_cfg != MODEL
            or model.enabled_branches != VARIANTS["full"]
            or model.match_mode != "global_latent"):
        raise ValueError("Train the full ShapeFHR model with the current source configuration first")
    return model, state
