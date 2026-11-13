from pathlib import Path
from typing import Any

import torch

from config import MODEL_VERSION, ModelConfig
from data.shapelets import ShapeletEntry, deserialize_shapelets, serialize_shapelets
from models.shapefhr import ShapeFHR


def save_checkpoint(
    path: str | Path,
    *,
    model: ShapeFHR,
    initial_shapelets: list[ShapeletEntry],
    threshold: float,
    epoch: int,
    val_metrics: dict,
    normalization: dict,
    dataset_key: str,
    signal_key: str,
    topk_per_class: int,
    seed: int,
    train_config: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_version": MODEL_VERSION,
        "model_state_dict": model.state_dict(),
        "model_settings": model.export_model_settings(),
        "train_config": train_config,
        "initial_shapelets": serialize_shapelets(initial_shapelets),
        "threshold": float(threshold),
        "epoch": int(epoch),
        "val_metrics": val_metrics,
        "normalization": {
            "mean": float(normalization["mean"]),
            "std": float(normalization["std"]),
        },
        "dataset_key": str(dataset_key),
        "signal_key": str(signal_key),
        "topk_per_class": int(topk_per_class),
        "seed": int(seed),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[ShapeFHR, dict[str, Any]]:
    path = Path(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("model_version") != MODEL_VERSION:
        raise RuntimeError(
            f"checkpoint version {payload.get('model_version')!r} does not match {MODEL_VERSION!r}"
        )

    settings = payload["model_settings"]
    model_cfg = ModelConfig(**settings["model_cfg"])
    shapelets = deserialize_shapelets(payload["initial_shapelets"])
    model = ShapeFHR(
        shapelets=shapelets,
        model_cfg=model_cfg,
        dropout=float(settings["dropout"]),
        enabled_branches=tuple(settings["enabled_branches"]),
        match_mode=settings.get("match_mode", "global_latent"),
        signal_global_stride=settings.get("signal_global_stride"),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, payload
