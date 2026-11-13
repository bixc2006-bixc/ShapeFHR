from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from config import TrainConfig
from models.shapefhr import ShapeFHR
from training.augmentation import augment_batch
from training.losses import FocalLoss, false_positive_penalty
from training.metrics import metrics_from_scores
from training.threshold import find_best_gmean_threshold


def _autocast_context(device: torch.device, enabled: bool):
    if device.type != "cuda" or not enabled:
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def _make_scaler(device: torch.device, enabled: bool):
    if device.type != "cuda" or not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=True)


def _criterion(cfg: TrainConfig) -> torch.nn.Module:
    if cfg.use_focal_loss:
        return FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
    return torch.nn.CrossEntropyLoss()


def _optimizer(model: ShapeFHR, cfg: TrainConfig) -> torch.optim.Optimizer:
    shapelet_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "shapelet_params" in name:
            shapelet_params.append(param)
        else:
            other_params.append(param)

    groups = [{"params": other_params, "lr": cfg.lr}]
    if shapelet_params:
        groups.append({"params": shapelet_params, "lr": cfg.lr * cfg.shapelet_lr_scale})
    return torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)


def _collect_scores(
    model: ShapeFHR,
    loader,
    device: torch.device,
    *,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    all_true: list[torch.Tensor] = []
    all_score: list[torch.Tensor] = []
    total_loss = 0.0
    total_n = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with _autocast_context(device, use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            score = torch.softmax(logits.float(), dim=1)[:, 1]
            all_true.append(y.detach().cpu())
            all_score.append(score.detach().cpu())
            total_loss += float(loss.detach().cpu()) * int(y.numel())
            total_n += int(y.numel())

    return (
        torch.cat(all_true).numpy().astype(np.int64),
        torch.cat(all_score).numpy().astype(np.float64),
        total_loss / max(total_n, 1),
    )


def _train_epoch(
    model: ShapeFHR,
    loader,
    device: torch.device,
    cfg: TrainConfig,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
) -> float:
    model.train()
    criterion = _criterion(cfg)
    amp_enabled = bool(cfg.use_amp and device.type == "cuda")
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_n = 0

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if cfg.train_time_augment:
            x = augment_batch(
                x,
                noise_std=cfg.aug_noise_std,
                mask_ratio=cfg.aug_mask_ratio,
                gain_std=cfg.aug_gain_std,
                baseline_std=cfg.aug_baseline_std,
                max_abs_z=cfg.aug_max_abs_z,
            )

        with _autocast_context(device, amp_enabled):
            if hasattr(model, "training_objective"):
                logits, loss = model.training_objective(x, y, criterion)
            else:
                logits = model(x)
                loss = criterion(logits, y)

            if cfg.fp_penalty_lambda > 0:
                loss = loss + cfg.fp_penalty_lambda * false_positive_penalty(logits, y)

            if getattr(model, "has_shapelet_branch", False) and cfg.shapelet_deviation_lambda > 0:
                loss = loss + (
                    cfg.shapelet_deviation_lambda
                    * model.shapelet_deviation_loss()
                )

            loss_for_backward = loss / max(1, cfg.grad_accum_steps)

        if not torch.isfinite(loss_for_backward):
            optimizer.zero_grad(set_to_none=True)
            continue

        do_step = ((step + 1) % max(1, cfg.grad_accum_steps) == 0) or (step + 1 == len(loader))

        if scaler is not None:
            scaler.scale(loss_for_backward).backward()
            if do_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss_for_backward.backward()
            if do_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().cpu()) * int(y.numel())
        total_n += int(y.numel())

    return total_loss / max(total_n, 1)


def train_model(
    model: ShapeFHR,
    loaders: dict,
    cfg: TrainConfig,
    device: torch.device,
) -> dict:
    model.to(device)
    optimizer = _optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.lr_scheduler_factor,
        patience=cfg.lr_scheduler_patience,
        min_lr=cfg.min_lr,
    )
    scaler = _make_scaler(device, cfg.use_amp)

    if getattr(model, "has_shapelet_branch", False) and cfg.freeze_shapelets_epochs > 0:
        model.set_shapelets_trainable(False)
        print(f"[shapelet] frozen for first {cfg.freeze_shapelets_epochs} epochs")

    history: list[dict] = []
    best_state = None
    best_threshold = 0.5
    best_epoch = 0
    best_val_metrics = None
    best_val_loss = float("inf")
    best_gmean = -float("inf")
    wait = 0

    for epoch in range(1, cfg.epochs + 1):
        if (
            getattr(model, "has_shapelet_branch", False)
            and cfg.freeze_shapelets_epochs > 0
            and epoch == cfg.freeze_shapelets_epochs + 1
        ):
            model.set_shapelets_trainable(True)
            print(f"[shapelet] unfrozen at epoch {epoch}")

        train_loss = _train_epoch(model, loaders["train"], device, cfg, optimizer, scaler)
        y_val, score_val, val_loss = _collect_scores(
            model,
            loaders["val"],
            device,
            use_amp=False,
        )
        threshold, val_metrics = find_best_gmean_threshold(y_val, score_val)
        scheduler.step(val_loss)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "threshold": threshold,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix"},
        }
        history.append(row)

        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"BA={val_metrics['balanced_accuracy']:.4f} Sens={val_metrics['sensitivity']:.4f} "
            f"Spec={val_metrics['specificity']:.4f} G={val_metrics['gmean']:.4f} "
            f"thr={threshold:.3f} lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_metrics["gmean"] > best_gmean + cfg.early_stopping_min_delta:
            best_gmean = float(val_metrics["gmean"])
            best_state = copy.deepcopy(model.state_dict())
            best_threshold = float(threshold)
            best_epoch = int(epoch)
            best_val_metrics = copy.deepcopy(val_metrics)
            best_val_loss = float(val_loss)
            wait = 0
        else:
            wait += 1

        if wait >= cfg.patience:
            print(f"[early-stop] best_epoch={best_epoch}, best_gmean={best_gmean:.6f}")
            break

    if best_state is None:
        raise RuntimeError("training finished without a valid checkpoint")

    model.load_state_dict(best_state)
    return {
        "model": model,
        "history": history,
        "best_threshold": best_threshold,
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "best_val_loss": best_val_loss,
        "best_gmean": best_gmean,
        "train_config": asdict(cfg),
    }


def evaluate_model(
    model: ShapeFHR,
    loader,
    device: torch.device,
    threshold: float,
) -> tuple[dict, np.ndarray, np.ndarray]:
    y_true, y_score, loss = _collect_scores(model, loader, device, use_amp=False)
    metrics = metrics_from_scores(y_true, y_score, threshold)
    metrics["loss"] = float(loss)
    return metrics, y_true, y_score


def predict_scores_array(
    model: ShapeFHR,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(x, dtype=np.float32))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tensor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    scores = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            logits = model(batch)
            scores.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu())
    return torch.cat(scores).numpy().astype(np.float64)
