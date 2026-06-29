#!/usr/bin/python
# -*- encoding: utf-8 -*-

import logging
import math
import os
from pathlib import Path
import random

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.utils as nn_utils
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Any, cast
from src.datasets.cityscapes import CityScapes
from src.datasets.uavid import UAVid, uavid_collate_fn
from src.models.cabinet import CABiNet
from src.models.constants import (
    DEFAULT_SCORE_THRESHOLD,
    OHEM_DIVISOR,
)
from src.scripts.evaluate import MscEvalV0
from src.utils.exceptions import ConfigurationError
from src.utils.logger import RichConsoleManager
from src.utils.loss import OhemCELoss
from src.utils.optimizer import Optimizer

logger = logging.getLogger(__name__)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Must be False when deterministic=True


def _model_state_dict(net: torch.nn.Module) -> dict[Any, Any]:
    """Return the model state dict, unwrapping DDP's .module if present."""
    inner = getattr(net, "module", net)
    if not isinstance(inner, torch.nn.Module):
        inner = net
    return cast(dict[Any, Any], inner.state_dict())


def _save_checkpoint(
    path: Path,
    epoch: int,
    net: torch.nn.Module,
    optim: Optimizer,
    scaler: torch.amp.GradScaler,
    best_miou: float,
    best_loss: float,
) -> None:
    """Save a full training checkpoint (model + optimizer + scheduler state)."""
    torch.save(
        {
            "epoch": epoch,
            "model_state": _model_state_dict(net),
            "optimizer_state": optim.state_dict(),
            "optimizer_it": optim.it,
            "scaler_state": scaler.state_dict(),
            "best_miou": best_miou,
            "best_loss": best_loss,
        },
        path,
    )


def _load_checkpoint(
    path: Path,
    net: torch.nn.Module,
    optim: Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple:
    """Load checkpoint; returns (start_epoch, best_miou, best_loss)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    net.load_state_dict(ckpt["model_state"])
    optim.load_state_dict(ckpt["optimizer_state"])
    optim.it = ckpt["optimizer_it"]
    scaler.load_state_dict(ckpt["scaler_state"])
    start_epoch = ckpt["epoch"] + 1
    best_miou = ckpt.get("best_miou", 0.0)
    best_loss = ckpt.get("best_loss", float("inf"))
    return start_epoch, best_miou, best_loss


def _run_miou_eval(
    net: torch.nn.Module,
    dl: DataLoader,
    device: torch.device,
    n_classes: int,
    ignore_idx: int,
    cropsize: int,
    scales: tuple = (1.0,),
    flip: bool = False,
) -> dict:
    """Run MscEvalV0 and return the results dict (or {} on non-rank-0)."""
    evaluator = MscEvalV0(
        model=net,
        dataloader=dl,
        device=device,
        n_classes=n_classes,
        ignore_label=ignore_idx,
        scales=scales,
        flip=flip,
        cropsize=cropsize,
    )
    return evaluator.evaluate()


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def train_and_evaluate(cfg: DictConfig) -> None:
    console = RichConsoleManager.get_console()
    console.print(OmegaConf.to_yaml(cfg), style="warning")

    respth = Path(cfg.training_config.experiments_path)
    respth.mkdir(parents=True, exist_ok=True)

    # ── Dataset params ────────────────────────────────────────────────────────
    n_classes = cfg.dataset.num_classes
    batch_size = cfg.training_config.batch_size
    n_workers = cfg.training_config.num_workers
    cropsize = cfg.dataset.cropsize
    ignore_idx = cfg.dataset.ignore_idx
    seed_everything(cfg.dataset.seed)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    console.print("Preparing dataloaders!", style="info")

    DATASET_REGISTRY = {
        "cityscapes": CityScapes,
        "uavid": UAVid,
    }
    dataset_cls = DATASET_REGISTRY.get(cfg.dataset.name.lower())
    if dataset_cls is None:
        raise NotImplementedError(f"Dataset '{cfg.dataset.name}' not supported.")

    is_uavid = cfg.dataset.name.lower() == "uavid"
    collate = uavid_collate_fn if is_uavid else None
    common_args = dict(
        config_file=cfg.dataset.config_file,
        ignore_lb=ignore_idx,
        rootpth=cfg.dataset.dataset_path,
        cropsize=cropsize,
    )

    ds_train = dataset_cls(**common_args, mode="train")
    ds_val = dataset_cls(**common_args, mode="val")

    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=n_workers > 0,
        collate_fn=collate,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=n_workers > 0,
        collate_fn=collate,
    )
    # Separate eval loader: smaller batch size, no shuffle
    dl_test = DataLoader(
        ds_val,
        batch_size=cfg.validation_config.batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=n_workers > 0,
        collate_fn=collate,
    )
    console.log("Dataloaders ready!", style="info")

    # ── Training hyper-params ─────────────────────────────────────────────────
    epochs = cfg.training_config.epochs
    accum_steps = cfg.training_config.accum_steps
    max_grad_norm = float(cfg.training_config.get("max_grad_norm", 1.0) or 0.0)
    eval_every_n = int(cfg.validation_config.get("eval_every_n_epochs", 1))

    # ── Model ─────────────────────────────────────────────────────────────────
    base_path_pretrained = Path("src/models/pretrained_backbones")
    backbone_weights = (base_path_pretrained / cfg.model.pretrained_weights).resolve()

    net = CABiNet(
        n_classes=n_classes,
        backbone_weights=backbone_weights,
        mode=cfg.model.mode,
        cfgs=cfg.model.cfgs,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    console.log("Model moved to device!", style="info")

    # ── Loss ──────────────────────────────────────────────────────────────────
    score_thres = DEFAULT_SCORE_THRESHOLD
    n_min = max(1, batch_size * cropsize[0] * cropsize[1] // OHEM_DIVISOR)

    if cfg.training_config.class_balancing:
        try:
            weight = torch.tensor(cfg.dataset.class_weights).float().to(device)
        except (AttributeError, KeyError) as e:
            raise ConfigurationError(f"Invalid class_weights in config: {e}")
    else:
        weight = None

    criteria_p = OhemCELoss(
        thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx, weight=weight
    )
    criteria_16 = OhemCELoss(
        thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx, weight=weight
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # max_iter MUST be in optimizer steps (not batches).
    # Each optimizer step spans accum_steps gradient-accumulation micro-batches.
    # We compute this from epochs × batches_per_epoch, then divide by accum_steps.
    cfg_max_iter = cfg.training_config.get("max_iterations", None)
    if cfg_max_iter is not None:
        max_iter = int(cfg_max_iter)
        logger.info(
            f"[train] max_iter overridden by config: {max_iter} optimizer steps"
        )
    else:
        max_iter = math.ceil(epochs * len(dl_train) / accum_steps)
        logger.info(
            f"[train] max_iter auto-computed: {epochs} epochs × {len(dl_train)} batches "
            f"/ {accum_steps} accum_steps = {max_iter} optimizer steps"
        )

    warmup_steps = int(cfg.training_config.get("warmup_steps", 0))
    warmup_start_lr = float(
        cfg.training_config.get(
            "warmup_start_lr", cfg.training_config.optimizer_lr_start / 10
        )
    )

    optim = Optimizer(
        model=net,
        lr0=cfg.training_config.optimizer_lr_start,
        momentum=cfg.training_config.optimizer_momentum,
        wd=cfg.training_config.optimizer_weight_decay,
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
        max_iter=max_iter,
        power=cfg.training_config.optimizer_power,
    )

    scaler = torch.amp.GradScaler(device=device.type)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 0
    best_miou = 0.0
    best_loss = float("inf")

    ckpt_path = respth / "checkpoint_last.pth"
    if cfg.training_config.get("resume", False):
        if ckpt_path.exists():
            start_epoch, best_miou, best_loss = _load_checkpoint(
                ckpt_path, net, optim, scaler, device
            )
            console.print(
                f"✅ Resumed from checkpoint: epoch {start_epoch - 1} "
                f"(best mIoU={best_miou:.4f})",
                style="info",
            )
        else:
            console.print(
                f"[yellow]resume=True but no checkpoint found at {ckpt_path}. "
                "Starting fresh.[/yellow]"
            )

    # ── Inner step functions ───────────────────────────────────────────────────
    def _optimizer_step() -> None:
        """Unscale → clip → step → update → zero_grad."""
        if max_grad_norm > 0:
            # GradScaler stubs require torch.optim.Optimizer; our wrapper is
            # compatible at runtime (exposes .param_groups, .step(), .zero_grad()).
            scaler.unscale_(optim)  # type: ignore[arg-type]
            nn_utils.clip_grad_norm_(net.parameters(), max_grad_norm)
        scaler.step(optim)  # type: ignore[arg-type]
        scaler.update()
        optim.zero_grad()

    def train_step(im: torch.Tensor, lb: torch.Tensor, i: int) -> float:
        im = im.to(device, non_blocking=True)
        lb = lb.to(device, non_blocking=True).squeeze(1)

        with torch.amp.autocast(device_type=device.type, enabled=True):
            out, out16 = net(im)
            loss = (criteria_p(out, lb) + criteria_16(out16, lb)) / accum_steps
        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0:
            _optimizer_step()

        return float(loss.item())

    def val_step(im: torch.Tensor, lb: torch.Tensor) -> float:
        im = im.to(device, non_blocking=True)
        lb = lb.to(device, non_blocking=True).squeeze(1)
        out, out16 = net(im)
        return float((criteria_p(out, lb) + criteria_16(out16, lb)).item())

    # ── Training loop ─────────────────────────────────────────────────────────
    console.rule("[bold green]Starting Training[/bold green]")

    try:
        for epoch in range(start_epoch, epochs):
            torch.cuda.empty_cache()

            # --- Train ---
            net.train()
            train_loss = 0.0
            optim.zero_grad()
            train_pbar = tqdm(dl_train, desc=f"Epoch [{epoch+1}/{epochs}] - Train")
            for i, (ims, lbs) in enumerate(train_pbar):
                loss = train_step(ims, lbs, i)
                train_loss += loss
                train_pbar.set_postfix(
                    loss=f"{loss:.4f}",
                    lr=f"{optim.get_lr(0, optim.optim.param_groups[0]):.2e}",
                )

            # Flush any trailing partial accumulation window
            if len(dl_train) % accum_steps != 0:
                _optimizer_step()

            train_loss /= len(dl_train)

            # --- Validate (loss) ---
            torch.cuda.empty_cache()
            net.eval()
            val_loss = 0.0
            val_pbar = tqdm(dl_val, desc="Validation")
            for ims, lbs in val_pbar:
                val_loss += val_step(ims, lbs)
                val_pbar.set_postfix(val_loss=f"{val_loss:.4f}")
            val_loss /= len(dl_val)

            # --- Per-epoch mIoU (lightweight: 1 scale, no flip) ---
            epoch_miou = 0.0
            if (epoch + 1) % eval_every_n == 0:
                miou_results = _run_miou_eval(
                    net,
                    dl_test,
                    device,
                    n_classes,
                    ignore_idx,
                    cropsize=max(cropsize),
                    scales=(1.0,),
                    flip=False,
                )
                if miou_results:
                    epoch_miou = miou_results["mIoU"]

            console.print(
                f"Epoch [{epoch+1}/{epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
                + (f" | mIoU: {epoch_miou:.4f}" if epoch_miou > 0 else ""),
            )

            # --- Save best model (mIoU criterion) ---
            if epoch_miou > best_miou:
                best_miou = epoch_miou
                save_name = cfg.training_config.model_save_name.replace(
                    ".pth", "_best.pth"
                )
                torch.save(_model_state_dict(net), respth / save_name)
                console.print(
                    f"[bold yellow]✨ New best model[/bold yellow] "
                    f"(mIoU={best_miou:.4f}) → {save_name}"
                )

            # Also track best loss for backward compatibility
            if val_loss < best_loss:
                best_loss = val_loss

            # --- Checkpoint (every epoch) ---
            _save_checkpoint(ckpt_path, epoch, net, optim, scaler, best_miou, best_loss)

            train_pbar.close()
            val_pbar.close()

        # ── End of training ───────────────────────────────────────────────────
        console.rule("[bold blue]Training Completed[/bold blue]")
        console.print(f"✅ Trained for {epochs} epochs.")
        console.print(f"🏆 Best mIoU (single-scale): {best_miou:.4f}")
        console.print(f"🏆 Best val loss:            {best_loss:.4f}")

    except KeyboardInterrupt:
        console.print("[red]Training interrupted by user.[/red]")
    except Exception as e:
        console.print(f"[red]Error during training: {e}[/red]")
        raise
    finally:
        torch.cuda.empty_cache()

    # ── Save final model ──────────────────────────────────────────────────────
    save_pth_final = respth / cfg.training_config.model_save_name
    torch.save(_model_state_dict(net), str(save_pth_final))
    console.print(f"💾 Final model saved to: {save_pth_final}")

    config_out = respth / "config.yaml"
    with open(config_out, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    console.print(f"📄 Config saved to: {config_out}")

    # ── Final multi-scale evaluation ──────────────────────────────────────────
    console.print("Starting final multi-scale evaluation...", style="info")
    eval_scales = tuple(cfg.validation_config.get("eval_scales", (1.0,)))
    eval_flip = bool(cfg.validation_config.get("flip", True))
    results = _run_miou_eval(
        net,
        dl_test,
        device,
        n_classes,
        ignore_idx,
        cropsize=max(cropsize),
        scales=eval_scales,
        flip=eval_flip,
    )
    if results:
        console.print(f"🏁 Final mIoU:     {results['mIoU']:.4f}", style="info")
        console.print(f"🏁 Final accuracy: {results['accuracy']:.4f}", style="info")


if __name__ == "__main__":
    train_and_evaluate()
