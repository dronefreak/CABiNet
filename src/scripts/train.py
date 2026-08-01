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
from src.datasets.registry import DATASET_KWARGS_BUILDERS, DATASET_REGISTRY
from src.models.cabinet import CABiNet
from src.utils.class_weights import compute_class_weights, get_class_pixel_counts
from src.models.constants import (
    DEFAULT_SCORE_THRESHOLD,
    OHEM_DIVISOR,
)
from src.scripts.evaluate import MscEvalV0
from src.utils.early_stopping import EarlyStopping
from src.utils.ema import ModelEMA
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
    ema: ModelEMA,
    stopper: EarlyStopping,
    best_miou: float,
    best_loss: float,
) -> None:
    """Save a full training checkpoint (model + optimizer + scheduler + EMA + early-stop state)."""
    # float(...) defensively: MscEvalV0 returns numpy.float64 (np.nanmean),
    # and torch.save of a numpy scalar can't be loaded back with
    # weights_only=True ("Unsupported global: numpy...scalar"). Cast here so
    # this can't regress regardless of what the caller passes in.
    torch.save(
        {
            "epoch": epoch,
            "model_state": _model_state_dict(net),
            "optimizer_state": optim.state_dict(),
            "optimizer_it": optim.it,
            "scaler_state": scaler.state_dict(),
            "best_miou": float(best_miou),
            "best_loss": float(best_loss),
            "ema_state": ema.ema.state_dict(),
            "ema_updates": ema.updates,
            "early_stop_best_fitness": float(stopper.best_fitness),
            "early_stop_best_epoch": stopper.best_epoch,
        },
        path,
    )


def _load_checkpoint(
    path: Path,
    net: torch.nn.Module,
    optim: Optimizer,
    scaler: torch.amp.GradScaler,
    ema: ModelEMA,
    stopper: EarlyStopping,
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

    if "ema_state" in ckpt:
        ema.ema.load_state_dict(ckpt["ema_state"])
        ema.updates = ckpt.get("ema_updates", 0)
    else:
        logger.warning(
            "No EMA state in checkpoint (pre-EMA checkpoint) — "
            "reinitializing EMA from the resumed model weights."
        )
        ema.ema.load_state_dict(_model_state_dict(net))
        ema.updates = 0

    stopper.best_fitness = ckpt.get("early_stop_best_fitness", 0.0)
    # Default to the checkpoint's own epoch (not 0) so a resume from a late
    # epoch doesn't immediately look like "patience epochs with no improvement".
    stopper.best_epoch = ckpt.get("early_stop_best_epoch", ckpt["epoch"])

    return start_epoch, best_miou, best_loss


def _load_pretrained_checkpoint(net: torch.nn.Module, path: Path, device: torch.device) -> None:
    """Warm-start model weights from a checkpoint trained on a (possibly
    different) dataset, e.g. finetuning on AeroScapes/VDD/etc. from a
    UAVid-trained checkpoint — converges much faster than backbone-only
    (ImageNet) initialization since the CAB/spatial/fusion layers and the
    backbone are all already adapted to aerial imagery.

    Unlike ``_load_checkpoint`` (used by ``resume``), this only ever touches
    model weights: the optimizer, scaler, EMA shadow, and epoch counter all
    start fresh — this is a new training run, not a continuation of the
    checkpoint's own run.

    Accepts either a raw model state_dict or a full training checkpoint
    (``{"model_state": ..., ...}``, as produced by ``_save_checkpoint``).
    Only parameters whose name AND shape match the current model are loaded
    — this is what actually enables cross-dataset finetuning, since the two
    classifier heads (``ab.b4``, ``conv_out.conv_out``) are sized by
    ``n_classes`` and virtually never match between datasets (UAVid=8,
    AeroScapes=12, VDD=7, Cityscapes=19, …). Those are silently left at
    their freshly-initialized values; everything else (backbone, CAB,
    spatial branch, feature fusion) transfers.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    pretrained_state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt

    model_state = _model_state_dict(net)
    compatible = {
        k: v
        for k, v in pretrained_state.items()
        if k in model_state and v.shape == model_state[k].shape
    }
    skipped_shape_mismatch = [
        k for k in pretrained_state if k in model_state and k not in compatible
    ]
    skipped_unknown = [k for k in pretrained_state if k not in model_state]

    model_state.update(compatible)
    net.load_state_dict(model_state)

    logger.info(
        "Loaded %d/%d pretrained tensors from %s (%d skipped: shape mismatch — "
        "likely the classifier heads, expected when n_classes differs)",
        len(compatible),
        len(pretrained_state),
        path,
        len(skipped_shape_mismatch),
    )
    if skipped_shape_mismatch:
        logger.info("  Shape-mismatched (left at fresh init): %s", skipped_shape_mismatch)
    if skipped_unknown:
        logger.warning("  Unknown keys in checkpoint (ignored): %s", skipped_unknown)


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

    dataset_name = cfg.dataset.name.lower()
    dataset_cls = DATASET_REGISTRY.get(dataset_name)
    if dataset_cls is None:
        raise NotImplementedError(f"Dataset '{cfg.dataset.name}' not supported.")

    is_uavid = dataset_name == "uavid"

    # UAVid source images are not uniform resolution (both 3840x2160 and
    # 4096x2160 occur), and val/test mode applies no crop — a DataLoader
    # batching more than one sample will crash inside torch.stack. Rather
    # than silently coerce the config, fail early with a clear message.
    if is_uavid and int(cfg.validation_config.batch_size) != 1:
        raise ConfigurationError(
            "validation_config.batch_size must be 1 for UAVid — source images "
            "are not uniform resolution and val/test mode applies no crop, so "
            "a larger batch cannot be stacked. Set validation_config.batch_size=1."
        )
    # dl_val (the per-epoch loss-monitoring loader) piggybacks on the training
    # batch_size rather than its own config knob — for UAVid it must also be 1.
    val_loss_batch_size = 1 if is_uavid else batch_size

    common_args = DATASET_KWARGS_BUILDERS[dataset_name](cfg, ignore_idx, cropsize)

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
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=val_loss_batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=n_workers > 0,
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

    # Warm-start from a full checkpoint trained on a (possibly different)
    # dataset — e.g. finetune AeroScapes/VDD from a UAVid-trained checkpoint,
    # which converges much faster than backbone-only (ImageNet) init since
    # the CAB/spatial/fusion layers are already adapted to aerial imagery.
    # Independent of `resume` (below): this only ever warm-starts weights —
    # optimizer/EMA/epoch always start fresh. If `resume=True` also finds an
    # existing checkpoint_last.pth in THIS run's experiments_path, that full
    # state (including further-trained weights) takes priority and overwrites
    # what's loaded here.
    pretrained_ckpt = cfg.training_config.get("pretrained_ckpt_path")
    if pretrained_ckpt:
        pretrained_ckpt_path = Path(pretrained_ckpt)
        if not pretrained_ckpt_path.exists():
            raise ConfigurationError(
                f"training_config.pretrained_ckpt_path does not exist: {pretrained_ckpt_path}"
            )
        _load_pretrained_checkpoint(net, pretrained_ckpt_path, device)
        console.print(
            f"✅ Warm-started model weights from {pretrained_ckpt_path}", style="info"
        )

    # EMA shadow model — constructed right after net.to(device) so the
    # internal deepcopy inherits the correct device with no extra .to() call.
    ema = ModelEMA(
        net,
        decay=float(cfg.training_config.get("ema_decay", 0.9999)),
        tau=int(cfg.training_config.get("ema_tau", 2000)),
    )
    stopper = EarlyStopping(patience=int(cfg.training_config.get("patience", 0) or 0))

    # ── Loss ──────────────────────────────────────────────────────────────────
    score_thres = DEFAULT_SCORE_THRESHOLD
    n_min = max(1, batch_size * cropsize[0] * cropsize[1] // OHEM_DIVISOR)

    cls_pw = float(cfg.training_config.get("cls_pw", 0.0) or 0.0)
    if cls_pw > 0:
        class_counts = get_class_pixel_counts(ds_train, n_classes, ignore_lb=ignore_idx)
        weight_np = compute_class_weights(class_counts, cls_pw=cls_pw)
        weight = torch.tensor(weight_np, dtype=torch.float32, device=device)
        console.print(
            f"Class weights (cls_pw={cls_pw}): {weight_np.round(3).tolist()}",
            style="info",
        )
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
                ckpt_path, net, optim, scaler, ema, stopper, device
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
        """Unscale → clip → step → update → zero_grad → EMA update."""
        if max_grad_norm > 0:
            # GradScaler stubs require torch.optim.Optimizer; our wrapper is
            # compatible at runtime (exposes .param_groups, .step(), .zero_grad()).
            scaler.unscale_(optim)  # type: ignore[arg-type]
            nn_utils.clip_grad_norm_(net.parameters(), max_grad_norm)
        prev_it = optim.it
        scaler.step(optim)  # type: ignore[arg-type]
        scaler.update()
        optim.zero_grad()
        # scaler.step() silently skips the wrapped optim.step() on inf/nan
        # gradients (common during early AMP calibration); optim.it only
        # advances when a real step happened, so gate the EMA update on that
        # too — otherwise a skipped step would fold a no-op into the average.
        if optim.it != prev_it:
            ema.update(net)

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

    @torch.no_grad()
    def val_step(im: torch.Tensor, lb: torch.Tensor) -> float:
        # Intentionally the raw (actively-training) model, not the EMA shadow
        # — this tracks live training-loss trajectory, a different concern
        # from "what gets evaluated for mIoU / saved to disk" (see below).
        # no_grad matters more than usual here: UAVid validation runs one
        # full native-resolution image (up to 4096x2160) at a time with no
        # sliding-window tiling, so retaining an autograd graph for a
        # forward-only pass can be the difference between fitting in memory
        # and an OOM.
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
            train_pbar = tqdm(dl_train, desc=f"Epoch [{epoch + 1}/{epochs}] - Train")
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

            # --- Per-epoch mIoU (lightweight: 1 scale, no flip; EMA weights) ---
            epoch_fitness = None  # None = no eval this epoch (distinct from 0.0)
            epoch_miou = 0.0
            if (epoch + 1) % eval_every_n == 0:
                miou_results = _run_miou_eval(
                    ema.ema,
                    dl_test,
                    device,
                    n_classes,
                    ignore_idx,
                    cropsize=max(cropsize),
                    scales=(1.0,),
                    flip=False,
                )
                if miou_results:
                    # MscEvalV0 returns numpy.float64 (np.nanmean); torch.save
                    # of a numpy scalar can't be loaded back with
                    # weights_only=True ("Unsupported global: numpy...scalar")
                    # — cast to a plain Python float before it ever reaches a
                    # checkpoint.
                    epoch_miou = float(miou_results["mIoU"])
                    epoch_fitness = epoch_miou

            console.print(
                f"Epoch [{epoch + 1}/{epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
                + (f" | mIoU: {epoch_miou:.4f}" if epoch_miou > 0 else ""),
            )

            # --- Save best model (mIoU criterion, EMA weights) ---
            if epoch_miou > best_miou:
                best_miou = epoch_miou
                save_name = cfg.training_config.model_save_name.replace(
                    ".pth", "_best.pth"
                )
                torch.save(_model_state_dict(ema.ema), respth / save_name)
                console.print(
                    f"[bold yellow]✨ New best model[/bold yellow] "
                    f"(mIoU={best_miou:.4f}) → {save_name}"
                )

            # Also track best loss for backward compatibility
            if val_loss < best_loss:
                best_loss = val_loss

            # --- Early stopping (mIoU fitness, EMA-evaluated) ---
            # Must run BEFORE _save_checkpoint: stopper's internal state
            # (best_fitness/best_epoch) needs to reflect *this* epoch's
            # result before it's persisted, otherwise a resumed run's
            # checkpoint always carries one-epoch-stale early-stop state.
            should_stop = stopper(epoch, epoch_fitness)

            # --- Checkpoint (every epoch) ---
            _save_checkpoint(
                ckpt_path, epoch, net, optim, scaler, ema, stopper, best_miou, best_loss
            )

            train_pbar.close()
            val_pbar.close()

            if should_stop:
                console.print(
                    f"[yellow]Early stopping: no mIoU improvement in "
                    f"{stopper.patience} epochs (best={stopper.best_fitness:.4f} "
                    f"@ epoch {stopper.best_epoch + 1}).[/yellow]"
                )
                break

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

    # ── Save final model (EMA weights) ────────────────────────────────────────
    save_pth_final = respth / cfg.training_config.model_save_name
    torch.save(_model_state_dict(ema.ema), str(save_pth_final))
    console.print(f"💾 Final model saved to: {save_pth_final}")

    config_out = respth / "config.yaml"
    with open(config_out, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    console.print(f"📄 Config saved to: {config_out}")

    # ── Final multi-scale evaluation (EMA weights) ────────────────────────────
    console.print("Starting final multi-scale evaluation...", style="info")
    eval_scales = tuple(cfg.validation_config.get("eval_scales", (1.0,)))
    eval_flip = bool(cfg.validation_config.get("flip", True))
    results = _run_miou_eval(
        ema.ema,
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
