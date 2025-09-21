#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
from pathlib import Path
import random

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

# import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.cityscapes import CityScapes
from src.datasets.uavid import UAVid
from src.models.cabinet import CABiNet
from src.scripts.evaluate import MscEvalV0
from src.utils.logger import RichConsoleManager
from src.utils.loss import OhemCELoss
from src.utils.optimizer import Optimizer


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


@hydra.main(version_base=None, config_path="../../configs", config_name="train_citys")
def train_and_evaluate(cfg: DictConfig) -> None:
    console = RichConsoleManager.get_console()
    console.print(OmegaConf.to_yaml(cfg), style="warning")

    respth = Path(cfg.training_config.experiments_path)
    respth.mkdir(parents=True, exist_ok=True)
    """Set Dataset Params."""
    n_classes = cfg.dataset_config.num_classes
    batch_size = cfg.training_config.batch_size
    n_workers = cfg.training_config.num_workers
    cropsize = cfg.dataset_config.cropsize
    ignore_idx = cfg.dataset_config.ignore_idx
    seed_everything(cfg.dataset_config.seed)
    """Prepare DataLoaders."""
    console.print("Preparing dataloaders!", style="info")
    if cfg.dataset_config.name == "cityscapes":
        ds_train = CityScapes(
            config_file=cfg.dataset_config.dataset_config_file,
            ignore_lb=ignore_idx,
            rootpth=cfg.dataset_config.dataset_path,
            cropsize=cropsize,
            mode="train",
        )
        ds_val = CityScapes(
            config_file=cfg.dataset_config.dataset_config_file,
            ignore_lb=ignore_idx,
            rootpth=cfg.dataset_config.dataset_path,
            cropsize=cropsize,
            mode="val",
        )
    elif cfg.dataset_config.name == "uavid":
        ds_train = UAVid(
            config_file=cfg.dataset_config.dataset_config_file,
            ignore_lb=ignore_idx,
            rootpth=cfg.dataset_config.dataset_path,
            cropsize=cropsize,
            mode="train",
        )
        ds_val = UAVid(
            config_file=cfg.dataset_config.dataset_config_file,
            ignore_lb=ignore_idx,
            rootpth=cfg.dataset_config.dataset_path,
            cropsize=cropsize,
            mode="val",
        )
    else:
        raise NotImplementedError(f"Dataset {cfg.dataset_config.name} not supported.")

    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if n_workers > 0 else False,  # Avoid worker restart
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,  # <<<<<<<<<< FIX: Validation should NOT shuffle
        num_workers=n_workers,
        pin_memory=True,
        drop_last=False,  # <<<<<<<<<< Better for eval consistency
        persistent_workers=True if n_workers > 0 else False,
    )
    console.log("Dataloaders ready!", style="info")
    """Build Model."""
    base_path_pretrained = Path("src/models/pretrained_backbones")
    backbone_weights = (base_path_pretrained / cfg.model.pretrained_weights).resolve()
    mode = cfg.model.mode
    cfgs = cfg.model.cfgs

    net = CABiNet(
        n_classes=n_classes, backbone_weights=backbone_weights, mode=mode, cfgs=cfgs
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    console.log("Model moved to device!", style="info")
    """Define Loss Functions."""
    score_thres = 0.7
    # Fix: Ensure n_min is at least 1 and correctly computed
    n_min = batch_size * cropsize[0] * cropsize[1] // 16
    n_min = max(1, n_min)  # Prevent zero/negative

    # Get class weights from config
    if cfg.training_config.class_weights.use_weights:
        weight = (
            torch.tensor(cfg.training_config.class_weights.cityscapes_class_weights)
            .float()
            .to(device)
        )
    else:
        weight = None

    # Create losses
    criteria_p = OhemCELoss(
        thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx, weight=weight
    )
    criteria_16 = OhemCELoss(
        thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx, weight=weight
    )
    """Optimizer Setup."""
    momentum = cfg.training_config.optimizer_momentum
    weight_decay = cfg.training_config.optimizer_weight_decay
    lr_start = cfg.training_config.optimizer_lr_start
    max_iter = cfg.training_config.max_iterations
    power = cfg.training_config.optimizer_power

    # CRITICAL FIX: Typo in config key
    warmup_steps = cfg.training_config.get("warmup_steps", 0)  # Was 'warmup_stemps'
    warmup_start_lr = cfg.training_config.get("warmup_start_lr", lr_start / 10)

    optim = Optimizer(
        model=net,
        lr0=lr_start,
        momentum=momentum,
        wd=weight_decay,
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
        max_iter=max_iter,
        power=power,
    )
    """Training Loop."""
    epochs = cfg.training_config.epochs
    best_loss = float("inf")
    global_step = 0

    scaler = torch.amp.GradScaler(device=device)  # Mixed precision scaler

    def train_step(im, lb):
        im = im.to(device, non_blocking=True)
        lb = lb.to(device, non_blocking=True).squeeze(1)  # Remove channel dim

        optim.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=True):
            out, out16 = net(im)
            loss = criteria_p(out, lb) + criteria_16(out16, lb)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        torch.cuda.synchronize()  # Only needed if measuring time

        return loss.item()

    @torch.no_grad()  # <<<<<<<<<<<<<<<<<: Disable gradients in val
    def val_step(im, lb):
        im = im.to(device, non_blocking=True)
        lb = lb.to(device, non_blocking=True).squeeze(1)

        out, out16 = net(im)
        loss1 = criteria_p(out, lb)
        loss2 = criteria_16(out16, lb)
        loss = loss1 + loss2

        return loss.item()

    console.rule("[bold green]Starting Training[/bold green]")

    try:
        for epoch in range(epochs):
            torch.cuda.empty_cache()  # Light cleanup before epoch

            # --- Training Phase ---
            net.train()
            train_loss = 0.0
            train_pbar = tqdm(dl_train, desc=f"Epoch [{epoch+1}/{epochs}] - Train")
            for ims, lbs in train_pbar:
                loss = train_step(ims, lbs)
                train_loss += loss
                global_step += 1
                train_pbar.set_postfix(loss=loss)

            train_loss /= len(dl_train)  # Proper average

            # --- Validation Phase ---
            torch.cuda.empty_cache()
            net.eval()
            val_loss = 0.0
            val_pbar = tqdm(dl_val, desc="Validation")
            for ims, lbs in val_pbar:
                loss = val_step(ims, lbs)
                val_loss += loss
                val_pbar.set_postfix(val_loss=loss)

            val_loss /= len(dl_val)

            console.print(
                f"Epoch [{epoch+1}/{epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

            # --- Save Best Model ---
            if val_loss < best_loss:
                best_loss = val_loss
                save_name = cfg.training_config.model_save_name.replace(
                    ".pth", "_best.pth"
                )
                save_pth = respth / save_name

                # Save without context of DDP/module wrapping
                state_dict = (
                    net.module.state_dict()
                    if hasattr(net, "module")
                    else net.state_dict()
                )
                torch.save(state_dict, str(save_pth))
                console.print(
                    f"[bold yellow]New best model saved:[/bold yellow] {save_pth}"
                )

            # End of epoch TQDM cleanup
            train_pbar.close()
            val_pbar.close()

        # End of training
        console.rule("[bold blue]Training Completed[/bold blue]")
        console.print(f"✅ Final model trained for {epochs} epochs.")
        console.print(f"🏆 Best validation loss: {best_loss:.4f}")

    except KeyboardInterrupt:
        console.print("[red]Training interrupted by user.[/red]")
    except Exception as e:
        console.print(f"[red]Error during training: {e}[/red]")
        raise
    finally:
        # Always attempt cleanup
        torch.cuda.empty_cache()

    # Save final model
    save_pth_final = respth / cfg.training_config.model_save_name
    state_dict = net.module.state_dict() if hasattr(net, "module") else net.state_dict()
    torch.save(state_dict, str(save_pth_final))
    console.print(f"💾 Final model saved to: {save_pth_final}")

    # Optional: Save config
    config_out = respth / "config.yaml"
    with open(config_out, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    console.print(f"📄 Config saved to: {config_out}")

    # Final evaluation
    console.print("Starting final evaluation...", style="info")
    evaluator = MscEvalV0(
        model=net,
        dataloader=dl_val,
        device=device,
        n_classes=n_classes,
        ignore_label=ignore_idx,
        scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=True,
    )
    results = evaluator.evaluate()
    mIoU = results["mIoU"]
    accuracy = results["accuracy"]
    console.print(f"🏁 Final mIoU on validation set: {mIoU}", style="info")
    console.print(f"🏁 Final Accuracy on validation set: {accuracy}", style="info")


if __name__ == "__main__":
    train_and_evaluate()
