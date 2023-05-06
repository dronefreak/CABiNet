#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
from torch.utils.data import DataLoader
import hydra
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from src.models.cabinet import CABiNet
from src.datasets.cityscapes import CityScapes
from src.datasets.uavid import UAVid
from src.utils.loss import OhemCELoss
from src.utils.optimizer import Optimizer
from src.utils.logger import get_rich_console


@hydra.main(version_base=None, config_path="../configs", config_name="train_citys")
def train_and_evaluate(cfg: DictConfig) -> None:

    console = get_rich_console()
    console.print(OmegaConf.to_yaml(cfg), style="warning")
    respth = Path(cfg.training_config.experiments_path)
    Path.mkdir(respth, parents=True, exist_ok=True)

    """ Set Dataset Params """
    n_classes = cfg.dataset_config.num_classes
    n_img_per_gpu = cfg.training_config.batch_size
    n_workers = cfg.training_config.num_workers
    cropsize = cfg.dataset_config.cropsize

    """ Prepare DataLoader """
    console.print("Preparing dataloaders!", style="info")
    if cfg.dataset_config.name == "cityscapes":
        ds_train = CityScapes(
            config_file=cfg.dataset_config.dataset_config_file,
            ignore_lb=cfg.dataset_config.ignore_idx,
            rootpth=cfg.dataset_config.dataset_path,
            cropsize=cfg.dataset_config.cropsize,
            mode="train",
        )
        ds_val = CityScapes(
            config_file=cfg.dataset_config.dataset_config_file,
            ignore_lb=cfg.dataset_config.ignore_idx,
            rootpth=cfg.dataset_config.dataset_path,
            cropsize=cfg.dataset_config.cropsize,
            mode="val",
        )
    elif cfg.dataset_config.name == "uavid":
        ds_train = UAVid(
            config_file=cfg.dataset_config.dataset_config_file,
            ignore_lb=cfg.dataset_config.ignore_idx,
            rootpth=cfg.dataset_config.dataset_path,
            cropsize=cfg.dataset_config.cropsize,
            mode="train",
        )
        ds_val = UAVid(
            config_file=cfg.dataset_config.dataset_config_file,
            ignore_lb=cfg.dataset_config.ignore_idx,
            rootpth=cfg.dataset_config.dataset_path,
            cropsize=cfg.dataset_config.cropsize,
            mode="val",
        )
    else:
        raise NotImplementedError
    dl_train = DataLoader(
        ds_train,
        batch_size=n_img_per_gpu,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=n_img_per_gpu,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=True,
    )
    console.log("Dataset ready!", style="info")

    """ Set Model of CABiNet """
    ignore_idx = cfg.dataset_config.ignore_idx
    base_path_pretrained = Path("src/models/pretrained_backbones")
    backbone_weights = (
        base_path_pretrained / cfg.training_config.backbone_weights
    ).resolve()
    net = CABiNet(n_classes=n_classes, backbone_weights=backbone_weights)
    net.cuda()
    net.train()
    console.log("Model ready!", style="info")

    # Set loss functions
    score_thres = 0.7
    n_min = n_img_per_gpu * cropsize[0] * cropsize[1] // 16
    criteria_p = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_16 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)

    """ Set Optimization Parameters """
    momentum = cfg.training_config.optimizer_momentum
    weight_decay = cfg.training_config.optimizer_weight_decay
    lr_start = cfg.training_config.optimizer_lr_start
    max_iter = cfg.training_config.max_iterations
    power = cfg.training_config.optimizer_power
    warmup_steps = cfg.training_config.warmup_stemps
    warmup_start_lr = cfg.training_config.warmup_start_lr
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

    """ Set Train Loop Params """
    epochs = cfg.training_config.epochs
    epoch = 0
    best_loss = float("inf")

    def train_step(im, lb):
        im = im.cuda()
        lb = lb.cuda()
        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        out, out16 = net(im)
        loss1 = criteria_p(out, lb)
        loss2 = criteria_16(out16, lb)
        loss = loss1 + loss2
        loss.backward()
        optim.step()
        torch.cuda.synchronize()

        step_logs = {}
        step_logs["loss"] = loss.item()
        return step_logs

    def val_step(im, lb):
        im = im.cuda()
        lb = lb.cuda()
        lb = torch.squeeze(lb, 1)

        out, out16 = net(im)
        loss1 = criteria_p(out, lb)
        loss2 = criteria_16(out16, lb)
        loss = loss1 + loss2

        step_logs = {}
        step_logs["loss"] = loss.item()
        return step_logs

    console.rule("Begining training ...")
    for epoch in range(epochs):
        step_count = 0
        torch.cuda.empty_cache()
        train_dataloader_loop = tqdm(dl_train)
        train_logs = {}
        train_logs["loss"] = 0
        net.cuda()
        net.train()
        for im, lb in train_dataloader_loop:
            step_logs = train_step(im, lb)
            step_count += 1
            train_logs["loss"] = step_count / (step_count + 1) * train_logs[
                "loss"
            ] + step_logs["loss"] / (step_count + 1)

            train_dataloader_loop.set_description(f"Epoch [{epoch}/{epochs}]")
            train_dataloader_loop.set_postfix(loss=train_logs["loss"])

        step_count = 0
        torch.cuda.empty_cache()
        val_dataloader_loop = tqdm(dl_val)
        val_logs = {}
        val_logs["loss"] = 0
        net.eval()
        for im, lb in val_dataloader_loop:
            val_step_logs = val_step(im, lb)
            step_count += 1
            val_logs["loss"] = step_count / (step_count + 1) * val_logs[
                "loss"
            ] + val_step_logs["loss"] / (step_count + 1)

            val_dataloader_loop.set_description("Val Step!")
            val_dataloader_loop.set_postfix(val_loss=val_logs["loss"])

        if val_logs["loss"] < best_loss:
            console.print(
                f"Val loss improved from {best_loss:.4f} to {val_logs['loss']:.4f}!"
            )
            best_loss = val_logs["loss"]
            save_name = (
                cfg.training_config.model_save_name.split(".pth")[0] + "_best_model.pth"
            )
            save_pth = respth / save_name
            console.print(f"Saving model to {str(save_pth)}!")
            net.cpu()
            state = (
                net.module.state_dict() if hasattr(net, "module") else net.state_dict()
            )
            torch.save(state, str(save_pth))

        """ Log Values """

    """ Dump and Save the Final Model """
    console.rule("Training finished!")
    console.print(f"[INFO]: Epochs Completed {epoch + 1}")
    save_pth = respth / cfg.training_config.model_save_name
    net.cpu()
    state = net.module.state_dict() if hasattr(net, "module") else net.state_dict()
    torch.save(state, str(save_pth))
    # logger.info("Training Finished!; Model Saved to: {}".format(save_pth))
    torch.cuda.empty_cache()

    """ Save the Config Files with Experiment """
    # config_file_out = respth / config
    # copyfile(config, config_file_out)
    # p = Path(".")
    # file_list = list(p.glob("**/*.py"))
    # for file in file_list:
    #     file_out = respth / file
    # copyfile(str(file), str(file_out))


if __name__ == "__main__":
    train_and_evaluate()
