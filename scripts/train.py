#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist


import time
import datetime
import hydra
from pathlib import Path
from shutil import copyfile
from omegaconf import DictConfig, OmegaConf
from src.utils.logger import setup_logger
from src.models.cabinet import CABiNet
from src.datasets.cityscapes import CityScapes
from src.datasets.uavid import UAVid
from src.utils.loss import OhemCELoss
from src.utils.optimizer import Optimizer
from evaluate import MscEval


@hydra.main(version_base=None, config_path="../configs", config_name="train_citys")
def train_and_evaluate(cfg: DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg))
    respth = Path(cfg.training_config.experiments_path)
    Path.mkdir(respth, parents=True, exist_ok=True)

    torch.cuda.set_device(cfg.training_config.gpu_id)
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:33271",
        world_size=torch.cuda.device_count(),
        rank=cfg.training_config.gpu_id,
    )
    setup_logger(respth)
    torch.cuda.synchronize()

    """ Set Dataset Params """
    n_classes = cfg.dataset_config.num_classes
    n_img_per_gpu = cfg.training_config.batch_size
    n_workers = cfg.training_config.num_workers
    cropsize = cfg.dataset_config.cropsize

    """ Prepare DataLoader """
    if cfg.dataset_config.name == "cityscapes":
        ds_train = CityScapes(
            config=cfg.dataset_config.num_classes,
            ignore_lb=cfg.dataset_config.ignore_lb,
            rootpth=cfg.dataset_config.dataset_path,
            cropsize=cfg.dataset_config.cropsize,
            mode="train",
        )
        ds_val = CityScapes(
            config=cfg.dataset_config.num_classes,
            ignore_lb=cfg.dataset_config.ignore_lb,
            rootpth=cfg.dataset_config.dataset_path,
            cropsize=cfg.dataset_config.cropsize,
            mode="val",
        )
    elif cfg.dataset_config.name == "uavid":
        ds_train = UAVid(
            config=cfg.dataset_config.num_classes,
            ignore_lb=cfg.dataset_config.ignore_lb,
            rootpth=cfg.dataset_config.dataset_path,
            cropsize=cfg.dataset_config.cropsize,
            mode="train",
        )
        ds_val = UAVid(
            config=cfg.dataset_config.num_classes,
            ignore_lb=cfg.dataset_config.ignore_lb,
            rootpth=cfg.dataset_config.dataset_path,
            cropsize=cfg.dataset_config.cropsize,
            mode="val",
        )
    else:
        raise NotImplementedError
    sampler = torch.utils.data.distributed.DistributedSampler(ds_train)
    dl_train = DataLoader(
        ds_train,
        batch_size=n_img_per_gpu,
        shuffle=False,
        sampler=sampler,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=n_img_per_gpu,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=True,
    )

    """ Set Model of CABiNet """
    ignore_idx = cfg.dataset_config.ignore_idx
    base_path_pretrained = Path("core/models/pretrained_backbones")
    backbone_weights = (
        base_path_pretrained / cfg.training_config.backbone_weights
    ).resolve()
    net = CABiNet(n_classes=n_classes, backbone_weights=backbone_weights)
    net.cuda()
    net.train()
    net = nn.parallel.DistributedDataParallel(
        net,
        device_ids=[
            cfg.training_config.gpu_id,
        ],
        output_device=cfg.training_config.gpu_id,
        find_unused_parameters=True,
    )
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
        model=net.module,
        lr0=lr_start,
        momentum=momentum,
        wd=weight_decay,
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
        max_iter=max_iter,
        power=power,
    )

    """ Set Train Loop Params """
    msg_iter = cfg.training_config.msg_iterations
    save_steps = int(cfg.training_config.max_iterations / 10)
    best_score = 0.0
    loss_avg = []
    st = glob_st = time.time()
    diter = iter(dl_train)
    epoch = 0
    logger.info("\n")
    logger.info("====" * 20)
    logger.info("[INFO]: Begining Training of Model ...\n")
    for it in range(max_iter):
        try:
            im, lb = next(diter)
            if not im.size()[0] == n_img_per_gpu:
                raise StopIteration
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            diter = iter(dl_train)
            im, lb = next(diter)
        im = im.cuda()
        lb = lb.cuda()
        H, W = im.size()[2:]
        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        out, out16 = net(im)
        loss1 = criteria_p(out, lb)
        loss2 = criteria_16(out16, lb)
        loss = loss1 + loss2
        loss.backward()
        optim.step()
        torch.cuda.synchronize()
        loss_avg.append(loss.item())

        """ Log Values """
        if (it + 1) % msg_iter == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))
            msg = "".join(
                [
                    "it: {it}/{max_it} || ",
                    "lr: {lr:4f} || ",
                    "loss: {loss:.4f} || ",
                    "eta: {eta} || ",
                    "time: {time:.4f}",
                ]
            ).format(
                it=it + 1, max_it=max_iter, lr=lr, loss=loss_avg, time=t_intv, eta=eta
            )
            logger.info(msg)
            loss_avg = []
            st = ed

        if (it + 1) % save_steps == 0:
            save_name = (
                cfg.training_config.model_save_name.split(".pth")[0]
                + f"_iter_{it + 1}.pth"
            )
            save_pth = respth / save_name
            net.cpu()
            state = (
                net.module.state_dict() if hasattr(net, "module") else net.state_dict()
            )
            if dist.get_rank() == 0:
                torch.save(state, str(save_pth))
            logger.info(
                f"[INFO]: {it + 1} iterations Finished!; Model Saved to: {save_pth}"
            )
            net.cuda()
            net.eval()
            torch.cuda.synchronize()
            evaluator = MscEval(net, dl_val, params)
            current_score = evaluator.evaluate()
            if current_score > best_score:
                save_name = (
                    cfg.training_config.model_save_name.split(".pth")[0]
                    + f"_iter_{it + 1}_best_mIOU_{current_score:.4f}.pth"
                )
                save_pth = respth / save_name
                state = (
                    net.module.state_dict()
                    if hasattr(net, "module")
                    else net.state_dict()
                )
                if dist.get_rank() == 0:
                    torch.save(state, str(save_pth))
                print(
                    f"[INFO]: mIOU imporved from {best_score:.4f} to"
                    f" {current_score:.4f}"
                )
                best_score = current_score
            else:
                print(f"[INFO]: mIOU did not improve from {best_score:.4f}")
            net.cuda()
            net.train()
            torch.cuda.synchronize()

    """ Dump and Save the Final Model """
    print(f"[INFO]: Epochs Completed {epoch}")
    save_pth = respth / cfg.training_config.model_save_name
    net.cpu()
    state = net.module.state_dict() if hasattr(net, "module") else net.state_dict()
    if dist.get_rank() == 0:
        torch.save(state, str(save_pth))
    logger.info("Training Finished!; Model Saved to: {}".format(save_pth))
    torch.cuda.empty_cache()

    """ Save the Config Files with Experiment """
    config_file_out = respth / config
    copyfile(config, config_file_out)
    p = Path(".")
    file_list = list(p.glob("**/*.py"))
    for file in file_list:
        file_out = respth / file
    copyfile(str(file), str(file_out))


if __name__ == "__main__":
    train_and_evaluate()
