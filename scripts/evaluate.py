#!/usr/bin/python
# -*- encoding: utf-8 -*-

from src.utils.logger import setup_logger
from src.models.cabinet import CABiNet
from src.datasets.cityscapes import CityScapes

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

import logging
import numpy as np
from tqdm import tqdm
import math


class MscEval(object):
    def __init__(self, model, dataloader, params):
        self.scales = params["validation_config"]["eval_scales"]
        self.n_classes = params["dataset_config"]["num_classes"]
        self.lb_ignore = params["dataset_config"]["ignore_idx"]
        self.flip = params["validation_config"]["flip"]
        self.cropsize = params["dataset_config"]["cropsize"][0]
        self.dl = dataloader
        self.net = model

    def pad_tensor(self, inten, size):
        N, C, H, W = inten.size()
        outten = torch.zeros(N, C, size[0], size[1]).cuda()
        outten.requires_grad = False
        margin_h, margin_w = size[0] - H, size[1] - W
        hst, hed = margin_h // 2, margin_h // 2 + H
        wst, wed = margin_w // 2, margin_w // 2 + W
        outten[:, :, hst:hed, wst:wed] = inten
        return outten, [hst, hed, wst, wed]

    def eval_chip(self, crop):
        with torch.no_grad():
            out = self.net(crop)[0]
            prob = F.softmax(out, 1)
            if self.flip:
                crop = torch.flip(crop, dims=(3,))
                out = self.net(crop)[0]
                out = torch.flip(out, dims=(3,))
                prob += F.softmax(out, 1)
            prob = torch.exp(prob)
        return prob

    def crop_eval(self, im):
        cropsize = self.cropsize
        stride_rate = 5 / 6.0
        N, C, H, W = im.size()
        long_size, short_size = (H, W) if H > W else (W, H)
        if long_size < cropsize:
            im, indices = self.pad_tensor(im, (cropsize, cropsize))
            prob = self.eval_chip(im)
            prob = prob[:, :, indices[0] : indices[1], indices[2] : indices[3]]
        else:
            stride = math.ceil(cropsize * stride_rate)
            if short_size < cropsize:
                if H < W:
                    im, indices = self.pad_tensor(im, (cropsize, W))
                else:
                    im, indices = self.pad_tensor(im, (H, cropsize))
            N, C, H, W = im.size()
            n_x = math.ceil((W - cropsize) / stride) + 1
            n_y = math.ceil((H - cropsize) / stride) + 1
            prob = torch.zeros(N, self.n_classes, H, W).cuda()
            prob.requires_grad = False
            for iy in range(n_y):
                for ix in range(n_x):
                    hed, wed = min(H, stride * iy + cropsize), min(
                        W, stride * ix + cropsize
                    )
                    hst, wst = hed - cropsize, wed - cropsize
                    chip = im[:, :, hst:hed, wst:wed]
                    prob_chip = self.eval_chip(chip)
                    prob[:, :, hst:hed, wst:wed] += prob_chip
            if short_size < cropsize:
                prob = prob[:, :, indices[0] : indices[1], indices[2] : indices[3]]
        return prob

    def scale_crop_eval(self, im, scale):
        N, C, H, W = im.size()
        new_hw = [int(H * scale), int(W * scale)]
        im = F.interpolate(im, new_hw, mode="bilinear", align_corners=True)
        prob = self.crop_eval(im)
        prob = F.interpolate(prob, (H, W), mode="bilinear", align_corners=True)
        return prob

    def compute_hist(self, pred, lb):
        n_classes = self.n_classes
        ignore_idx = self.lb_ignore
        keep = np.logical_not(lb == ignore_idx)
        merge = pred[keep] * n_classes + lb[keep]
        hist = np.bincount(merge, minlength=n_classes**2)
        hist = hist.reshape((n_classes, n_classes))
        return hist

    def evaluate(self):
        ## evaluate
        n_classes = self.n_classes
        hist = np.zeros((n_classes, n_classes), dtype=np.float32)
        dloader = tqdm(self.dl)
        if dist.is_initialized() and not dist.get_rank() == 0:
            dloader = self.dl
        for i, (imgs, label) in enumerate(dloader):
            N, _, H, W = label.shape
            probs = torch.zeros((N, self.n_classes, H, W))
            probs.requires_grad = False
            imgs = imgs.cuda()
            for sc in self.scales:
                prob = self.scale_crop_eval(imgs, sc)
                probs += prob.detach().cpu()
            probs = probs.data.numpy()
            preds = np.argmax(probs, axis=1)

            hist_once = self.compute_hist(preds, label.data.numpy().squeeze(1))
            hist = hist + hist_once
        IOUs = np.diag(hist) / (
            np.sum(hist, axis=0) + np.sum(hist, axis=1) - np.diag(hist)
        )
        mIOU = np.mean(IOUs)
        return mIOU


def evaluate(params, save_pth):
    """Setup Logger and Params."""
    setup_logger(params["validation_config"]["validation_output_folder"])
    logger = logging.getLogger()

    """ Setup Model and Load Saved Weights """
    logger.info("\n")
    logger.info("====" * 20)
    logger.info("[INFO]: Begining Evaluation of Model ...\n")
    n_classes = params["dataset_config"]["num_classes"]
    net = CABiNet(n_classes=n_classes)
    net.load_state_dict(torch.load(save_pth))
    net.cuda()
    net.eval()

    """ Setup Validation Dataset """
    batchsize = params["validation_config"]["batch_size"]
    n_workers = params["training_config"]["num_workers"]
    dsval = CityScapes(params, mode="val")
    dl = DataLoader(
        dsval,
        batch_size=batchsize,
        shuffle=False,
        num_workers=n_workers,
        drop_last=False,
    )

    """ Initialize Evaluator Class """
    logger.info("[INFO]: Computing mIOU...")
    evaluator = MscEval(model=net, dataloader=dl, params=params)

    """ Evaluate """
    mIOU = evaluator.evaluate()
    logger.info("[INFO]: mIOU is: {:.6f}".format(mIOU))
    return mIOU


if __name__ == "__main__":
    setup_logger("./res")
    score = evaluate()
