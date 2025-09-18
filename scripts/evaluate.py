#!/usr/bin/python
# -*- encoding: utf-8 -*-

import logging
import math
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.cityscapes import CityScapes
from src.models.cabinet import CABiNet
from src.utils.logger import setup_logger

# For optional distributed support
try:
    import torch.distributed as dist
except ImportError:
    dist = None


class MscEvalV0(object):
    """Multi-Scale Crop Evaluation for semantic segmentation.

    Supports flipping, scaling, sliding window inference.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        n_classes: int,
        ignore_label: int = 255,
        scales: Tuple[float] = (1.0,),
        flip: bool = False,
        cropsize: int = 1024,
        device: torch.device = None,
    ):
        self.model = model
        self.dl = dataloader
        self.n_classes = n_classes
        self.ignore_label = ignore_label
        self.scales = scales
        self.flip = flip
        self.cropsize = cropsize
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def pad_tensor(
        self, tensor: torch.Tensor, size: tuple
    ) -> Tuple[torch.Tensor, list]:
        """Pad tensor to target size, return crop indices."""
        N, C, H, W = tensor.shape
        pad_H, pad_W = max(size[0] - H, 0), max(size[1] - W, 0)
        hst, wst = pad_H // 2, pad_W // 2
        hed, wed = hst + H, wst + W

        padded = torch.zeros(N, C, size[0], size[1], device=tensor.device)
        padded[:, :, hst:hed, wst:wed] = tensor
        indices = [hst, hed, wst, wed]
        return padded, indices

    def eval_chip(self, crop: torch.Tensor) -> torch.Tensor:
        """Forward pass on a single chip."""
        with torch.no_grad():
            logits = self.model(crop)[0]  # (B, C, h, w)
            prob = F.softmax(logits, dim=1)

            if self.flip:
                flipped_crop = torch.flip(crop, dims=(3,))
                flipped_logits = self.model(flipped_crop)[0]
                flipped_prob = F.softmax(torch.flip(flipped_logits, dims=(3,)), dim=1)
                prob += flipped_prob
                prob *= 0.5  # Average

        return prob  # already exp(log_softmax) → probability

    def crop_eval(self, image: torch.Tensor) -> torch.Tensor:
        """Sliding window inference with overlap.

        Args:
            image: (N, 3, H, W)
        Returns:
            fused probability map: (N, n_classes, H, W)
        """
        cropsize = self.cropsize
        stride_rate = 5 / 6.0
        N, C, H, W = image.shape

        # Case 1: Image smaller than cropsize
        if H < cropsize or W < cropsize:
            long_size = max(H, W)
            target_size = (
                (cropsize, cropsize)
                if long_size < cropsize
                else (cropsize if H < W else H, cropsize if W < H else W)
            )
            image, indices = self.pad_tensor(image, target_size)
            full_H, full_W = image.shape[2:]
        else:
            full_H, full_W = H, W
            indices = None

        # Prepare output buffer
        prob = torch.zeros((N, self.n_classes, full_H, full_W), device=image.device)

        if full_H < cropsize or full_W < cropsize:
            chip = image
            prob += self.eval_chip(chip)
        else:
            stride = int(cropsize * stride_rate)
            n_x = math.ceil((full_W - cropsize) / stride) + 1
            n_y = math.ceil((full_H - cropsize) / stride) + 1

            for iy in range(n_y):
                for ix in range(n_x):
                    y_end = min(full_H, stride * iy + cropsize)
                    x_end = min(full_W, stride * ix + cropsize)
                    y_start = y_end - cropsize
                    x_start = x_end - cropsize

                    chip = image[:, :, y_start:y_end, x_start:x_end]
                    chip_prob = self.eval_chip(chip)
                    prob[:, :, y_start:y_end, x_start:x_end] += chip_prob

        # Remove padding if applied
        if indices is not None:
            hst, hed, wst, wed = indices
            prob = prob[:, :, hst:hed, wst:wed]

        return prob

    def scale_crop_eval(self, image: torch.Tensor, scale: float) -> torch.Tensor:
        """Apply multi-scale evaluation."""
        N, C, H, W = image.shape
        new_size = [int(H * scale), int(W * scale)]
        scaled_img = F.interpolate(
            image, new_size, mode="bilinear", align_corners=False
        )
        prob = self.crop_eval(scaled_img)
        prob = F.interpolate(prob, (H, W), mode="bilinear", align_corners=False)
        return prob

    @staticmethod
    def compute_hist(
        pred: np.ndarray, label: np.ndarray, n_classes: int, ignore_label: int
    ):
        """Compute confusion matrix."""
        valid = label != ignore_label
        pred = pred[valid].astype(int)
        label = label[valid].astype(int)

        # Clip to valid range
        pred = np.clip(pred, 0, n_classes - 1)

        intersection = pred * n_classes + label
        hist = np.bincount(intersection, minlength=n_classes**2)
        hist = hist.reshape(n_classes, n_classes)
        return hist

    def evaluate(self) -> Dict[str, Any]:
        """Run full evaluation."""
        self.model.eval()
        device = next(self.model.parameters()).device  # Auto-detect model device
        hist = np.zeros((self.n_classes, self.n_classes), dtype=np.float64)

        # Use rank 0 for progress bar
        is_dist = dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        # world_size = dist.get_world_size() if is_dist else 1

        iterator = self.dl
        if rank == 0:
            iterator = tqdm(self.dl, desc="Evaluating", dynamic_ncols=True)

        with torch.no_grad():
            for images, labels in iterator:
                images = images.to(device, non_blocking=True)
                labels = labels.cpu().numpy().squeeze(1)  # Move to CPU early

                # Aggregate predictions across scales
                probs = torch.zeros(
                    (images.size(0), self.n_classes, *images.shape[-2:]), device=device
                )
                for scale in self.scales:
                    probs += self.scale_crop_eval(images, scale)
                preds = torch.argmax(probs, dim=1).cpu().numpy()

                # Update histogram
                for i in range(labels.shape[0]):
                    hist += self.compute_hist(
                        preds[i], labels[i], self.n_classes, self.ignore_label
                    )

        # Sync across processes if distributed
        if is_dist:
            hist_tensor = torch.from_numpy(hist).to(device)
            dist.reduce(hist_tensor, dst=0, op=dist.ReduceOp.SUM)
            if rank == 0:
                hist = hist_tensor.cpu().numpy()

        # Compute metrics
        if rank == 0:
            ious = np.diag(hist) / (
                hist.sum(axis=0) + hist.sum(axis=1) - np.diag(hist) + 1e-8
            )
            miou = np.nanmean(ious)
            acc = np.diag(hist).sum() / hist.sum()
            cls_iou = {f"class_{i}": ious[i] for i in range(len(ious))}

            return {
                "mIoU": miou,
                "accuracy": acc,
                "iou_per_class": cls_iou,
                "confusion_matrix": hist,
            }
        else:
            return {}

    def __call__(self):
        return self.evaluate()


def evaluate(params: Dict[str, Any], save_pth: str):
    """Main evaluation function."""
    log_path = params["validation_config"].get("validation_output_folder", "./res")
    setup_logger(log_path)
    logger = logging.getLogger(__name__)

    logger.info("=" * 40)
    logger.info("Starting Model Evaluation")
    logger.info("=" * 40)

    # Extract config
    n_classes = params["dataset_config"]["num_classes"]
    ignore_idx = params["dataset_config"]["ignore_idx"]
    batch_size = params["validation_config"]["batch_size"]
    num_workers = params["training_config"]["num_workers"]
    eval_scales = params["validation_config"].get("eval_scales", [1.0])
    flip = params["validation_config"].get("flip", False)
    cropsize = params["dataset_config"]["cropsize"][0]

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = CABiNet(n_classes=n_classes).to(device)

    logger.info(f"Loading weights from {save_pth}")
    state_dict = torch.load(save_pth, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()

    # Dataset
    ds_val = CityScapes(
        config_file=params["dataset_config"]["dataset_config_file"],
        ignore_lb=ignore_idx,
        rootpth=params["dataset_config"]["dataset_path"],
        cropsize=params["dataset_config"]["cropsize"],
        mode="val",
    )
    dl = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
    )

    # Evaluator
    logger.info("Computing mIoU...")
    evaluator = MscEvalV0(
        model=net,
        dataloader=dl,
        n_classes=n_classes,
        ignore_label=ignore_idx,
        scales=eval_scales,
        flip=flip,
        cropsize=cropsize,
        device=device,
    )

    # Evaluate
    result = evaluator()
    if result:  # Only rank 0 has results
        miou = result["mIoU"]
        acc = result["accuracy"]
        logger.info(f"✅ Accuracy: {acc:.4f}")
        logger.info(f"🎯 mIoU: {miou:.4f}")
        return result
    return {}


if __name__ == "__main__":
    # Example usage
    # You'll need to load your actual params here
    # For now, just test structure
    try:
        print("Evaluation script loaded successfully.")
        print("Call `evaluate(params, 'path/to/model.pth')` to run.")
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise
