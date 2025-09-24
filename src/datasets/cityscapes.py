#!/usr/bin/python
# -*- encoding: utf-8 -*-

import json
import os
import os.path as osp
from typing import Tuple
import warnings

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from src.datasets.transform import (
    Compose,
    RandomColorJitter,
    RandomCrop,
    RandomCutout,
    RandomGamma,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomNoise,
    RandomScale,
)


class CityScapes(Dataset):
    """CityScapes Dataset with proper label remapping, safe augmentation, and thread-
    safe preprocessing."""

    def __init__(
        self,
        config_file: str,
        ignore_lb: int,
        rootpth: str,
        cropsize: Tuple[int, int],
        mode: str = "train",
    ):
        super(CityScapes, self).__init__()

        assert mode in ("train", "val", "test"), f"Mode {mode} not supported."
        assert osp.exists(rootpth), f"Dataset root path {rootpth} does not exist."

        self.mode = mode
        self.config_file = config_file
        self.ignore_lb = ignore_lb
        self.rootpth = rootpth
        self.cropsize = tuple(cropsize)

        # Load label mapping: id -> trainId
        with open(self.config_file, "r") as fr:
            labels_info = json.load(fr)
        self.lb_map = {el["id"]: el["trainId"] for el in labels_info}

        # Create reverse mapping as NumPy-friendly lookup table (fast!)
        self._mapping = np.full(256, self.ignore_lb, dtype=np.int64)
        for k, v in self.lb_map.items():
            if k >= 0 and k < 256:
                self._mapping[k] = v

        # Parse image and label paths
        self.imnames = []
        self.imgs = {}
        self.labels = {}

        impth = osp.join(rootpth, "leftImg8bit", mode)
        gtpth = osp.join(rootpth, "gtFine", mode)

        if not osp.exists(impth):
            raise FileNotFoundError(f"Image directory not found: {impth}")
        if not osp.exists(gtpth):
            raise FileNotFoundError(f"Label directory not found: {gtpth}")

        for folder in sorted(os.listdir(impth)):
            im_folder = osp.join(impth, folder)
            gt_folder = osp.join(gtpth, folder)

            for im_name in os.listdir(im_folder):
                if not im_name.endswith("_leftImg8bit.png"):
                    continue

                base_name = im_name.replace("_leftImg8bit.png", "")
                im_path = osp.join(im_folder, im_name)

                # Find corresponding label
                lb_name = f"{base_name}_gtFine_labelIds.png"
                lb_path = osp.join(gt_folder, lb_name)
                if not osp.exists(lb_path):
                    warnings.warn(f"Missing label for {base_name}, skipping.")
                    continue

                self.imnames.append(base_name)
                self.imgs[base_name] = im_path
                self.labels[base_name] = lb_path

        if len(self.imnames) == 0:
            raise RuntimeError(f"No valid image-label pairs found in {mode} set.")

        # Preprocessing
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

        # Training augmentations
        # Only applied in 'train' mode
        # Geometric → Photometric → Regularization is the recommended order.
        self.trans_train = (
            Compose(
                [
                    # Geometric
                    RandomHorizontalFlip(p=0.5),
                    RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
                    RandomCrop(
                        size=self.cropsize,
                        pad_if_needed=True,
                        ignore_label=self.ignore_lb,
                    ),
                    # Photometric
                    RandomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                    RandomGrayscale(p=0.2),
                    RandomGamma(gamma_range=(0.8, 1.2), p=0.3),
                    RandomNoise(mode="gaussian", sigma=0.03, p=0.3),
                    # Regularization
                    RandomCutout(p=0.3, size=64),
                ]
            )
            if mode == "train"
            else None
        )

        self.len = len(self.imnames)

    def __getitem__(self, idx):
        fn = self.imnames[idx]
        impth = self.imgs[fn]
        lbpth = self.labels[fn]
        img = Image.open(impth).convert("RGB")
        label = Image.open(lbpth)  # Keep as PIL until after transforms

        if self.mode == "train":
            im_lb = dict(im=img, lb=label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb["im"], im_lb["lb"]

        # Convert to tensor
        img = self.to_tensor(img)

        # Now convert label to numpy, remap, then tensor
        label = np.array(label, dtype=np.int64)  # Shape (H, W)
        label = self.convert_labels(label)
        label = torch.from_numpy(label).long()  # No extra dim!

        return img, label

    def __len__(self) -> int:
        return self.len

    def convert_labels(self, mask: np.ndarray) -> np.ndarray:
        """Fast label remapping using precomputed lookup table.

        Input: (H, W), values are original IDs
        Output: (H, W), values mapped to trainIds, with ignored labels handled.
        """
        # Use vectorized lookup via indexing
        new_mask = self._mapping[mask]
        return new_mask


# === Test Block (Fixed) ===
if __name__ == "__main__":
    from pathlib import Path
    import sys

    # Resolve config path relative to this file
    proj_root = Path(__file__).parent.parent.parent
    config_path = proj_root / "legacy" / "train_citys.json"

    if not config_path.exists():
        print(f"Config not found at {config_path}, skipping test.")
        sys.exit(0)

    with open(config_path, "r") as f:
        params = json.load(f)

    dataset_config = params["dataset_config"]
    ds = CityScapes(
        config_file=dataset_config["dataset_config_file"],
        ignore_lb=dataset_config["ignore_idx"],
        rootpth=dataset_config["dataset_path"],
        cropsize=dataset_config["cropsize"],
        mode="train",
    )

    print(f"Dataset loaded with {len(ds)} samples.")

    uni = []
    from tqdm import tqdm

    for img, lb in tqdm(ds, desc="Validating labels"):
        lb_np = lb.numpy()
        unique_labels = np.unique(lb_np[lb_np != ds.ignore_lb])  # Exclude ignore
        uni.extend(unique_labels.tolist())

    print("Unique training IDs found:", sorted(set(uni)))
