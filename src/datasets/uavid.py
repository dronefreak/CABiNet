#!/usr/bin/python
# -*- encoding: utf-8 -*-

import json
import os
import os.path as osp

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
    RandomHorizontalFlip,
    RandomNoise,
    RandomRotate,
    RandomScale,
)


def uavid_collate_fn(batch):
    """Collate function for UAVid that flattens patch lists into batch dimension.

    Each item in batch has 4 patches → output batch size = 4 * N
    """
    all_imgs = []
    all_labels = []
    names = []

    for item in batch:
        # item: dict with 'img_patches', 'label_patches', 'name'
        all_imgs.extend(item["img_patches"])  # List of 4 tensors
        all_labels.extend(item["label_patches"])
        names.extend([item["name"]] * 4)  # Track source

    # Stack into single batch tensors
    batched_imgs = torch.stack(all_imgs, dim=0)  # (4*N, 3, 1080, 1920)
    batched_labels = torch.stack(all_labels, dim=0)  # (4*N, 1080, 1920)

    return batched_imgs, batched_labels


class UAVid(Dataset):
    def __init__(self, config_file, ignore_lb, rootpth, cropsize, mode="train"):
        super(UAVid, self).__init__()
        self.mode = mode
        self.config_file = config_file
        self.ignore_lb = ignore_lb
        self.rootpth = rootpth
        self.cropsize = tuple(cropsize)

        assert self.mode in ("train", "val"), f"Mode {mode} not supported."
        assert osp.exists(rootpth), f"Dataset path {rootpth} does not exist!"

        # We don't actually use config_file for
        # anything because labels are already trainIds
        # But keep it for consistency
        with open(self.config_file, "r") as fr:
            labels_info = json.load(fr)
        print(f"[INFO] Loaded {len(labels_info)} classes from {config_file}")
        """Parse Image Directory."""
        self.imgs = {}
        imgnames = []
        impth = osp.join(self.rootpth, self.mode)
        folders = sorted(os.listdir(impth))
        for fd in folders:
            fdpth = osp.join(impth, fd, "Images")
            if not osp.exists(fdpth):
                continue
            im_names = [f for f in os.listdir(fdpth) if f.endswith(".png")]
            names = [os.path.splitext(fn)[0] for fn in im_names]
            paths = [osp.join(fdpth, fn) for fn in im_names]
            imgnames.extend(names)
            self.imgs.update(dict(zip(names, paths)))

        """ Parse GT Directory """
        self.labels = {}
        gtnames = []
        gtpth = osp.join(self.rootpth, self.mode)
        folders = sorted(os.listdir(gtpth))
        for fd in folders:
            fdpth = osp.join(gtpth, fd, "TrainId")
            if not osp.exists(fdpth):
                continue
            lb_names = [f for f in os.listdir(fdpth) if f.endswith(".png")]
            names = [os.path.splitext(fn)[0] for fn in lb_names]
            paths = [osp.join(fdpth, fn) for fn in lb_names]
            gtnames.extend(names)
            self.labels.update(dict(zip(names, paths)))

        self.imnames = imgnames
        self.len = len(self.imnames)

        # Safety check
        missing = set(self.imnames) - set(self.labels.keys())
        if missing:
            print(
                f"[WARN] Missing labels for {len(missing)}"
                f" images: {list(missing)[:5]}..."
            )
        self.imnames = [name for name in self.imnames if name in self.labels]
        self.len = len(self.imnames)
        """Preprocessing and Augmentation."""
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.480, 0.499, 0.457), std=(0.225, 0.208, 0.228)
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
                    RandomHorizontalFlip(p=0.2),
                    RandomRotate(degrees=(-10, 10)),
                    RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
                    RandomCrop(
                        size=self.cropsize,
                        pad_if_needed=True,
                        ignore_label=self.ignore_lb,
                    ),
                    # Photometric
                    RandomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                    RandomGamma(gamma_range=(0.8, 1.2), p=0.3),
                    RandomNoise(mode="gaussian", sigma=0.03, p=0.3),
                    # Regularization
                    RandomCutout(p=0.3, size=64),
                ]
            )
            if mode == "train"
            else None
        )

        print(f"[INFO] UAVid dataset loaded: {self.len} samples ({mode})")

    def __getitem__(self, idx):
        fn = self.imnames[idx]
        impth = self.imgs[fn]
        lbpth = self.labels[fn]

        img = Image.open(impth).convert("RGB")  # (3840, 2160, 3)
        label = Image.open(lbpth)  # (3840, 2160), mode='L', values=trainId

        w, h = img.size  # Should be 3840 x 2160
        if w != 3840 or h != 2160:
            # Resize only if needed (e.g., test set might vary)
            img = img.resize((3840, 2160), Image.BILINEAR)
            label = label.resize((3840, 2160), Image.NEAREST)

        half_w, half_h = w // 2, h // 2  # 1920, 1080

        img_patches = []
        label_patches = []

        # Define the four quadrants
        patches = [
            (0, 0, half_w, half_h),  # top-left
            (half_w, 0, w, half_h),  # top-right
            (0, half_h, half_w, h),  # bottom-left
            (half_w, half_h, w, h),  # bottom-right
        ]

        for i, (left, upper, right, lower) in enumerate(patches):
            box = (left, upper, right, lower)

            # Crop image and label
            img_patch = img.crop(box)
            label_patch = label.crop(box)

            # Apply training augmentations (optional: shared RNG for consistency?)
            if self.mode == "train" and self.trans_train is not None:
                im_lb = {"im": img_patch, "lb": label_patch}
                try:
                    im_lb = self.trans_train(im_lb)
                    img_patch, label_patch = im_lb["im"], im_lb["lb"]
                except Exception as e:
                    print(f"[WARN] Augmentation failed on patch {i} of {fn}: {e}")

            # Convert to tensor
            img_tensor = self.to_tensor(img_patch)  # (3, H, W)

            # Convert label to numpy -> long tensor
            label_np = np.array(label_patch, dtype=np.int64)
            label_tensor = torch.from_numpy(label_np).long()  # (H, W)

            img_patches.append(img_tensor)
            label_patches.append(label_tensor)

        # Return list of patches (will be flattened in collate_fn)
        return {
            "img_patches": img_patches,
            "label_patches": label_patches,
            "name": fn,
            "original_size": (h, w),
        }

    def __len__(self):
        return self.len


# === Test Block (Fixed) ===
if __name__ == "__main__":
    from pathlib import Path
    import sys

    # Resolve config path relative to this file
    proj_root = Path(__file__).parent.parent.parent
    config_path = proj_root / "legacy" / "train_uavid.json"

    if not config_path.exists():
        print(f"Config not found at {config_path}, skipping test.")
        sys.exit(0)

    with open(config_path, "r") as f:
        params = json.load(f)

    dataset_config = params["dataset_config"]
    ds = UAVid(
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
