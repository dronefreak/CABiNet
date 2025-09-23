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
    ColorJitter,
    Compose,
    HorizontalFlip,
    RandomCrop,
    RandomScale,
)


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
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

        self.trans_train = (
            Compose(
                [
                    ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                    HorizontalFlip(p=0.5),
                    RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
                    RandomCrop(cropsize, pad_if_needed=True, ignore_label=ignore_lb),
                ]
            )
            if mode == "train"
            else None
        )

        print(f"[INFO] UAVid dataset loaded: {self.len} samples ({mode})")

    def __getitem__(self, idx):
        name = self.imnames[idx]
        try:
            img_path = self.imgs[name]
            lb_path = self.labels[name]

            img = Image.open(img_path).convert("RGB")
            label_img = Image.open(lb_path)  # Grayscale, pixel value = trainId

            # Apply augmentations
            if self.trans_train is not None:
                im_lb = {"im": img, "lb": label_img}
                im_lb = self.trans_train(im_lb)
                img, label_img = im_lb["im"], im_lb["lb"]

            # To tensor
            img = self.to_tensor(img)
            label = np.array(label_img, dtype=np.int64)
            label = torch.from_numpy(label).long()

            return img, label

        except Exception as e:
            print(f"[ERROR] Failed to load {name}: {e}")
            return self.__getitem__(np.random.randint(0, len(self)))

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
