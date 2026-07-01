#!/usr/bin/python
# -*- encoding: utf-8 -*-

import json
import os
import os.path as osp
from typing import Any, Dict, List, Optional, Tuple

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


def uavid_collate_fn(batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function for UAVid that flattens patch lists into batch dimension.

    Each item in batch has 4 patches → output batch size = 4 * N
    """
    all_imgs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for item in batch:
        all_imgs.extend(item["img_patches"])
        all_labels.extend(item["label_patches"])

    batched_imgs = torch.stack(all_imgs, dim=0)  # (4*N, 3, H, W)
    batched_labels = torch.stack(all_labels, dim=0)  # (4*N, H, W)

    return batched_imgs, batched_labels


class UAVid(Dataset):
    """UAVid aerial semantic segmentation dataset.

    Dataset layout (as distributed by UAVid)::

        uavid_train/          ← *rootpth*
        ├── seq1/
        │   ├── Images/       ← RGB input images  (*.png)
        │   └── Labels/       ← RGB colour-coded masks (*.png)
        ├── seq2/
        │   └── …
        └── seq16/            ← example validation sequence

    The ``Labels/`` directory contains **3-channel RGB colour-coded masks**
    where each pixel colour encodes a semantic class (defined in
    ``UAVid_info.json``).  This loader converts them to single-channel
    trainId masks on-the-fly via a pre-built lookup table.

    Class mapping (n_classes=8)
    ---------------------------
    Clutter is included in the loss (trainId=0) but should be excluded when
    computing the mIoU metric (``ignoreInEval=True``).

    Parameters
    ----------
    config_file:
        Path to ``UAVid_info.json`` (colour palette + class metadata).
    ignore_lb:
        Label value used for pixels not matching any known colour
        (default 255, passed in from config).
    rootpth:
        Root directory that directly contains the sequence folders
        (``uavid_train/`` or equivalent).
    cropsize:
        ``(H, W)`` crop applied during training augmentation.
    mode:
        ``"train"`` or ``"val"``.
    val_seqs:
        Sequence folder names that belong to the validation split.
        All other sequences found in *rootpth* are used for training.
        Defaults to ``["seq16"]``.
    """

    def __init__(
        self,
        config_file: str,
        ignore_lb: int,
        rootpth: str,
        cropsize: Tuple[int, int],
        mode: str = "train",
        val_seqs: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.ignore_lb = ignore_lb
        self.rootpth = rootpth
        self.cropsize = tuple(cropsize)
        self.val_seqs: set = set(val_seqs if val_seqs is not None else ["seq16"])

        if self.mode not in ("train", "val"):
            raise ValueError(f"Mode '{mode}' not supported. Choose 'train' or 'val'.")
        if not osp.exists(rootpth):
            raise FileNotFoundError(f"Dataset root does not exist: {rootpth}")

        # --- Build RGB → trainId lookup table from colour palette ----------
        with open(config_file) as f:
            labels_info: List[Dict[str, Any]] = json.load(f)
        self._trainid_lut = self._build_trainid_lut(labels_info, ignore_lb)
        print(f"[INFO] Loaded {len(labels_info)} classes from {config_file}")

        # --- Discover sequences for this split ------------------------------
        all_seqs = sorted(
            s for s in os.listdir(rootpth) if osp.isdir(osp.join(rootpth, s))
        )
        if mode == "train":
            use_seqs = [s for s in all_seqs if s not in self.val_seqs]
        else:
            use_seqs = [s for s in all_seqs if s in self.val_seqs]

        if not use_seqs:
            raise FileNotFoundError(
                f"No sequences found for split='{mode}' in {rootpth}. "
                f"val_seqs={self.val_seqs}, available={all_seqs}"
            )

        # --- Load image and label paths; key = "{seq}/{stem}" ---------------
        # Using "{seq}/{stem}" prevents key collisions when the same filename
        # (e.g. "000001.png") appears in multiple sequence folders.
        self.imgs: Dict[str, str] = {}
        self.labels: Dict[str, str] = {}
        imgnames: List[str] = []

        for seq in use_seqs:
            img_dir = osp.join(rootpth, seq, "Images")
            label_dir = osp.join(rootpth, seq, "Labels")
            if not osp.exists(img_dir):
                print(f"[WARN] Images/ not found for sequence {seq}, skipping.")
                continue
            for fn in sorted(os.listdir(img_dir)):
                if not fn.endswith(".png"):
                    continue
                stem = osp.splitext(fn)[0]
                key = f"{seq}/{stem}"
                self.imgs[key] = osp.join(img_dir, fn)
                label_path = osp.join(label_dir, fn)
                if osp.exists(label_path):
                    self.labels[key] = label_path
                imgnames.append(key)

        # Drop any image that has no matching label
        missing = set(imgnames) - set(self.labels.keys())
        if missing:
            print(
                f"[WARN] {len(missing)} image(s) have no Labels/ mask "
                f"and will be skipped: {sorted(missing)[:5]}…"
            )
        self.imnames = [k for k in imgnames if k in self.labels]
        self.len = len(self.imnames)

        # --- Image normalisation -------------------------------------------------
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.480, 0.499, 0.457),
                    std=(0.225, 0.208, 0.228),
                ),
            ]
        )

        # --- Training augmentation (Geometric → Photometric → Regularisation) ---
        self.trans_train = (
            Compose(
                [
                    RandomHorizontalFlip(p=0.2),
                    RandomRotate(degrees=(-10, 10), ignore_label=self.ignore_lb),
                    RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
                    RandomCrop(
                        size=self.cropsize,
                        pad_if_needed=True,
                        ignore_label=self.ignore_lb,
                    ),
                    RandomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                    RandomGamma(gamma_range=(0.8, 1.2), p=0.3),
                    RandomNoise(mode="gaussian", sigma=0.03, p=0.3),
                    RandomCutout(p=0.3, size=64),
                ]
            )
            if mode == "train"
            else None
        )

        print(
            f"[INFO] UAVid dataset loaded: {self.len} samples ({mode}) "
            f"from {len(use_seqs)} sequence(s)"
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_trainid_lut(
        labels_info: List[Dict[str, Any]], ignore_lb: int
    ) -> np.ndarray:
        """Build a (256, 256, 256) uint8 LUT mapping RGB colour → trainId.

        Known colours are mapped to their ``trainId`` value (0-7 for UAVid).
        Unknown colours default to *ignore_lb* (255).

        This is distinct from the YOLO LUT in ``convert_uavid_to_yolo.py``
        which maps Clutter → 255; here Clutter maps to its trainId (0) so
        that it is included in the cross-entropy loss while still being
        excluded from the mIoU metric during evaluation.
        """
        lut = np.full((256, 256, 256), ignore_lb, dtype=np.uint8)
        for cls in labels_info:
            r, g, b = cls["color"]
            lut[r, g, b] = cls["trainId"]
        return lut

    def _rgb_label_to_trainid(self, label_rgb: Image.Image) -> Image.Image:
        """Convert a 3-channel RGB label PIL image to a single-channel trainId image."""
        arr = np.array(label_rgb, dtype=np.uint8)  # (H, W, 3)
        trainid = self._trainid_lut[arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]]  # (H, W)
        return Image.fromarray(trainid)  # mode='L', uint8

    # -----------------------------------------------------------------------
    # Dataset interface
    # -----------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        key = self.imnames[idx]
        img = Image.open(self.imgs[key]).convert("RGB")  # (W, H) = 3840×2160
        label_rgb = Image.open(self.labels[key]).convert("RGB")  # RGB colour mask

        # Convert RGB colour mask → single-channel trainId mask.
        # This must happen BEFORE augmentation so that the pipeline receives
        # a mode-L image (values 0-7 or ignore_lb), not a 3-channel array.
        label = self._rgb_label_to_trainid(label_rgb)  # mode='L'

        w, h = img.size  # 3840, 2160 for standard UAVid
        if w != 3840 or h != 2160:
            img = img.resize((3840, 2160), Image.BILINEAR)
            label = label.resize((3840, 2160), Image.NEAREST)
            w, h = 3840, 2160

        half_w, half_h = w // 2, h // 2  # 1920, 1080

        img_patches: List[torch.Tensor] = []
        label_patches: List[torch.Tensor] = []

        for left, upper, right, lower in [
            (0, 0, half_w, half_h),  # top-left
            (half_w, 0, w, half_h),  # top-right
            (0, half_h, half_w, h),  # bottom-left
            (half_w, half_h, w, h),  # bottom-right
        ]:
            img_patch = img.crop((left, upper, right, lower))
            label_patch = label.crop((left, upper, right, lower))

            if self.mode == "train" and self.trans_train is not None:
                im_lb = {"im": img_patch, "lb": label_patch}
                try:
                    im_lb = self.trans_train(im_lb)
                    img_patch, label_patch = im_lb["im"], im_lb["lb"]
                except Exception as exc:
                    print(f"[WARN] Augmentation failed on {key}: {exc}")

            img_patches.append(self.to_tensor(img_patch))
            label_np = np.array(label_patch, dtype=np.int64)
            label_patches.append(torch.from_numpy(label_np).long())

        return {
            "img_patches": img_patches,
            "label_patches": label_patches,
            "name": key,
            "original_size": (h, w),
        }

    def __len__(self) -> int:
        return self.len


# ---------------------------------------------------------------------------
# Smoke test (run: python src/datasets/uavid.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    rootpth = os.environ.get("UAVID_TRAIN_ROOT", "")
    config_file = "configs/UAVid_info.json"
    if not rootpth:
        print("Set UAVID_TRAIN_ROOT=/path/to/uavid_train and re-run.")
        sys.exit(0)

    for split in ("train", "val"):
        ds = UAVid(
            config_file=config_file,
            ignore_lb=255,
            rootpth=rootpth,
            cropsize=(1024, 1024),
            mode=split,
            val_seqs=["seq16"],
        )
        print(f"{split}: {len(ds)} samples")
        if len(ds) == 0:
            continue
        item = ds[0]
        for lb in item["label_patches"]:
            unique = torch.unique(lb[lb != 255])
            print(f"  unique trainIds in first item: {sorted(unique.tolist())}")
        break
