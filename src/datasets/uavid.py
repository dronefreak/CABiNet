#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import os.path as osp
from typing import Any, Dict, Optional, Tuple
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
    RandomHorizontalFlip,
    RandomHSV,
    RandomNoise,
    RandomRotate,
    RandomScale,
    RandomTranslate,
    RandomVerticalFlip,
)

# Mirrors configs/yolo/uavid_train.yaml / configs/train_yolo.yaml's
# augmentation block — CABiNet and the YOLO26 pipeline respond to the same
# knobs with (as close as architecturally possible) the same meaning.
# shear/perspective are omitted: both are disabled (0.0) on the YOLO side
# and CABiNet has no such transforms, so they already "match" by absence.
# mosaic/copy_paste are NOT implemented (see class docstring).
DEFAULT_AUGMENTATION: Dict[str, float] = {
    "degrees": 10.0,
    "translate": 0.05,
    "scale": 0.3,
    "flipud": 0.2,
    "fliplr": 0.5,
    "hsv_h": 0.01,
    "hsv_s": 0.4,
    "hsv_v": 0.3,
    "mixup": 0.1,
}


class UAVid(Dataset):
    """UAVid aerial semantic segmentation dataset.

    Consumes the pre-converted, YOLO-style dataset layout produced by
    ``src/scripts/convert_uavid_to_yolo.py`` — the SAME converted directory
    used by the Ultralytics YOLO26 semantic-segmentation pipeline::

        <rootpth>/
        ├── images/
        │   ├── train/   ← RGB PNGs
        │   ├── val/
        │   └── test/
        └── masks/
            ├── train/   ← single-channel PNGs, pixel value = class ID
            ├── val/
            └── test/

    Mask pixel values are already final trainIds (0-7; Clutter=0 … MovingCar=7,
    per the original UAVid paper all 8 classes are valid and none are
    ignored) — no RGB colour palette or lookup table is needed here, since
    the conversion step already did that once, up front. Pixel value 255 is
    reserved for genuinely unrecognized colours encountered during
    conversion (corrupted/anti-aliased source data), not a real class.

    Parameters
    ----------
    ignore_lb:
        Label value treated as "ignore" by the loss/metric (255).
    rootpth:
        Root of the *converted* dataset (i.e. ``convert_uavid_to_yolo.py``'s
        ``--dst``) — NOT the raw UAVid distribution.
    cropsize:
        ``(H, W)`` crop applied during training augmentation via
        ``RandomCrop``. UAVid source images are not uniform resolution
        (both 3840x2160 and 4096x2160 occur in practice), so training relies
        on ``RandomCrop(pad_if_needed=True)`` to handle arbitrary input size
        directly rather than forcing a canonical resolution up front.
    mode:
        ``"train"``, ``"val"``, or ``"test"``.
    augmentation:
        Optional dict overriding any subset of ``DEFAULT_AUGMENTATION``
        (degrees/translate/scale/flipud/fliplr/hsv_h/hsv_s/hsv_v/mixup),
        mirroring the YOLO26 pipeline's ``augmentation:`` config block
        (``configs/train_yolo.yaml``) so both pipelines can be tuned the
        same way. ``mosaic``/``copy_paste`` are intentionally not
        supported: both are multi-image augmentations requiring dataset-
        level access to other samples, and ``copy_paste`` in particular has
        no well-defined translation to pure semantic segmentation (no
        instance boundaries to paste). ``mixup`` *is* implemented, but with
        a necessary simplification: two images are alpha-blended
        continuously (``Beta(32, 32)``, matching Ultralytics exactly), but
        the two label maps cannot be blended the same way — there's no
        meaningful "average" of two class-ID masks — so the hard label is
        taken from whichever image contributed the larger blend weight.

    Note on validation batching
    ----------------------------
    In ``val``/``test`` mode no crop is applied (full-resolution images are
    evaluated via sliding-window inference), so a `DataLoader` batching more
    than one sample at a time will fail to stack mismatched image sizes.
    Callers must use ``batch_size=1`` for non-train modes.
    """

    def __init__(
        self,
        ignore_lb: int,
        rootpth: str,
        cropsize: Tuple[int, int],
        mode: str = "train",
        augmentation: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.ignore_lb = ignore_lb
        self.rootpth = rootpth
        self.cropsize = tuple(cropsize)
        self.aug = {**DEFAULT_AUGMENTATION, **(augmentation or {})}

        if self.mode not in ("train", "val", "test"):
            raise ValueError(
                f"Mode '{mode}' not supported. Choose 'train', 'val', or 'test'."
            )
        if not osp.exists(rootpth):
            raise FileNotFoundError(f"Dataset root does not exist: {rootpth}")

        img_dir = osp.join(rootpth, "images", mode)
        label_dir = osp.join(rootpth, "masks", mode)
        if not osp.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not osp.exists(label_dir):
            raise FileNotFoundError(f"Mask directory not found: {label_dir}")

        # --- Load image and label paths ------------------------------------
        self.imnames = []
        self.imgs = {}
        self.labels = {}

        imgnames = sorted(fn for fn in os.listdir(img_dir) if fn.endswith(".png"))
        for fn in imgnames:
            stem = osp.splitext(fn)[0]
            self.imgs[stem] = osp.join(img_dir, fn)
            label_path = osp.join(label_dir, fn)
            if osp.exists(label_path):
                self.labels[stem] = label_path
            self.imnames.append(stem)

        # Drop any image that has no matching mask
        missing = [name for name in self.imnames if name not in self.labels]
        if missing:
            warnings.warn(
                f"{len(missing)} image(s) have no matching mask in {label_dir} "
                f"and will be skipped: {sorted(missing)[:5]}…"
            )
        self.imnames = [name for name in self.imnames if name in self.labels]

        if len(self.imnames) == 0:
            raise RuntimeError(
                f"No valid image-mask pairs found for mode='{mode}' in {rootpth}."
            )

        self.len = len(self.imnames)

        # --- Image normalisation --------------------------------------------------
        # Mean/std computed from the UAVid train set (see
        # src/datasets/compute_uavid_stats.py) — distinct from ImageNet stats.
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
        # Geometric/photometric parameters mirror the YOLO26 pipeline's
        # augmentation config (see DEFAULT_AUGMENTATION above); RandomGamma/
        # RandomNoise/RandomCutout are CABiNet-specific extras layered on
        # top, not part of the YOLO26 alignment.
        degrees = float(self.aug["degrees"])
        scale = float(self.aug["scale"])
        self.trans_train = (
            Compose(
                [
                    RandomHorizontalFlip(p=float(self.aug["fliplr"])),
                    RandomVerticalFlip(p=float(self.aug["flipud"])),
                    RandomTranslate(
                        translate=float(self.aug["translate"]),
                        ignore_label=self.ignore_lb,
                    ),
                    RandomRotate(
                        degrees=(-degrees, degrees), ignore_label=self.ignore_lb
                    ),
                    RandomScale((1.0 - scale, 1.0 + scale), continuous=True),
                    RandomCrop(
                        size=self.cropsize,
                        pad_if_needed=True,
                        ignore_label=self.ignore_lb,
                    ),
                    RandomHSV(
                        hgain=float(self.aug["hsv_h"]),
                        sgain=float(self.aug["hsv_s"]),
                        vgain=float(self.aug["hsv_v"]),
                    ),
                    RandomColorJitter(contrast=0.5),
                    RandomGamma(gamma_range=(0.8, 1.2), p=0.3),
                    RandomNoise(mode="gaussian", sigma=0.03, p=0.3),
                    RandomCutout(p=0.3, size=64),
                ]
            )
            if mode == "train"
            else None
        )
        self.mixup_p = float(self.aug["mixup"]) if mode == "train" else 0.0

        print(
            f"[INFO] UAVid dataset loaded: {self.len} samples ({mode}) from {img_dir}"
        )

    def _load_one(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load, augment, and tensorize a single sample (no MixUp)."""
        stem = self.imnames[idx]
        img = Image.open(self.imgs[stem]).convert("RGB")
        label = Image.open(self.labels[stem])
        if label.mode != "L":
            label = label.convert("L")

        if self.mode == "train" and self.trans_train is not None:
            im_lb = self.trans_train({"im": img, "lb": label})
            img, label = im_lb["im"], im_lb["lb"]

        img_t = self.to_tensor(img)
        label_np = np.array(label, dtype=np.int64)  # (H, W), already final class IDs
        label_t = torch.from_numpy(label_np).long()
        return img_t, label_t

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, label = self._load_one(idx)

        if (
            self.mode == "train"
            and self.mixup_p > 0
            and np.random.random() < self.mixup_p
        ):
            other_idx = int(np.random.randint(0, self.len))
            img2, label2 = self._load_one(other_idx)
            # Both samples went through the same RandomCrop, so shapes match.
            r = float(np.random.beta(32.0, 32.0))  # matches Ultralytics' MixUp exactly
            img = img * r + img2 * (1.0 - r)
            # Class-ID masks can't be blended the same way (no meaningful
            # "average" of two class indices) — take the label from
            # whichever image contributed the larger share of the blend.
            label = label if r >= 0.5 else label2

        return img, label

    def __len__(self) -> int:
        return self.len


# ---------------------------------------------------------------------------
# Smoke test (run: python src/datasets/uavid.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    rootpth = os.environ.get("UAVID_YOLO_ROOT", "")
    if not rootpth:
        print(
            "Set UAVID_YOLO_ROOT=/path/to/converted/uavid_yolo and re-run "
            "(this must be the OUTPUT of convert_uavid_to_yolo.py, not raw UAVid data)."
        )
        sys.exit(0)

    for split in ("train", "val", "test"):
        try:
            ds = UAVid(
                ignore_lb=255,
                rootpth=rootpth,
                cropsize=(1024, 1024),
                mode=split,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"{split}: skipped ({exc})")
            continue
        print(f"{split}: {len(ds)} samples")
        img, lb = ds[0]
        unique = torch.unique(lb[lb != 255])
        print(f"  img shape: {tuple(img.shape)}, label shape: {tuple(lb.shape)}")
        print(f"  unique class IDs in first item: {sorted(unique.tolist())}")
