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
    ResizeIfLarger,
)

# Same knobs/semantics as src/datasets/uavid.py's DEFAULT_AUGMENTATION — VDD
# is also UAV-captured imagery, so the same augmentation recipe applies
# unchanged. See uavid.py's module docstring for the rationale behind each
# knob (mosaic/copy_paste unsupported, mixup's hard-label simplification,
# etc.) — identical here.
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


class VDD(Dataset):
    """VDD (Varied Drone Dataset) semantic segmentation dataset.

    Consumes the pre-converted, YOLO-style dataset layout produced by
    ``src/scripts/convert_vdd_to_yolo.py``::

        <rootpth>/
        ├── images/
        │   ├── train/   ← RGB JPEGs (uniformly 4000x3000)
        │   ├── val/
        │   └── test/
        └── masks/
            ├── train/   ← single-channel PNGs, pixel value = class ID
            ├── val/
            └── test/

    Mask pixel values are already final trainIds (0-6; Other=0 … Water=6 —
    see https://github.com/RussRobin/VDD) — no RGB colour palette or lookup
    table is needed here, unlike UAVid's raw distribution. Pixel value 255
    is reserved for genuinely unrecognized values, not a real class. Unlike
    AeroScapes, VDD ships with all three splits (train/val/test) already
    defined by the source distribution's own directory layout.

    Parameters
    ----------
    ignore_lb:
        Label value treated as "ignore" by the loss/metric (255).
    rootpth:
        Root of the *converted* dataset (i.e. ``convert_vdd_to_yolo.py``'s
        ``--dst``) — NOT the raw VDD distribution.
    cropsize:
        ``(H, W)`` crop applied during training augmentation via
        ``RandomCrop``. VDD source images are uniformly 4000x3000 (unlike
        UAVid's mixed resolutions), so ``RandomCrop`` mainly serves as a
        fixed-size training patch here — ``pad_if_needed=True`` is kept for
        robustness.
    mode:
        ``"train"``, ``"val"``, or ``"test"``.
    augmentation:
        Optional dict overriding any subset of ``DEFAULT_AUGMENTATION``
        (degrees/translate/scale/flipud/fliplr/hsv_h/hsv_s/hsv_v/mixup).

    Note on validation batching
    ----------------------------
    Like AeroScapes and unlike UAVid, VDD source images are uniformly
    4000x3000, so ``val``/``test`` mode (no crop applied) can still be
    batched with ``batch_size > 1`` via the default collate function — no
    special-casing is needed in train.py/evaluate.py for this dataset.
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

        imgnames = sorted(fn for fn in os.listdir(img_dir) if fn.endswith(".jpg"))
        for fn in imgnames:
            stem = osp.splitext(fn)[0]
            self.imgs[stem] = osp.join(img_dir, fn)
            label_path = osp.join(label_dir, f"{stem}.png")
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
        # Mean/std computed from the VDD train set (see
        # src/datasets/compute_vdd_stats.py) — distinct from ImageNet, UAVid,
        # and AeroScapes stats.
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.486, 0.487, 0.441),
                    std=(0.190, 0.178, 0.214),
                ),
            ]
        )

        # --- Training augmentation (Geometric → Photometric → Regularisation) ---
        degrees = float(self.aug["degrees"])
        scale = float(self.aug["scale"])
        self.trans_train = (
            Compose(
                [
                    # VDD's native images are 4000x3000 (~11x a 1024x1024
                    # crop) — cap the working resolution before the
                    # expensive geometric ops below, or CPU-side
                    # augmentation becomes the training bottleneck (GPU
                    # starves waiting on the DataLoader). See
                    # ResizeIfLarger's docstring.
                    ResizeIfLarger(max_size=2 * max(self.cropsize)),
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
            f"[INFO] VDD dataset loaded: {self.len} samples ({mode}) from {img_dir}"
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
# Smoke test (run: python src/datasets/vdd.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    rootpth = os.environ.get("VDD_YOLO_ROOT", "")
    if not rootpth:
        print(
            "Set VDD_YOLO_ROOT=/path/to/converted/vdd_yolo and re-run "
            "(this must be the OUTPUT of convert_vdd_to_yolo.py, not raw VDD data)."
        )
        sys.exit(0)

    for split in ("train", "val", "test"):
        try:
            ds = VDD(
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
