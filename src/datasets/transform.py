"""Data augmentation transforms for semantic segmentation.

This module provides various transformation classes for augmenting image and label pairs
during training. All transforms operate on dictionaries with 'im' (image) and 'lb' (label) keys.

Transforms include:
- Geometric: RandomScale, RandomHorizontalFlip, RandomCrop, RandomRotate
- Photometric: RandomColorJitter, RandomGamma, RandomNoise, RandomGrayscale
- Regularization: RandomCutout, RandomGaussianBlur
"""

import random  # nosec B311 — used for data augmentation, not security/cryptographic purposes
from typing import Any

from PIL import Image, ImageEnhance
import numpy as np


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, im_lb):
        for t in self.transforms:
            im_lb = t(im_lb)
        return im_lb


class RandomScale(object):
    """Random resize, either from a discrete list of scale factors
    (default) or a continuous range — pass ``continuous=True`` with
    ``scales=(low, high)`` to sample uniformly from ``[low, high]``,
    matching Ultralytics' ``scale`` augmentation (``scale=X`` means the
    continuous range ``[1-X, 1+X]``)."""

    def __init__(
        self,
        scales=(1,),
        continuous=False,
        interp_image=Image.BILINEAR,
        interp_label=Image.NEAREST,
    ):
        self.continuous = continuous
        if continuous:
            lo, hi = scales
            self.scale_range = (float(lo), float(hi))
        else:
            self.scales = [float(s) for s in scales]
        self.interp_image = interp_image
        self.interp_label = interp_label

    def __call__(self, im_lb):
        im = im_lb["im"]
        lb = im_lb["lb"]
        if not (isinstance(im, Image.Image) and isinstance(lb, Image.Image)):
            raise TypeError(f"Expected PIL images, got {type(im)}, {type(lb)}")

        if self.continuous:
            scale = random.uniform(*self.scale_range)  # nosec B311
        else:
            scale = random.choice(self.scales)  # nosec B311
        W, H = im.size
        w = int(round(W * scale))
        h = int(round(H * scale))
        return {
            "im": im.resize((w, h), self.interp_image),
            "lb": lb.resize((w, h), self.interp_label),
        }


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        im = im_lb["im"].transpose(Image.FLIP_LEFT_RIGHT)
        lb = im_lb["lb"].transpose(Image.FLIP_LEFT_RIGHT)
        return {"im": im, "lb": lb}


class RandomVerticalFlip(object):
    """Vertical flip — matches Ultralytics' ``flipud`` augmentation (valid
    for top-down aerial imagery, unlike ground-level datasets)."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        im = im_lb["im"].transpose(Image.FLIP_TOP_BOTTOM)
        lb = im_lb["lb"].transpose(Image.FLIP_TOP_BOTTOM)
        return {"im": im, "lb": lb}


class RandomTranslate(object):
    """Random translation by up to ``translate`` fraction of image size in
    each axis, matching Ultralytics' ``translate`` augmentation."""

    def __init__(self, translate=0.05, ignore_label=255):
        self.translate = translate
        self.ignore_label = ignore_label

    def __call__(self, im_lb):
        im = im_lb["im"]
        lb = im_lb["lb"]
        w, h = im.size
        dx = random.uniform(-self.translate, self.translate) * w  # nosec B311
        dy = random.uniform(-self.translate, self.translate) * h  # nosec B311
        im = im.transform(
            im.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), resample=Image.BILINEAR
        )
        lb = lb.transform(
            lb.size,
            Image.AFFINE,
            (1, 0, dx, 0, 1, dy),
            resample=Image.NEAREST,
            fillcolor=self.ignore_label,
        )
        return {"im": im, "lb": lb}


class RandomCrop(object):
    def __init__(self, size, pad_if_needed=True, ignore_label=255):
        self.size = tuple(size) if hasattr(size, "__iter__") else (size, size)
        self.pad_if_needed = pad_if_needed
        self.ignore_label = ignore_label

    def __call__(self, im_lb):
        im = im_lb["im"]
        lb = im_lb["lb"]
        if not (isinstance(im, Image.Image) and isinstance(lb, Image.Image)):
            raise TypeError(f"Expected PIL images, got {type(im)}, {type(lb)}")

        target_w, target_h = self.size
        w, h = im.size

        if self.pad_if_needed:
            pad_w = max(target_w - w, 0)
            pad_h = max(target_h - h, 0)
            if pad_w > 0 or pad_h > 0:
                # Pad image — pad_width type varies by ndim, annotated as Any
                im_np = np.array(im)
                pad_width: Any
                if len(im_np.shape) == 3:
                    pad_width = ((0, pad_h), (0, pad_w), (0, 0))
                else:
                    pad_width = ((0, pad_h), (0, pad_w))
                im_np = np.pad(im_np, pad_width, mode="reflect")
                im = Image.fromarray(im_np)

                # Pad label
                lb_np = np.array(lb)
                lb_np = np.pad(
                    lb_np, ((0, pad_h), (0, pad_w)), constant_values=self.ignore_label
                ).astype(np.uint8)
                lb = Image.fromarray(lb_np)

        w, h = im.size
        if w < target_w or h < target_h:
            scale = max(target_w / w, target_h / h)
            new_w, new_h = int(w * scale + 1), int(h * scale + 1)
            im = im.resize((new_w, new_h), Image.BILINEAR)
            lb = lb.resize((new_w, new_h), Image.NEAREST)

        sw = random.randint(0, w - target_w) if w > target_w else 0  # nosec B311
        sh = random.randint(0, h - target_h) if h > target_h else 0  # nosec B311
        crop_box = (sw, sh, sw + target_w, sh + target_h)

        im_lb["im"] = im.crop(crop_box)
        im_lb["lb"] = lb.crop(crop_box)
        return im_lb


class RandomHSV(object):
    """Multiplicative saturation/value + additive hue jitter in HSV colour
    space, matching Ultralytics' ``RandomHSV`` augmentation formula exactly:
    ``hue = (hue + gain_h * full_circle) % full_circle`` (additive, wraps),
    ``sat = clip(sat * (1 + gain_s), 0, max)``, ``val = clip(val * (1 +
    gain_v), 0, max)``, with each gain drawn as ``uniform(-1, 1) * hgain``
    (etc). Reimplemented via PIL's HSV conversion rather than OpenCV — PIL's
    H channel spans 0-255 for the full hue circle (vs OpenCV's 0-179), so
    the hue shift is scaled to 255 instead of 180 to preserve the same
    *fraction of the circle* shifted.
    """

    def __init__(self, hgain=0.015, sgain=0.4, vgain=0.3):
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, im_lb):
        if self.hgain or self.sgain or self.vgain:
            im = im_lb["im"]
            hsv = np.array(im.convert("HSV"), dtype=np.int16)

            r_h = random.uniform(-1, 1) * self.hgain  # nosec B311
            r_s = random.uniform(-1, 1) * self.sgain  # nosec B311
            r_v = random.uniform(-1, 1) * self.vgain  # nosec B311

            hsv[..., 0] = (hsv[..., 0] + round(r_h * 255)) % 255
            hsv[..., 1] = np.clip(hsv[..., 1] * (r_s + 1), 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] * (r_v + 1), 0, 255)
            hsv = hsv.astype(np.uint8)

            # Image.fromarray(..., mode="HSV") is deprecated (removed in
            # Pillow 13) — Image.merge is the non-deprecated equivalent.
            im_hsv = Image.merge(
                "HSV",
                [Image.fromarray(hsv[..., c]) for c in range(3)],
            )
            im_lb["im"] = im_hsv.convert("RGB")
        return im_lb


class RandomColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None):
        self.brightness = self._check(brightness)
        self.contrast = self._check(contrast)
        self.saturation = self._check(saturation)

    @staticmethod
    def _check(v):
        return None if v is None else [max(1 - v, 0), 1 + v]

    def __call__(self, im_lb):
        im = im_lb["im"]
        if self.brightness:
            r = random.uniform(*self.brightness)
            im = ImageEnhance.Brightness(im).enhance(r)
        if self.contrast:
            r = random.uniform(*self.contrast)
            im = ImageEnhance.Contrast(im).enhance(r)
        if self.saturation:
            r = random.uniform(*self.saturation)
            im = ImageEnhance.Color(im).enhance(r)
        im_lb["im"] = im
        return im_lb


class RandomCutout:
    def __init__(self, p=0.5, size=64):
        self.p = p
        self.size = size

    def __call__(self, im_lb):
        if random.random() < self.p:
            im = np.array(im_lb["im"])
            h, w, _ = im.shape
            y = random.randint(0, h - self.size)
            x = random.randint(0, w - self.size)
            im[y : y + self.size, x : x + self.size, :] = 0
            im_lb["im"] = Image.fromarray(im)
        return im_lb


class RandomGaussianBlur:
    def __init__(self, p=0.5, radius=(0.1, 2.0)):
        self.p = p
        self.radius = radius

    def __call__(self, im_lb):
        if random.random() < self.p:
            from PIL import ImageFilter

            r = random.uniform(*self.radius)
            im_lb["im"] = im_lb["im"].filter(ImageFilter.GaussianBlur(radius=r))
        return im_lb


class RandomGrayscale:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, im_lb):
        if random.random() < self.p:
            im = im_lb["im"].convert("L")  # convert to grayscale
            im = im.convert("RGB")  # back to 3 channels
            im_lb["im"] = im
        return im_lb


class RandomGamma:
    def __init__(self, gamma_range=(0.7, 1.5), p=0.5):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, im_lb):
        if random.random() < self.p:
            gamma = random.uniform(*self.gamma_range)
            im = np.array(im_lb["im"]).astype(np.float32) / 255.0
            im = np.clip(im**gamma, 0, 1)  # gamma correction
            im = (im * 255).astype(np.uint8)
            im_lb["im"] = Image.fromarray(im)
        return im_lb


class RandomNoise:
    def __init__(self, mode="gaussian", sigma=0.05, p=0.5):
        """
        mode: 'gaussian' or 'poisson'
        sigma: std for Gaussian (fraction of 255)
        p: probability of applying
        """
        self.mode = mode
        self.sigma = sigma
        self.p = p

    def __call__(self, im_lb):
        if random.random() < self.p:
            arr = np.array(im_lb["im"]).astype(np.float32)

            if self.mode == "gaussian":
                noise = np.random.normal(0, self.sigma * 255, arr.shape)
                arr = arr + noise
            elif self.mode == "poisson":
                vals = 2 ** np.ceil(np.log2(len(np.unique(arr))))
                arr = np.random.poisson(arr * vals) / float(vals)

            arr = np.clip(arr, 0, 255).astype(np.uint8)
            im_lb["im"] = Image.fromarray(arr)
        return im_lb


class RandomRotate(object):
    """Small random rotation to simulate UAV yaw changes."""

    def __init__(
        self,
        degrees=(-15, 15),
        interp_image=Image.BILINEAR,
        interp_label=Image.NEAREST,
        ignore_label=255,
    ):
        self.degrees = degrees
        self.interp_image = interp_image
        self.interp_label = interp_label
        self.ignore_label = ignore_label

    def __call__(self, im_lb):
        angle = random.uniform(*self.degrees)
        im = im_lb["im"].rotate(angle, resample=self.interp_image, expand=True)
        lb = im_lb["lb"].rotate(
            angle, resample=self.interp_label, expand=True, fillcolor=self.ignore_label
        )
        return {"im": im, "lb": lb}
