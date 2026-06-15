"""Data augmentation transforms for semantic segmentation.

This module provides various transformation classes for augmenting image and label pairs
during training. All transforms operate on dictionaries with 'im' (image) and 'lb' (label) keys.

Transforms include:
- Geometric: RandomScale, RandomHorizontalFlip, RandomCrop, RandomRotate
- Photometric: RandomColorJitter, RandomGamma, RandomNoise, RandomGrayscale
- Regularization: RandomCutout, RandomGaussianBlur
"""

import random

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
    def __init__(
        self, scales=(1,), interp_image=Image.BILINEAR, interp_label=Image.NEAREST
    ):
        self.scales = [float(s) for s in scales]
        self.interp_image = interp_image
        self.interp_label = interp_label

    def __call__(self, im_lb):
        im = im_lb["im"]
        lb = im_lb["lb"]
        assert isinstance(im, Image.Image) and isinstance(
            lb, Image.Image
        ), f"Expected PIL images, got {type(im)}, {type(lb)}"

        scale = random.choice(self.scales)
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


class RandomCrop(object):
    def __init__(self, size, pad_if_needed=True, ignore_label=255):
        self.size = tuple(size) if hasattr(size, "__iter__") else (size, size)
        self.pad_if_needed = pad_if_needed
        self.ignore_label = ignore_label

    def __call__(self, im_lb):
        im = im_lb["im"]
        lb = im_lb["lb"]
        assert isinstance(im, Image.Image) and isinstance(lb, Image.Image)

        target_w, target_h = self.size
        w, h = im.size

        if self.pad_if_needed:
            pad_w = max(target_w - w, 0)
            pad_h = max(target_h - h, 0)
            if pad_w > 0 or pad_h > 0:
                # Pad image
                im_np = np.array(im)
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
                )
                lb = Image.fromarray(lb_np, mode="L")

        w, h = im.size
        if w < target_w or h < target_h:
            scale = max(target_w / w, target_h / h)
            new_w, new_h = int(w * scale + 1), int(h * scale + 1)
            im = im.resize((new_w, new_h), Image.BILINEAR)
            lb = lb.resize((new_w, new_h), Image.NEAREST)

        sw = random.randint(0, w - target_w) if w > target_w else 0
        sh = random.randint(0, h - target_h) if h > target_h else 0
        crop_box = (sw, sh, sw + target_w, sh + target_h)

        im_lb["im"] = im.crop(crop_box)
        im_lb["lb"] = lb.crop(crop_box)
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
        self, degrees=(-15, 15), interp_image=Image.BILINEAR, interp_label=Image.NEAREST
    ):
        self.degrees = degrees
        self.interp_image = interp_image
        self.interp_label = interp_label

    def __call__(self, im_lb):
        angle = random.uniform(*self.degrees)
        im = im_lb["im"].rotate(angle, resample=self.interp_image, expand=True)
        lb = im_lb["lb"].rotate(angle, resample=self.interp_label, expand=True)
        return {"im": im, "lb": lb}
