#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""Safe and thread-aware image+label transformations for semantic segmentation.

Ensures synchronized augmentation between image and label.
"""

import random

from PIL import Image
import PIL.ImageEnhance as ImageEnhance
import numpy as np
import torch

# Try to use PyTorch's generator for better multi-worker randomness
try:
    _torch_has_generator = hasattr(torch, "Generator")
except:
    _torch_has_generator = False


def get_random_state():
    """Use PyTorch's RNG if available for DataLoader worker seeding."""
    if _torch_has_generator:
        try:
            return torch.randint(0, 2**32, (), dtype=torch.int64).item()
        except:
            pass
    return random.randint(0, 2**32 - 1)


class Compose(object):
    """Sequentially applies a list of transforms that accept dict{im, lb}."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, im_lb):
        for t in self.transforms:
            try:
                im_lb = t(im_lb)
            except Exception as e:
                raise RuntimeError(f"Transform {t.__class__.__name__} failed: {e}")
        return im_lb

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class RandomCrop(object):
    """Crop image and label at random position.

    If image is smaller than target size, pads with zero (or ignore value for label).
    """

    def __init__(self, size, pad_if_needed=False, ignore_label=255):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size  # (H, W)
        self.pad_if_needed = pad_if_needed
        self.ignore_label = ignore_label

    def __call__(self, im_lb):
        im = im_lb["im"]
        lb = im_lb["lb"]
        assert (
            im.size == lb.size
        ), f"Image and label size mismatch: {im.size} vs {lb.size}"

        target_w, target_h = self.size
        w, h = im.size

        # Pad if needed
        if self.pad_if_needed:
            pad_w = max(target_w - w, 0)
            pad_h = max(target_h - h, 0)
            if pad_w > 0 or pad_h > 0:
                im = Image.fromarray(
                    np.pad(
                        np.array(im), ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect"
                    ),
                    mode="RGB",
                )
                lb_np = np.array(lb, dtype=np.int64)
                lb_np = np.pad(
                    lb_np,
                    ((0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=self.ignore_label,
                )
                lb = Image.fromarray(lb_np, mode="L")
                w, h = im.size

        # If still too small even after padding, resize up
        if w < target_w or h < target_h:
            scale = max(target_w / w, target_h / h)
            new_w = int(w * scale + 1)
            new_h = int(h * scale + 1)
            im = im.resize((new_w, new_h), Image.BILINEAR)
            lb = lb.resize((new_w, new_h), Image.NEAREST)
            w, h = im.size

        # Random crop
        sw = random.randint(0, w - target_w) if w > target_w else 0
        sh = random.randint(0, h - target_h) if h > target_h else 0
        crop_box = (sw, sh, sw + target_w, sh + target_h)

        im = im.crop(crop_box)
        lb = lb.crop(crop_box)

        return {"im": im, "lb": lb}

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size},"

    "pad_if_needed={self.pad_if_needed})"


class HorizontalFlip(object):
    """Random horizontal flip with user-defined probability."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        im = im_lb["im"].transpose(Image.FLIP_LEFT_RIGHT)
        lb = im_lb["lb"].transpose(Image.FLIP_LEFT_RIGHT)
        return {"im": im, "lb": lb}

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


class RandomScale(object):
    """Randomly scales image and label using one of the provided scales.

    Uses consistent random selection per call.
    """

    def __init__(
        self, scales=(1.0,), interp_image=Image.BILINEAR, interp_label=Image.NEAREST
    ):
        if isinstance(scales, (float, int)):
            scales = [float(scales)]
        self.scales = [float(s) for s in scales]
        self.interp_image = interp_image
        self.interp_label = interp_label

    def __call__(self, im_lb):
        im = im_lb["im"]
        lb = im_lb["lb"]
        scale = random.choice(self.scales)
        W, H = im.size
        w, h = int(round(W * scale)), int(round(H * scale))
        if w != W or h != H:  # Avoid unnecessary resize
            im = im.resize((w, h), self.interp_image)
            lb = lb.resize((w, h), self.interp_label)
        return {"im": im, "lb": lb}

    def __repr__(self):
        return f"{self.__class__.__name__}(scales={self.scales})"


class ColorJitter(object):
    """Random brightness, contrast, saturation jitter.

    Safe: only applies if parameter is not None.
    """

    def __init__(self, brightness=None, contrast=None, saturation=None):
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")

    @staticmethod
    def _check_input(value, name):
        if value is None:
            return None
        if value < 0:
            raise ValueError(f"If {name} is specified, must be >= 0. Got {value}")
        return [max(1 - value, 0), 1 + value]

    def __call__(self, im_lb):
        im = im_lb["im"]

        # Apply in random order for realism
        transforms = []
        if self.brightness is not None:
            r = random.uniform(self.brightness[0], self.brightness[1])
            transforms.append(lambda img: ImageEnhance.Brightness(img).enhance(r))

        if self.contrast is not None:
            r = random.uniform(self.contrast[0], self.contrast[1])
            transforms.append(lambda img: ImageEnhance.Contrast(img).enhance(r))

        if self.saturation is not None:
            r = random.uniform(self.saturation[0], self.saturation[1])
            transforms.append(lambda img: ImageEnhance.Color(img).enhance(r))

        # Optional: Shuffle transform order
        random.shuffle(transforms)
        for t in transforms:
            im = t(im)

        return {"im": im, "lb": im_lb["lb"]}

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}, "
            f"contrast={self.contrast}, "
            f"saturation={self.saturation})"
        )


class MultiScale(object):
    """
    DEPRECATED FOR SEGMENTATION: Returns multiple scaled versions of image only.
    Does NOT support labels.
    Consider removing or upgrading to multi-scale inference wrapper.
    """

    def __init__(self, scales=(0.5, 0.75, 1.0, 1.25, 1.5)):
        self.scales = [float(s) for s in scales]

    def __call__(self, img):
        W, H = img.size
        sizes = [(int(W * s), int(H * s)) for s in self.scales]
        return [img.resize(size, Image.BILINEAR) for size in sizes]

    def __repr__(self):
        return f"{self.__class__.__name__}(scales={self.scales})"
