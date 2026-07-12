#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""Pixel-frequency class weighting for CABiNet's OHEM loss.

Mirrors the ENet inverse-log formula used by the YOLO26 semantic pipeline
(``ultralytics.models.yolo.semantic.train.SemanticSegmentationTrainer
.compute_class_weights``, itself from Paszke et al. 2016, arXiv:1606.02147)
so both pipelines share the same ``cls_pw`` knob and semantics:
``0.0`` = disabled (uniform weights), ``1.0`` = full inverse-frequency
weighting.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_class_pixel_counts(
    dataset: Dataset,
    num_classes: int,
    ignore_lb: int = 255,
    max_samples: Optional[int] = 200,
) -> np.ndarray:
    """Return per-class pixel counts, sampled from up to *max_samples* labels.

    Reads labels via the dataset's normal ``__getitem__`` — the resulting
    counts reflect whatever crop/augmentation the dataset applies, but class
    balance is a coarse correction factor, not a precise measurement, so a
    few hundred sampled (possibly cropped) labels give a perfectly adequate
    estimate without needing per-dataset raw-file access.

    Parameters
    ----------
    dataset:
        Any ``Dataset`` yielding ``(image, label)`` pairs where ``label`` is
        an integer class-ID tensor/array.
    num_classes:
        Number of valid classes (0..num_classes-1).
    ignore_lb:
        Label value excluded from the count (e.g. 255).
    max_samples:
        Cap on how many samples to scan (evenly spaced across the dataset).
        ``None`` scans the entire dataset.
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    n = len(dataset)  # type: ignore[arg-type]
    if n == 0:
        return counts

    if max_samples is not None and n > max_samples:
        indices = np.linspace(0, n - 1, max_samples).astype(int)
    else:
        indices = np.arange(n)

    for idx in tqdm(indices, desc="Computing class weights"):
        _, label = dataset[idx]
        label_np = label.numpy() if torch.is_tensor(label) else np.asarray(label)
        valid = label_np != ignore_lb
        if not valid.any():
            continue
        classes, class_counts = np.unique(label_np[valid], return_counts=True)
        for c, cnt in zip(classes, class_counts):
            if 0 <= c < num_classes:
                counts[int(c)] += int(cnt)

    return counts


def compute_class_weights(class_counts: np.ndarray, cls_pw: float) -> np.ndarray:
    """ENet inverse-log class weights: ``(1 / ln(1.02 + p)) ** cls_pw``.

    ``p`` is the per-class pixel frequency (fraction of all labelled
    pixels). ``cls_pw=0.0`` yields uniform weights (``1.0`` for every
    class, regardless of ``class_counts``); ``cls_pw=1.0`` is full ENet
    weighting, matching the YOLO26 pipeline's default.
    """
    if cls_pw == 0.0:
        return np.ones_like(class_counts, dtype=np.float64)

    total = max(class_counts.sum(), 1)
    p = class_counts / total
    return (1.0 / np.log(1.02 + p)) ** cls_pw
