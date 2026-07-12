"""Unit tests for src/utils/class_weights.py

Covers:
  - compute_class_weights: cls_pw=0 disables weighting; matches the ENet
    inverse-log formula for cls_pw=1; rarer classes get higher weight;
    weight magnitude increases monotonically with cls_pw
  - get_class_pixel_counts: basic counting, ignore_lb exclusion,
    max_samples capping, empty dataset, out-of-range class ids
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.utils.class_weights import compute_class_weights, get_class_pixel_counts

# ---------------------------------------------------------------------------
# Fake dataset helper
# ---------------------------------------------------------------------------


class _FakeDataset:
    """Minimal Dataset-like object yielding (image, label) pairs from a
    fixed list, tracking which indices were accessed via __getitem__."""

    def __init__(self, labels: list) -> None:
        self.labels = labels
        self.accessed_indices: list[int] = []

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        self.accessed_indices.append(idx)
        return torch.zeros(3, 2, 2), self.labels[idx]


# ---------------------------------------------------------------------------
# get_class_pixel_counts
# ---------------------------------------------------------------------------


class TestGetClassPixelCounts:
    def test_basic_counts(self):
        # label 0: all class 0 (4 pixels); label 1: 2x class 1, 2x class 2
        labels = [
            torch.zeros(2, 2, dtype=torch.long),
            torch.tensor([[1, 1], [2, 2]], dtype=torch.long),
        ]
        ds = _FakeDataset(labels)
        counts = get_class_pixel_counts(ds, num_classes=3, ignore_lb=255)
        assert counts.tolist() == [4, 2, 2]

    def test_ignore_label_excluded(self):
        label = torch.tensor([[0, 255], [255, 255]], dtype=torch.long)
        ds = _FakeDataset([label])
        counts = get_class_pixel_counts(ds, num_classes=1, ignore_lb=255)
        assert counts.tolist() == [1]

    def test_max_samples_caps_scanned_items(self):
        labels = [torch.zeros(2, 2, dtype=torch.long) for _ in range(50)]
        ds = _FakeDataset(labels)
        get_class_pixel_counts(ds, num_classes=1, ignore_lb=255, max_samples=5)
        assert len(ds.accessed_indices) == 5

    def test_max_samples_none_scans_everything(self):
        labels = [torch.zeros(2, 2, dtype=torch.long) for _ in range(10)]
        ds = _FakeDataset(labels)
        get_class_pixel_counts(ds, num_classes=1, ignore_lb=255, max_samples=None)
        assert len(ds.accessed_indices) == 10

    def test_empty_dataset_returns_zeros(self):
        ds = _FakeDataset([])
        counts = get_class_pixel_counts(ds, num_classes=4, ignore_lb=255)
        assert counts.tolist() == [0, 0, 0, 0]

    def test_out_of_range_class_id_ignored(self):
        """A stray class id >= num_classes (corrupt label) must not crash
        or be counted, not even into an out-of-bounds slot."""
        label = torch.tensor([[0, 99]], dtype=torch.long)
        ds = _FakeDataset([label])
        counts = get_class_pixel_counts(ds, num_classes=2, ignore_lb=255)
        assert counts.tolist() == [1, 0]

    def test_accepts_numpy_labels(self):
        label = np.array([[0, 1], [1, 1]], dtype=np.int64)
        ds = _FakeDataset([label])
        counts = get_class_pixel_counts(ds, num_classes=2, ignore_lb=255)
        assert counts.tolist() == [1, 3]


# ---------------------------------------------------------------------------
# compute_class_weights
# ---------------------------------------------------------------------------


class TestComputeClassWeights:
    def test_cls_pw_zero_gives_uniform_weights(self):
        counts = np.array([1000, 1, 500], dtype=np.int64)
        weights = compute_class_weights(counts, cls_pw=0.0)
        assert weights.tolist() == [1.0, 1.0, 1.0]

    def test_cls_pw_one_matches_enet_formula(self):
        counts = np.array([80, 20], dtype=np.int64)
        weights = compute_class_weights(counts, cls_pw=1.0)

        total = 100
        expected = [
            (1.0 / math.log(1.02 + 80 / total)) ** 1.0,
            (1.0 / math.log(1.02 + 20 / total)) ** 1.0,
        ]
        assert weights == pytest.approx(expected, rel=1e-6)

    def test_rare_class_gets_higher_weight(self):
        counts = np.array([9000, 100], dtype=np.int64)  # class 1 is rare
        weights = compute_class_weights(counts, cls_pw=1.0)
        assert weights[1] > weights[0]

    def test_weight_increases_monotonically_with_cls_pw(self):
        counts = np.array([9500, 500], dtype=np.int64)
        w_low = compute_class_weights(counts, cls_pw=0.5)[1]
        w_high = compute_class_weights(counts, cls_pw=1.0)[1]
        w_zero = compute_class_weights(counts, cls_pw=0.0)[1]
        assert w_zero == 1.0
        assert w_zero < w_low < w_high

    def test_all_zero_counts_does_not_crash_or_produce_nan(self):
        counts = np.zeros(4, dtype=np.int64)
        weights = compute_class_weights(counts, cls_pw=1.0)
        assert np.isfinite(weights).all()

    def test_output_length_matches_input(self):
        counts = np.array([10, 20, 30, 40, 50], dtype=np.int64)
        weights = compute_class_weights(counts, cls_pw=0.7)
        assert len(weights) == 5
