"""Unit tests for src/datasets/vdd.py

Tests the pre-converted images/+masks/ dataset loader:
  - __init__ error paths: bad mode, missing root/images/masks dirs
  - file discovery: matching image/mask pairs, missing-mask skip-with-warning
  - __getitem__: 2-tuple (img, label) output, mask pixel values pass through
    untouched (no LUT translation — the source masks are already final
    class IDs), train-mode crop size, val-mode native size
  - DataLoader integration with default collation, including a batch_size>1
    val-mode batch (VDD source images are uniformly 4000x3000, unlike
    UAVid's mixed resolutions, so this is safe here)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from torch.utils.data import DataLoader

from src.datasets.vdd import VDD


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_dataset_root(
    root: Path, mode: str, names_and_values: dict, size: tuple = (8, 8)
) -> None:
    """Build a minimal <root>/images/<mode>/ + <root>/masks/<mode>/ tree.

    names_and_values: {stem: fill_value} — each entry becomes one RGB image
    (content irrelevant) and one single-channel mask filled with fill_value.
    """
    h, w = size
    img_dir = root / "images" / mode
    mask_dir = root / "masks" / mode
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for stem, value in names_and_values.items():
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        Image.fromarray(rgb).save(img_dir / f"{stem}.jpg")
        mask = np.full((h, w), value, dtype=np.uint8)
        Image.fromarray(mask).save(mask_dir / f"{stem}.png")


# ---------------------------------------------------------------------------
# __init__ error paths
# ---------------------------------------------------------------------------


class TestVDDInitErrors:
    def test_invalid_mode_raises(self, tmp_path):
        _make_dataset_root(tmp_path, "train", {"DJI_0008": 1})
        with pytest.raises(ValueError, match="not supported"):
            VDD(ignore_lb=255, rootpth=str(tmp_path), cropsize=(4, 4), mode="bogus")

    def test_missing_root_raises(self):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            VDD(ignore_lb=255, rootpth="/nonexistent/path", cropsize=(4, 4))

    def test_missing_images_dir_raises(self, tmp_path):
        (tmp_path / "masks" / "train").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="Image directory not found"):
            VDD(ignore_lb=255, rootpth=str(tmp_path), cropsize=(4, 4), mode="train")

    def test_missing_masks_dir_raises(self, tmp_path):
        (tmp_path / "images" / "train").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="Mask directory not found"):
            VDD(ignore_lb=255, rootpth=str(tmp_path), cropsize=(4, 4), mode="train")

    def test_no_valid_pairs_raises(self, tmp_path):
        (tmp_path / "images" / "train").mkdir(parents=True)
        (tmp_path / "masks" / "train").mkdir(parents=True)
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
            tmp_path / "images" / "train" / "DJI_0008.jpg"
        )
        with pytest.raises(RuntimeError, match="No valid image-mask pairs"):
            VDD(ignore_lb=255, rootpth=str(tmp_path), cropsize=(4, 4), mode="train")


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


class TestFileDiscovery:
    def test_finds_all_matching_pairs(self, tmp_path):
        _make_dataset_root(
            tmp_path, "train", {"DJI_0008": 0, "DJI_0010": 1, "DJI_0011": 2}
        )
        ds = VDD(ignore_lb=255, rootpth=str(tmp_path), cropsize=(4, 4), mode="train")
        assert len(ds) == 3

    def test_image_with_no_matching_mask_is_skipped_with_warning(self, tmp_path):
        _make_dataset_root(tmp_path, "train", {"DJI_0008": 0})
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
            tmp_path / "images" / "train" / "DJI_0099.jpg"
        )
        with pytest.warns(UserWarning, match="no matching mask"):
            ds = VDD(ignore_lb=255, rootpth=str(tmp_path), cropsize=(4, 4), mode="train")
        assert len(ds) == 1

    def test_train_val_test_are_independent(self, tmp_path):
        _make_dataset_root(tmp_path, "train", {"a": 0, "b": 1})
        _make_dataset_root(tmp_path, "val", {"c": 0})
        _make_dataset_root(tmp_path, "test", {"d": 0, "e": 0, "f": 0})

        assert len(VDD(255, str(tmp_path), (4, 4), mode="train")) == 2
        assert len(VDD(255, str(tmp_path), (4, 4), mode="val")) == 1
        assert len(VDD(255, str(tmp_path), (4, 4), mode="test")) == 3


# ---------------------------------------------------------------------------
# __getitem__
# ---------------------------------------------------------------------------


class TestGetItem:
    def test_returns_two_tuple(self, tmp_path):
        _make_dataset_root(tmp_path, "val", {"a": 3}, size=(16, 16))
        ds = VDD(ignore_lb=255, rootpth=str(tmp_path), cropsize=(4, 4), mode="val")
        item = ds[0]
        assert isinstance(item, tuple) and len(item) == 2
        img, label = item
        assert img.ndim == 3  # (C, H, W)
        assert label.ndim == 2  # (H, W)

    def test_mask_pixel_values_pass_through_untouched(self, tmp_path):
        """Pixel value 6 (Water) and 255 (ignore) must survive exactly — masks
        are already final class IDs, no LUT translation."""
        _make_dataset_root(tmp_path, "val", {"class6": 6, "ignore": 255}, size=(8, 8))
        ds = VDD(ignore_lb=255, rootpth=str(tmp_path), cropsize=(4, 4), mode="val")

        labels_by_name = {}
        for i, name in enumerate(ds.imnames):
            _, label = ds[i]
            labels_by_name[name] = label

        assert (labels_by_name["class6"] == 6).all()
        assert (labels_by_name["ignore"] == 255).all()

    def test_train_mode_output_matches_cropsize(self, tmp_path):
        """cropsize is kept >= 64 here because RandomCutout has a hardcoded
        size=64 with no bounds check — harmless in production (real cropsize
        is 1024x1024) but not exercised by a smaller test cropsize."""
        _make_dataset_root(tmp_path, "train", {"a": 1}, size=(128, 160))
        ds = VDD(ignore_lb=255, rootpth=str(tmp_path), cropsize=(80, 80), mode="train")
        img, label = ds[0]
        assert tuple(img.shape[-2:]) == (80, 80)
        assert tuple(label.shape) == (80, 80)

    def test_val_mode_preserves_native_size_no_crop(self, tmp_path):
        _make_dataset_root(tmp_path, "val", {"a": 1}, size=(20, 30))
        ds = VDD(ignore_lb=255, rootpth=str(tmp_path), cropsize=(4, 4), mode="val")
        img, label = ds[0]
        assert tuple(img.shape[-2:]) == (20, 30)
        assert tuple(label.shape) == (20, 30)

    def test_train_mode_handles_source_smaller_than_cropsize(self, tmp_path):
        _make_dataset_root(tmp_path, "train", {"a": 1}, size=(8, 8))
        ds = VDD(ignore_lb=255, rootpth=str(tmp_path), cropsize=(64, 64), mode="train")
        img, label = ds[0]
        assert tuple(img.shape[-2:]) == (64, 64)
        assert tuple(label.shape) == (64, 64)


# ---------------------------------------------------------------------------
# DataLoader integration
# ---------------------------------------------------------------------------


class TestDataLoaderIntegration:
    def test_default_collate_produces_correct_batch_shape(self, tmp_path):
        _make_dataset_root(
            tmp_path,
            "train",
            {f"DJI_{i:04d}": 0 for i in range(6)},
            size=(128, 128),
        )
        ds = VDD(ignore_lb=255, rootpth=str(tmp_path), cropsize=(64, 64), mode="train")
        dl = DataLoader(ds, batch_size=4, shuffle=False)
        imgs, labels = next(iter(dl))
        assert imgs.shape == (4, 3, 64, 64)
        assert labels.shape == (4, 64, 64)

    def test_val_batch_size_greater_than_one_works_for_uniform_resolution(
        self, tmp_path
    ):
        """Unlike UAVid (mixed source resolution → batch_size=1 required),
        VDD source images are uniformly 4000x3000, so a batched val
        DataLoader must work with the default collate function."""
        _make_dataset_root(
            tmp_path, "val", {"a": 0, "b": 1, "c": 2, "d": 3}, size=(20, 20)
        )
        ds = VDD(ignore_lb=255, rootpth=str(tmp_path), cropsize=(4, 4), mode="val")
        dl = DataLoader(ds, batch_size=4, shuffle=False)
        imgs, labels = next(iter(dl))
        assert imgs.shape == (4, 3, 20, 20)
        assert labels.shape == (4, 20, 20)


# ---------------------------------------------------------------------------
# Augmentation config
# ---------------------------------------------------------------------------


class TestAugmentationConfig:
    def test_default_augmentation_used_when_none_passed(self, tmp_path):
        _make_dataset_root(tmp_path, "train", {"a": 1}, size=(32, 32))
        ds = VDD(ignore_lb=255, rootpth=str(tmp_path), cropsize=(8, 8), mode="train")
        assert ds.aug["fliplr"] == 0.5
        assert ds.aug["flipud"] == 0.2
        assert ds.aug["mixup"] == 0.1

    def test_partial_override_falls_back_to_defaults(self, tmp_path):
        _make_dataset_root(tmp_path, "train", {"a": 1}, size=(32, 32))
        ds = VDD(
            ignore_lb=255,
            rootpth=str(tmp_path),
            cropsize=(8, 8),
            mode="train",
            augmentation={"mixup": 0.0},
        )
        assert ds.aug["mixup"] == 0.0
        assert ds.aug["degrees"] == 10.0  # untouched default

    def test_mixup_disabled_in_non_train_modes(self, tmp_path):
        _make_dataset_root(tmp_path, "train", {"a": 1}, size=(16, 16))
        _make_dataset_root(tmp_path, "val", {"b": 1}, size=(16, 16))
        ds = VDD(
            ignore_lb=255,
            rootpth=str(tmp_path),
            cropsize=(8, 8),
            mode="val",
            augmentation={"mixup": 1.0},
        )
        assert ds.mixup_p == 0.0


# ---------------------------------------------------------------------------
# MixUp
# ---------------------------------------------------------------------------


class TestMixUp:
    def test_mixup_probability_zero_never_blends(self, tmp_path, monkeypatch):
        _make_dataset_root(tmp_path, "train", {"a": 1, "b": 2}, size=(128, 128))
        ds = VDD(
            ignore_lb=255,
            rootpth=str(tmp_path),
            cropsize=(64, 64),
            mode="train",
            augmentation={"mixup": 0.0},
        )
        call_count = 0
        original = ds._load_one

        def counting_load_one(idx):
            nonlocal call_count
            call_count += 1
            return original(idx)

        monkeypatch.setattr(ds, "_load_one", counting_load_one)
        for i in range(len(ds)):
            ds[i]
        assert call_count == len(ds), "mixup=0.0 must never trigger a second load"

    def test_mixup_probability_one_always_blends(self, tmp_path, monkeypatch):
        _make_dataset_root(tmp_path, "train", {"a": 1, "b": 2, "c": 3}, size=(128, 128))
        ds = VDD(
            ignore_lb=255,
            rootpth=str(tmp_path),
            cropsize=(64, 64),
            mode="train",
            augmentation={"mixup": 1.0},
        )
        call_count = 0
        original = ds._load_one

        def counting_load_one(idx):
            nonlocal call_count
            call_count += 1
            return original(idx)

        monkeypatch.setattr(ds, "_load_one", counting_load_one)
        ds[0]
        assert call_count == 2

    def test_mixup_blends_image_but_label_stays_hard(self, tmp_path):
        _make_dataset_root(tmp_path, "train", {"a": 1, "b": 6}, size=(128, 128))
        ds = VDD(
            ignore_lb=255,
            rootpth=str(tmp_path),
            cropsize=(64, 64),
            mode="train",
            augmentation={
                "mixup": 1.0,
                "degrees": 0.0,
                "translate": 0.0,
                "scale": 0.0,
                "flipud": 0.0,
                "fliplr": 0.0,
                "hsv_h": 0.0,
                "hsv_s": 0.0,
                "hsv_v": 0.0,
            },
        )
        _, label = ds[0]
        unique = set(label.unique().tolist())
        assert unique.issubset({1, 6}), (
            f"Blended label must be a hard copy of one source label, got {unique}"
        )

    def test_mixup_never_crashes_on_single_sample_dataset(self, tmp_path):
        _make_dataset_root(tmp_path, "train", {"a": 1}, size=(128, 128))
        ds = VDD(
            ignore_lb=255,
            rootpth=str(tmp_path),
            cropsize=(64, 64),
            mode="train",
            augmentation={"mixup": 1.0},
        )
        img, label = ds[0]
        assert img.shape[-2:] == (64, 64)
        assert label.shape == (64, 64)
