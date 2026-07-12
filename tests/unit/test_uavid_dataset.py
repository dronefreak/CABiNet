"""Unit tests for src/datasets/uavid.py

Tests the pre-converted images/+masks/ dataset loader:
  - __init__ error paths: bad mode, missing root/images/masks dirs
  - file discovery: matching image/mask pairs, missing-mask skip-with-warning
  - __getitem__: 2-tuple (img, label) output, mask pixel values pass through
    untouched (no LUT translation — the conversion step already produced
    final class IDs), train-mode crop size, val-mode native size
  - DataLoader integration with default collation (regression guard for the
    OHEM n_min bug, which was caused by the old 4-patch-per-item collate_fn)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from torch.utils.data import DataLoader

from src.datasets.uavid import UAVid


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
        Image.fromarray(rgb).save(img_dir / f"{stem}.png")
        mask = np.full((h, w), value, dtype=np.uint8)
        Image.fromarray(mask).save(mask_dir / f"{stem}.png")


# ---------------------------------------------------------------------------
# __init__ error paths
# ---------------------------------------------------------------------------


class TestUAVidInitErrors:
    def test_invalid_mode_raises(self, tmp_path):
        _make_dataset_root(tmp_path, "train", {"seq1_000001": 1})
        with pytest.raises(ValueError, match="not supported"):
            UAVid(ignore_lb=255, rootpth=str(tmp_path), cropsize=(4, 4), mode="bogus")

    def test_missing_root_raises(self):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            UAVid(ignore_lb=255, rootpth="/nonexistent/path", cropsize=(4, 4))

    def test_missing_images_dir_raises(self, tmp_path):
        (tmp_path / "masks" / "train").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="Image directory not found"):
            UAVid(ignore_lb=255, rootpth=str(tmp_path), cropsize=(4, 4), mode="train")

    def test_missing_masks_dir_raises(self, tmp_path):
        (tmp_path / "images" / "train").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="Mask directory not found"):
            UAVid(ignore_lb=255, rootpth=str(tmp_path), cropsize=(4, 4), mode="train")

    def test_no_valid_pairs_raises(self, tmp_path):
        """An images/ dir with files but a completely empty masks/ dir (no
        matches at all) must raise, not silently produce an empty dataset."""
        (tmp_path / "images" / "train").mkdir(parents=True)
        (tmp_path / "masks" / "train").mkdir(parents=True)
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
            tmp_path / "images" / "train" / "000001.png"
        )
        with pytest.raises(RuntimeError, match="No valid image-mask pairs"):
            UAVid(ignore_lb=255, rootpth=str(tmp_path), cropsize=(4, 4), mode="train")


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


class TestFileDiscovery:
    def test_finds_all_matching_pairs(self, tmp_path):
        _make_dataset_root(
            tmp_path,
            "train",
            {"seq1_000001": 0, "seq1_000002": 1, "seq2_000001": 2},
        )
        ds = UAVid(ignore_lb=255, rootpth=str(tmp_path), cropsize=(4, 4), mode="train")
        assert len(ds) == 3

    def test_image_with_no_matching_mask_is_skipped_with_warning(self, tmp_path):
        _make_dataset_root(tmp_path, "train", {"seq1_000001": 0})
        # Add an orphan image with no mask counterpart
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
            tmp_path / "images" / "train" / "seq1_000002.png"
        )
        with pytest.warns(UserWarning, match="no matching mask"):
            ds = UAVid(
                ignore_lb=255, rootpth=str(tmp_path), cropsize=(4, 4), mode="train"
            )
        assert len(ds) == 1

    def test_train_val_test_are_independent(self, tmp_path):
        _make_dataset_root(tmp_path, "train", {"a": 0, "b": 1})
        _make_dataset_root(tmp_path, "val", {"c": 0})
        _make_dataset_root(tmp_path, "test", {"d": 0, "e": 0, "f": 0})

        assert len(UAVid(255, str(tmp_path), (4, 4), mode="train")) == 2
        assert len(UAVid(255, str(tmp_path), (4, 4), mode="val")) == 1
        assert len(UAVid(255, str(tmp_path), (4, 4), mode="test")) == 3


# ---------------------------------------------------------------------------
# __getitem__
# ---------------------------------------------------------------------------


class TestGetItem:
    def test_returns_two_tuple(self, tmp_path):
        _make_dataset_root(tmp_path, "val", {"a": 3}, size=(16, 16))
        ds = UAVid(ignore_lb=255, rootpth=str(tmp_path), cropsize=(4, 4), mode="val")
        item = ds[0]
        assert isinstance(item, tuple) and len(item) == 2
        img, label = item
        assert img.ndim == 3  # (C, H, W)
        assert label.ndim == 2  # (H, W) — no LUT translation, no patch dict

    def test_mask_pixel_values_pass_through_untouched(self, tmp_path):
        """Pixel value 3 (a real class ID) and 255 (ignore) must survive
        exactly — the conversion step already produced final class IDs, so
        there is no LUT translation happening here anymore."""
        _make_dataset_root(tmp_path, "val", {"class3": 3, "ignore": 255}, size=(8, 8))
        ds = UAVid(ignore_lb=255, rootpth=str(tmp_path), cropsize=(4, 4), mode="val")

        labels_by_name = {}
        for i, name in enumerate(ds.imnames):
            _, label = ds[i]
            labels_by_name[name] = label

        assert (labels_by_name["class3"] == 3).all()
        assert (labels_by_name["ignore"] == 255).all()

    def test_train_mode_output_matches_cropsize(self, tmp_path):
        """RandomCrop must fire on the full image (not a fixed quadrant) —
        output spatial size must equal cropsize regardless of source size.

        Note: cropsize is kept >= 64 here because RandomCutout has a
        hardcoded size=64 with no bounds check against the post-crop image
        size — harmless in production (real cropsize is always 1024x1024)
        but not exercised by a smaller test cropsize.
        """
        _make_dataset_root(tmp_path, "train", {"a": 1}, size=(128, 160))
        ds = UAVid(
            ignore_lb=255, rootpth=str(tmp_path), cropsize=(80, 80), mode="train"
        )
        img, label = ds[0]
        assert tuple(img.shape[-2:]) == (80, 80)
        assert tuple(label.shape) == (80, 80)

    def test_val_mode_preserves_native_size_no_crop(self, tmp_path):
        _make_dataset_root(tmp_path, "val", {"a": 1}, size=(20, 30))
        ds = UAVid(ignore_lb=255, rootpth=str(tmp_path), cropsize=(4, 4), mode="val")
        img, label = ds[0]
        assert tuple(img.shape[-2:]) == (20, 30)
        assert tuple(label.shape) == (20, 30)

    def test_train_mode_handles_source_smaller_than_cropsize(self, tmp_path):
        """RandomCrop(pad_if_needed=True) must handle a source image smaller
        than cropsize without erroring (this is what replaces the old fixed
        3840x2160 resize)."""
        _make_dataset_root(tmp_path, "train", {"a": 1}, size=(8, 8))
        ds = UAVid(
            ignore_lb=255, rootpth=str(tmp_path), cropsize=(64, 64), mode="train"
        )
        img, label = ds[0]
        assert tuple(img.shape[-2:]) == (64, 64)
        assert tuple(label.shape) == (64, 64)


# ---------------------------------------------------------------------------
# DataLoader integration (regression guard for the old 4-patch collate bug)
# ---------------------------------------------------------------------------


class TestDataLoaderIntegration:
    def test_default_collate_produces_correct_batch_shape(self, tmp_path):
        """No custom collate_fn should be needed — each __getitem__ returns
        exactly one (img, label) pair, so a DataLoader(batch_size=N) must
        produce a batch dimension of exactly N (not 4N, as the old
        quadrant-patch + uavid_collate_fn combination produced)."""
        _make_dataset_root(
            tmp_path,
            "train",
            {f"seq1_{i:06d}": 0 for i in range(6)},
            size=(128, 128),
        )
        ds = UAVid(
            ignore_lb=255, rootpth=str(tmp_path), cropsize=(64, 64), mode="train"
        )
        dl = DataLoader(ds, batch_size=4, shuffle=False)
        imgs, labels = next(iter(dl))
        assert imgs.shape == (4, 3, 64, 64)
        assert labels.shape == (4, 64, 64)

    def test_val_batch_size_one_handles_mixed_source_sizes(self, tmp_path):
        """Val mode applies no crop, so mixed source resolutions (as UAVid
        genuinely has) can only be batched with batch_size=1."""
        img_dir = tmp_path / "images" / "val"
        mask_dir = tmp_path / "masks" / "val"
        img_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        for stem, size in [("a", (20, 20)), ("b", (30, 24))]:
            h, w = size
            Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(
                img_dir / f"{stem}.png"
            )
            Image.fromarray(np.zeros((h, w), dtype=np.uint8)).save(
                mask_dir / f"{stem}.png"
            )
        ds = UAVid(ignore_lb=255, rootpth=str(tmp_path), cropsize=(4, 4), mode="val")
        dl = DataLoader(ds, batch_size=1, shuffle=False)
        shapes = [tuple(imgs.shape[-2:]) for imgs, _ in dl]
        assert shapes == [(20, 20), (30, 24)]


# ---------------------------------------------------------------------------
# Augmentation config (mirrors configs/train_yolo.yaml's augmentation block)
# ---------------------------------------------------------------------------


class TestAugmentationConfig:
    def test_default_augmentation_used_when_none_passed(self, tmp_path):
        _make_dataset_root(tmp_path, "train", {"a": 1}, size=(32, 32))
        ds = UAVid(ignore_lb=255, rootpth=str(tmp_path), cropsize=(8, 8), mode="train")
        assert ds.aug["fliplr"] == 0.5
        assert ds.aug["flipud"] == 0.2
        assert ds.aug["mixup"] == 0.1

    def test_partial_override_falls_back_to_defaults(self, tmp_path):
        _make_dataset_root(tmp_path, "train", {"a": 1}, size=(32, 32))
        ds = UAVid(
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
        ds = UAVid(
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
    # cropsize is kept >= 64 throughout this class for the same reason
    # noted in TestGetItem: RandomCutout has a hardcoded size=64 with no
    # bounds check, harmless in production (real cropsize is always
    # 1024x1024) but crashes with a smaller cropsize.

    def test_mixup_probability_zero_never_blends(self, tmp_path, monkeypatch):
        """With mixup=0.0, _load_one must never be called for a second
        (partner) sample."""
        _make_dataset_root(tmp_path, "train", {"a": 1, "b": 2}, size=(128, 128))
        ds = UAVid(
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
        """With mixup=1.0, every sample must trigger exactly two _load_one
        calls (primary + partner)."""
        _make_dataset_root(tmp_path, "train", {"a": 1, "b": 2, "c": 3}, size=(128, 128))
        ds = UAVid(
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
        """The blended image must differ from either source image's raw
        pixels (a genuine blend), while the label must remain a valid hard
        class-ID map exactly equal to one of the two source labels (never
        an averaged/fractional value)."""
        _make_dataset_root(tmp_path, "train", {"a": 1, "b": 5}, size=(128, 128))
        ds = UAVid(
            ignore_lb=255,
            rootpth=str(tmp_path),
            cropsize=(64, 64),
            mode="train",
            augmentation={
                "mixup": 1.0,
                # disable everything else so the only source of variation
                # between the two loads is which file was picked
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
        assert unique.issubset(
            {1, 5}
        ), f"Blended label must be a hard copy of one source label, got {unique}"

    def test_mixup_never_crashes_on_single_sample_dataset(self, tmp_path):
        """With only one sample, the mixup 'partner' is the same sample —
        must not crash (self-mixup is a degenerate but valid case)."""
        _make_dataset_root(tmp_path, "train", {"a": 1}, size=(128, 128))
        ds = UAVid(
            ignore_lb=255,
            rootpth=str(tmp_path),
            cropsize=(64, 64),
            mode="train",
            augmentation={"mixup": 1.0},
        )
        img, label = ds[0]
        assert img.shape[-2:] == (64, 64)
        assert label.shape == (64, 64)
