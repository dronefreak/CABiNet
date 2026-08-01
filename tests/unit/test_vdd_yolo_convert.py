"""Tests for src/scripts/convert_vdd_to_yolo.py

Like AeroScapes, VDD masks are already single-channel class IDs (no RGB
colour decoding), so this converter just symlinks files while validating
mask pixel values. Coverage:
  - discover_stems: finds stems present in both src/ and gt/
  - validate_mask: accepts known class IDs + IGNORE_LABEL, rejects anything else
  - convert_split: symlinks (not copies) matching pairs, skips missing
    counterparts with a warning, dry-run writes nothing, extension is
    normalized to lowercase .jpg
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.scripts.convert_vdd_to_yolo import (
    IGNORE_LABEL,
    VALID_CLASS_IDS,
    convert_split,
    discover_stems,
    validate_mask,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_vdd_split(root: Path, split: str, stems_and_values: dict, size=(4, 4)) -> None:
    """Create a minimal VDD-style <root>/<split>/{src,gt}/ tree.

    stems_and_values: {stem: mask_fill_value} — each entry becomes one
    uppercase-extension JPEG image and one single-channel PNG mask.
    """
    h, w = size
    src_dir = root / split / "src"
    gt_dir = root / split / "gt"
    src_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    for stem, value in stems_and_values.items():
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        Image.fromarray(rgb).save(src_dir / f"{stem}.JPG")
        mask = np.full((h, w), value, dtype=np.uint8)
        Image.fromarray(mask).save(gt_dir / f"{stem}.png")


# ---------------------------------------------------------------------------
# discover_stems
# ---------------------------------------------------------------------------


class TestDiscoverStems:
    def test_finds_common_stems(self, tmp_path):
        _make_vdd_split(tmp_path, "train", {"DJI_0008": 0, "DJI_0010": 1})
        stems = discover_stems(tmp_path / "train")
        assert stems == ["DJI_0008", "DJI_0010"]

    def test_missing_split_dir_returns_empty(self, tmp_path):
        assert discover_stems(tmp_path / "nonexistent") == []

    def test_orphan_stems_excluded(self, tmp_path):
        _make_vdd_split(tmp_path, "train", {"DJI_0008": 0})
        (tmp_path / "train" / "src" / "DJI_0099.JPG").write_bytes(b"")
        stems = discover_stems(tmp_path / "train")
        assert stems == ["DJI_0008"]


# ---------------------------------------------------------------------------
# validate_mask
# ---------------------------------------------------------------------------


class TestValidateMask:
    def test_valid_class_id_passes(self, tmp_path):
        mask = np.full((4, 4), 5, dtype=np.uint8)
        p = tmp_path / "mask.png"
        Image.fromarray(mask).save(p)
        valid, _ = validate_mask(p)
        assert valid

    def test_ignore_label_passes(self, tmp_path):
        mask = np.full((4, 4), IGNORE_LABEL, dtype=np.uint8)
        p = tmp_path / "mask.png"
        Image.fromarray(mask).save(p)
        valid, _ = validate_mask(p)
        assert valid

    def test_out_of_range_value_fails(self, tmp_path):
        mask = np.full((4, 4), 200, dtype=np.uint8)
        p = tmp_path / "mask.png"
        Image.fromarray(mask).save(p)
        valid, msg = validate_mask(p)
        assert not valid
        assert "200" in msg

    def test_all_seven_classes_valid(self, tmp_path):
        assert VALID_CLASS_IDS == frozenset(range(7))
        for c in range(7):
            mask = np.full((2, 2), c, dtype=np.uint8)
            p = tmp_path / f"mask_{c}.png"
            Image.fromarray(mask).save(p)
            valid, _ = validate_mask(p)
            assert valid, f"class id {c} should be valid"


# ---------------------------------------------------------------------------
# convert_split
# ---------------------------------------------------------------------------


class TestConvertSplit:
    def test_links_images_and_masks(self, tmp_path):
        src = tmp_path / "vdd"
        dst = tmp_path / "vdd_yolo"
        _make_vdd_split(src, "train", {"DJI_0008": 0, "DJI_0010": 5})
        n = convert_split(src, dst, "train")
        assert n == 2
        assert (dst / "images" / "train" / "DJI_0008.jpg").exists()
        assert (dst / "masks" / "train" / "DJI_0008.png").exists()

    def test_extension_normalized_to_lowercase_jpg(self, tmp_path):
        src = tmp_path / "vdd"
        dst = tmp_path / "vdd_yolo"
        _make_vdd_split(src, "train", {"DJI_0008": 0})
        convert_split(src, dst, "train")
        assert (dst / "images" / "train" / "DJI_0008.jpg").exists()
        assert not list((dst / "images" / "train").glob("*.JPG"))

    def test_links_are_symlinks(self, tmp_path):
        src = tmp_path / "vdd"
        dst = tmp_path / "vdd_yolo"
        _make_vdd_split(src, "train", {"DJI_0008": 0})
        convert_split(src, dst, "train")
        out_img = dst / "images" / "train" / "DJI_0008.jpg"
        out_mask = dst / "masks" / "train" / "DJI_0008.png"
        assert out_img.is_symlink()
        assert out_mask.is_symlink()
        assert out_img.resolve() == (src / "train" / "src" / "DJI_0008.JPG").resolve()

    def test_missing_mask_skipped_with_warning(self, tmp_path, capsys):
        src = tmp_path / "vdd"
        dst = tmp_path / "vdd_yolo"
        _make_vdd_split(src, "train", {"DJI_0008": 0})
        (src / "train" / "gt" / "DJI_0008.png").unlink()
        n = convert_split(src, dst, "train")
        assert n == 0
        assert "Missing mask" in capsys.readouterr().out

    def test_invalid_mask_pixel_skipped_with_warning(self, tmp_path, capsys):
        src = tmp_path / "vdd"
        dst = tmp_path / "vdd_yolo"
        _make_vdd_split(src, "train", {"DJI_0008": 200})  # out-of-range value
        n = convert_split(src, dst, "train")
        assert n == 0
        assert "unexpected pixel value" in capsys.readouterr().out
        assert not (dst / "images" / "train" / "DJI_0008.jpg").exists()

    def test_dry_run_writes_nothing(self, tmp_path):
        src = tmp_path / "vdd"
        dst = tmp_path / "vdd_yolo"
        _make_vdd_split(src, "train", {"DJI_0008": 0})
        n = convert_split(src, dst, "train", dry_run=True)
        assert n == 1
        assert not dst.exists()

    def test_train_val_test_separation(self, tmp_path):
        src = tmp_path / "vdd"
        dst = tmp_path / "vdd_yolo"
        _make_vdd_split(src, "train", {"a": 0})
        _make_vdd_split(src, "val", {"b": 1})
        _make_vdd_split(src, "test", {"c": 2})

        convert_split(src, dst, "train")
        convert_split(src, dst, "val")
        convert_split(src, dst, "test")

        assert [p.stem for p in (dst / "images" / "train").glob("*.jpg")] == ["a"]
        assert [p.stem for p in (dst / "images" / "val").glob("*.jpg")] == ["b"]
        assert [p.stem for p in (dst / "images" / "test").glob("*.jpg")] == ["c"]
