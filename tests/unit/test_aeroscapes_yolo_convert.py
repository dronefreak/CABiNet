"""Tests for src/scripts/convert_aeroscapes_to_yolo.py

Unlike UAVid's converter, AeroScapes masks are already single-channel class
IDs (no RGB colour decoding), so this converter is much simpler: it just
copies files while validating mask pixel values. Coverage:
  - load_split_stems / discover_splits: ImageSets/{trn,val}.txt parsing
  - validate_mask: accepts known class IDs + IGNORE_LABEL, rejects anything else
  - convert_split: copies (not symlinks) matching pairs, skips missing
    counterparts with a warning, dry-run writes nothing, train/val separation
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.scripts.convert_aeroscapes_to_yolo import (
    IGNORE_LABEL,
    VALID_CLASS_IDS,
    convert_split,
    discover_splits,
    load_split_stems,
    validate_mask,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_aeroscapes_tree(root: Path, stems_and_values: dict, size=(4, 4)) -> None:
    """Create a minimal AeroScapes-style source tree.

    stems_and_values: {stem: mask_fill_value} — each entry becomes one JPEG
    image and one single-channel PNG mask filled with mask_fill_value.
    """
    h, w = size
    img_dir = root / "JPEGImages"
    mask_dir = root / "SegmentationClass"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for stem, value in stems_and_values.items():
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        Image.fromarray(rgb).save(img_dir / f"{stem}.jpg")
        mask = np.full((h, w), value, dtype=np.uint8)
        Image.fromarray(mask).save(mask_dir / f"{stem}.png")


def _write_imagesets(root: Path, trn: list[str], val: list[str]) -> None:
    imagesets_dir = root / "ImageSets"
    imagesets_dir.mkdir(parents=True, exist_ok=True)
    (imagesets_dir / "trn.txt").write_text("\n".join(trn) + "\n" if trn else "")
    (imagesets_dir / "val.txt").write_text("\n".join(val) + "\n" if val else "")


# ---------------------------------------------------------------------------
# load_split_stems / discover_splits
# ---------------------------------------------------------------------------


class TestLoadSplitStems:
    def test_parses_stems_sorted(self, tmp_path):
        _write_imagesets(tmp_path, trn=["b", "a", "c"], val=[])
        stems = load_split_stems(tmp_path / "ImageSets", "trn.txt")
        assert stems == ["a", "b", "c"]

    def test_missing_file_returns_empty(self, tmp_path):
        (tmp_path / "ImageSets").mkdir()
        assert load_split_stems(tmp_path / "ImageSets", "trn.txt") == []

    def test_blank_lines_ignored(self, tmp_path):
        imagesets_dir = tmp_path / "ImageSets"
        imagesets_dir.mkdir()
        (imagesets_dir / "trn.txt").write_text("a\n\nb\n\n")
        assert load_split_stems(imagesets_dir, "trn.txt") == ["a", "b"]


class TestDiscoverSplits:
    def test_returns_train_and_val(self, tmp_path):
        _write_imagesets(tmp_path, trn=["a", "b"], val=["c"])
        splits = discover_splits(tmp_path)
        assert splits["train"] == ["a", "b"]
        assert splits["val"] == ["c"]

    def test_no_test_key_produced(self, tmp_path):
        _write_imagesets(tmp_path, trn=["a"], val=["b"])
        splits = discover_splits(tmp_path)
        assert "test" not in splits


# ---------------------------------------------------------------------------
# validate_mask
# ---------------------------------------------------------------------------


class TestValidateMask:
    def test_valid_class_id_passes(self, tmp_path):
        mask = np.full((4, 4), 7, dtype=np.uint8)
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
        mask = np.full((4, 4), 200, dtype=np.uint8)  # not a class ID, not IGNORE_LABEL
        p = tmp_path / "mask.png"
        Image.fromarray(mask).save(p)
        valid, msg = validate_mask(p)
        assert not valid
        assert "200" in msg

    def test_all_twelve_classes_valid(self, tmp_path):
        assert VALID_CLASS_IDS == frozenset(range(12))
        for c in range(12):
            mask = np.full((2, 2), c, dtype=np.uint8)
            p = tmp_path / f"mask_{c}.png"
            Image.fromarray(mask).save(p)
            valid, _ = validate_mask(p)
            assert valid, f"class id {c} should be valid"


# ---------------------------------------------------------------------------
# convert_split
# ---------------------------------------------------------------------------


class TestConvertSplit:
    def test_copies_images_and_masks(self, tmp_path):
        src = tmp_path / "aeroscapes"
        dst = tmp_path / "aeroscapes_yolo"
        _make_aeroscapes_tree(src, {"000001_001": 0, "000001_002": 5})
        n = convert_split(src, dst, "train", ["000001_001", "000001_002"], workers=1)
        assert n == 2
        assert (dst / "images" / "train" / "000001_001.jpg").exists()
        assert (dst / "masks" / "train" / "000001_001.png").exists()

    def test_copies_are_not_symlinks(self, tmp_path):
        src = tmp_path / "aeroscapes"
        dst = tmp_path / "aeroscapes_yolo"
        _make_aeroscapes_tree(src, {"000001_001": 0})
        convert_split(src, dst, "train", ["000001_001"], workers=1)
        out_img = dst / "images" / "train" / "000001_001.jpg"
        out_mask = dst / "masks" / "train" / "000001_001.png"
        assert not out_img.is_symlink()
        assert not out_mask.is_symlink()

    def test_missing_image_skipped_with_warning(self, tmp_path, capsys):
        src = tmp_path / "aeroscapes"
        dst = tmp_path / "aeroscapes_yolo"
        _make_aeroscapes_tree(src, {"000001_001": 0})
        n = convert_split(src, dst, "train", ["000001_001", "ghost"], workers=1)
        assert n == 1
        assert "Missing image" in capsys.readouterr().out

    def test_missing_mask_skipped_with_warning(self, tmp_path, capsys):
        src = tmp_path / "aeroscapes"
        dst = tmp_path / "aeroscapes_yolo"
        _make_aeroscapes_tree(src, {"000001_001": 0})
        (src / "SegmentationClass" / "000001_001.png").unlink()
        n = convert_split(src, dst, "train", ["000001_001"], workers=1)
        assert n == 0
        assert "Missing mask" in capsys.readouterr().out

    def test_invalid_mask_pixel_skipped_with_warning(self, tmp_path, capsys):
        src = tmp_path / "aeroscapes"
        dst = tmp_path / "aeroscapes_yolo"
        _make_aeroscapes_tree(src, {"000001_001": 200})  # out-of-range value
        n = convert_split(src, dst, "train", ["000001_001"], workers=1)
        assert n == 0
        assert "unexpected pixel value" in capsys.readouterr().out
        assert not (dst / "images" / "train" / "000001_001.jpg").exists()

    def test_dry_run_writes_nothing(self, tmp_path):
        src = tmp_path / "aeroscapes"
        dst = tmp_path / "aeroscapes_yolo"
        _make_aeroscapes_tree(src, {"000001_001": 0})
        n = convert_split(src, dst, "train", ["000001_001"], workers=1, dry_run=True)
        assert n == 1
        assert not dst.exists()

    def test_train_val_separation(self, tmp_path):
        src = tmp_path / "aeroscapes"
        dst = tmp_path / "aeroscapes_yolo"
        _make_aeroscapes_tree(src, {"a": 0, "b": 1})

        convert_split(src, dst, "train", ["a"], workers=1)
        convert_split(src, dst, "val", ["b"], workers=1)

        train_imgs = list((dst / "images" / "train").glob("*.jpg"))
        val_imgs = list((dst / "images" / "val").glob("*.jpg"))
        assert [p.stem for p in train_imgs] == ["a"]
        assert [p.stem for p in val_imgs] == ["b"]

    def test_parallel_workers_produce_same_result(self, tmp_path):
        src = tmp_path / "aeroscapes"
        dst = tmp_path / "aeroscapes_yolo"
        stems = [f"000001_{i:03d}" for i in range(6)]
        _make_aeroscapes_tree(src, {s: i % 12 for i, s in enumerate(stems)})
        n = convert_split(src, dst, "train", stems, workers=4)
        assert n == 6
        assert len(list((dst / "images" / "train").glob("*.jpg"))) == 6
