"""Unit tests for src/datasets/uavid.py

Tests the parts that can run without the real 4 K UAVid dataset:
  - _build_trainid_lut: Clutter→0, Building→1, …, MovingCar→7, unknown→255
  - _rgb_label_to_trainid: pixel-level conversion correctness
  - key uniqueness: seq/stem scheme prevents collisions across sequences
  - UAVid.__init__ error paths: bad mode, missing root
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
from PIL import Image

from src.datasets.uavid import UAVid


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MINIMAL_INFO: List[Dict[str, Any]] = [
    {"name": "Clutter", "ignoreInEval": True, "color": [0, 0, 0], "trainId": 0},
    {"name": "Building", "ignoreInEval": False, "color": [128, 0, 0], "trainId": 1},
    {"name": "Road", "ignoreInEval": False, "color": [128, 64, 128], "trainId": 2},
    {"name": "Static Car", "ignoreInEval": False, "color": [192, 0, 192], "trainId": 3},
    {"name": "Tree", "ignoreInEval": False, "color": [0, 128, 0], "trainId": 4},
    {"name": "Vegetation", "ignoreInEval": False, "color": [128, 128, 0], "trainId": 5},
    {"name": "Human", "ignoreInEval": False, "color": [64, 64, 0], "trainId": 6},
    {"name": "Moving Car", "ignoreInEval": False, "color": [64, 0, 128], "trainId": 7},
]


@pytest.fixture()
def trainid_lut():
    return UAVid._build_trainid_lut(MINIMAL_INFO, ignore_lb=255)


# ---------------------------------------------------------------------------
# _build_trainid_lut
# ---------------------------------------------------------------------------


class TestBuildTrainidLut:
    def test_shape_and_dtype(self, trainid_lut):
        assert trainid_lut.shape == (256, 256, 256)
        assert trainid_lut.dtype == np.uint8

    def test_clutter_is_zero(self, trainid_lut):
        """Clutter must map to trainId=0 (included in loss, excluded from mIoU)."""
        assert trainid_lut[0, 0, 0] == 0

    def test_building_is_one(self, trainid_lut):
        assert trainid_lut[128, 0, 0] == 1

    def test_road_is_two(self, trainid_lut):
        assert trainid_lut[128, 64, 128] == 2

    def test_moving_car_is_seven(self, trainid_lut):
        assert trainid_lut[64, 0, 128] == 7

    def test_unknown_colour_is_ignore(self, trainid_lut):
        """Pixels not in the palette must get ignore_lb=255."""
        assert trainid_lut[7, 8, 9] == 255

    def test_all_eight_classes_present(self, trainid_lut):
        """All trainIds 0-7 must be reachable from the palette."""
        found = set()
        for cls in MINIMAL_INFO:
            r, g, b = cls["color"]
            found.add(int(trainid_lut[r, g, b]))
        assert found == set(range(8))


# ---------------------------------------------------------------------------
# _rgb_label_to_trainid (via a minimal UAVid instance)
# ---------------------------------------------------------------------------


def _make_minimal_dataset_root(root: Path) -> None:
    """Create a minimal uavid_train/ tree with one sequence and one image."""
    seq_dir = root / "seq1"
    img_dir = seq_dir / "Images"
    label_dir = seq_dir / "Labels"
    img_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    # 4×4 RGB image (content doesn't matter for LUT tests)
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(img_dir / "000001.png")
    # 4×4 label: Building pixels [128,0,0]
    arr = np.full((4, 4, 3), [128, 0, 0], dtype=np.uint8)
    Image.fromarray(arr).save(label_dir / "000001.png")


class TestRGBLabelConversion:
    @pytest.fixture()
    def ds(self, tmp_path, info_json_path):
        _make_minimal_dataset_root(tmp_path)
        return UAVid(
            config_file=str(info_json_path),
            ignore_lb=255,
            rootpth=str(tmp_path),
            cropsize=(4, 4),
            mode="train",
            val_seqs=["seq16"],  # seq16 doesn't exist → all seqs go to train
        )

    @pytest.fixture()
    def info_json_path(self, tmp_path):
        import json

        p = tmp_path / "UAVid_info.json"
        p.write_text(json.dumps(MINIMAL_INFO))
        return p

    def test_building_label_converts_to_trainid_one(self, tmp_path, info_json_path):
        """Building (RGB [128,0,0]) must produce trainId=1 after conversion."""

        _make_minimal_dataset_root(tmp_path)
        ds = UAVid(
            config_file=str(info_json_path),
            ignore_lb=255,
            rootpth=str(tmp_path),
            cropsize=(4, 4),
            mode="train",
            val_seqs=["seq16"],
        )
        # Build a small Building-coloured RGB image and convert it
        rgb = Image.fromarray(np.full((4, 4, 3), [128, 0, 0], dtype=np.uint8))
        result = ds._rgb_label_to_trainid(rgb)
        arr = np.array(result)
        assert (arr == 1).all(), f"Expected all 1, got unique values: {np.unique(arr)}"

    def test_clutter_label_converts_to_zero(self, tmp_path, info_json_path):
        """Clutter (RGB [0,0,0]) → trainId=0 (NOT 255 as in YOLO mapping)."""
        _make_minimal_dataset_root(tmp_path)
        ds = UAVid(
            config_file=str(info_json_path),
            ignore_lb=255,
            rootpth=str(tmp_path),
            cropsize=(4, 4),
            mode="train",
            val_seqs=["seq16"],
        )
        rgb = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
        result = ds._rgb_label_to_trainid(rgb)
        arr = np.array(result)
        assert (arr == 0).all()

    def test_unknown_colour_converts_to_ignore(self, tmp_path, info_json_path):
        _make_minimal_dataset_root(tmp_path)
        ds = UAVid(
            config_file=str(info_json_path),
            ignore_lb=255,
            rootpth=str(tmp_path),
            cropsize=(4, 4),
            mode="train",
            val_seqs=["seq16"],
        )
        rgb = Image.fromarray(np.full((4, 4, 3), [7, 8, 9], dtype=np.uint8))
        result = ds._rgb_label_to_trainid(rgb)
        assert (np.array(result) == 255).all()

    def test_output_is_mode_L(self, tmp_path, info_json_path):
        _make_minimal_dataset_root(tmp_path)
        ds = UAVid(
            config_file=str(info_json_path),
            ignore_lb=255,
            rootpth=str(tmp_path),
            cropsize=(4, 4),
            mode="train",
            val_seqs=["seq16"],
        )
        rgb = Image.fromarray(np.full((4, 4, 3), [128, 0, 0], dtype=np.uint8))
        result = ds._rgb_label_to_trainid(rgb)
        assert result.mode == "L"


# ---------------------------------------------------------------------------
# Key-collision safety: seq/stem scheme
# ---------------------------------------------------------------------------


class TestKeyUniqueness:
    def test_same_filename_in_two_seqs_gets_distinct_keys(
        self, tmp_path, info_json_path
    ):
        """
        UAVid filenames like '000001.png' repeat across sequences.
        The old code used bare stem as dict key → seq2 would overwrite seq1.
        New code uses 'seqN/stem' keys.
        """

        # Create two sequences, each with the same filename
        for seq in ("seq1", "seq2"):
            img_dir = tmp_path / seq / "Images"
            label_dir = tmp_path / seq / "Labels"
            img_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
                img_dir / "000001.png"
            )
            Image.fromarray(np.full((4, 4, 3), [128, 0, 0], dtype=np.uint8)).save(
                label_dir / "000001.png"
            )

        ds = UAVid(
            config_file=str(info_json_path),
            ignore_lb=255,
            rootpth=str(tmp_path),
            cropsize=(4, 4),
            mode="train",
            val_seqs=["seq16"],  # neither seq1 nor seq2 → both are train
        )
        assert len(ds) == 2, (
            f"Expected 2 distinct samples (one per sequence), got {len(ds)}. "
            "Duplicate key collision may have caused one to overwrite the other."
        )
        assert "seq1/000001" in ds.imnames
        assert "seq2/000001" in ds.imnames

    @pytest.fixture()
    def info_json_path(self, tmp_path):
        import json

        p = tmp_path / "UAVid_info.json"
        p.write_text(json.dumps(MINIMAL_INFO))
        return p


# ---------------------------------------------------------------------------
# __init__ error paths
# ---------------------------------------------------------------------------


class TestUAVidInitErrors:
    def test_invalid_mode_raises(self, tmp_path):
        import json

        p = tmp_path / "info.json"
        p.write_text(json.dumps(MINIMAL_INFO))
        (tmp_path / "seq1" / "Images").mkdir(parents=True)
        with pytest.raises(ValueError, match="not supported"):
            UAVid(str(p), 255, str(tmp_path), (4, 4), mode="test")

    def test_missing_root_raises(self, tmp_path):
        import json

        p = tmp_path / "info.json"
        p.write_text(json.dumps(MINIMAL_INFO))
        with pytest.raises(FileNotFoundError, match="does not exist"):
            UAVid(str(p), 255, "/nonexistent/path", (4, 4))

    def test_no_val_seqs_raises(self, tmp_path):
        """If val_seqs are specified but none exist in rootpth, raise."""
        import json

        p = tmp_path / "info.json"
        p.write_text(json.dumps(MINIMAL_INFO))
        (tmp_path / "seq1" / "Images").mkdir(parents=True)
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
            tmp_path / "seq1" / "Images" / "000001.png"
        )
        with pytest.raises(FileNotFoundError, match="No sequences found"):
            UAVid(str(p), 255, str(tmp_path), (4, 4), mode="val", val_seqs=["seq99"])
