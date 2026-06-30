"""Tests for src/scripts/convert_uavid_to_yolo.py

Covers:
  - build_colour_map: correct class IDs and ignore label for Clutter
  - build_lut: LUT entries match colour_map
  - convert_mask: pixel-level correctness, clutter→255, unknown colour→255
  - get_yolo_class_names: ordering and count
  - Integration: round-trip (write → read back, values match)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.scripts.convert_uavid_to_yolo import (
    IGNORE_LABEL,
    build_colour_map,
    build_lut,
    convert_mask,
    get_yolo_class_names,
    load_labels_info,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Minimal UAVid_info.json for testing (mirrors the real file structure)
MINIMAL_INFO = [
    {
        "hasInstances": False,
        "category": "void",
        "catid": 0,
        "name": "Clutter",
        "ignoreInEval": True,
        "id": 0,
        "color": [0, 0, 0],
        "trainId": 0,
    },
    {
        "hasInstances": False,
        "category": "construction",
        "catid": 1,
        "name": "Building",
        "ignoreInEval": False,
        "id": 1,
        "color": [128, 0, 0],
        "trainId": 1,
    },
    {
        "hasInstances": False,
        "category": "flat",
        "catid": 2,
        "name": "Road",
        "ignoreInEval": False,
        "id": 2,
        "color": [128, 64, 128],
        "trainId": 2,
    },
    {
        "hasInstances": False,
        "category": "vehicle",
        "catid": 3,
        "name": "Static Car",
        "ignoreInEval": False,
        "id": 3,
        "color": [192, 0, 192],
        "trainId": 3,
    },
    {
        "hasInstances": False,
        "category": "vegetation",
        "catid": 4,
        "name": "Tree",
        "ignoreInEval": False,
        "id": 4,
        "color": [0, 128, 0],
        "trainId": 4,
    },
    {
        "hasInstances": False,
        "category": "vegetation",
        "catid": 4,
        "name": "Vegetation",
        "ignoreInEval": False,
        "id": 5,
        "color": [128, 128, 0],
        "trainId": 5,
    },
    {
        "hasInstances": False,
        "category": "person",
        "catid": 5,
        "name": "Human",
        "ignoreInEval": False,
        "id": 6,
        "color": [64, 64, 0],
        "trainId": 6,
    },
    {
        "hasInstances": False,
        "category": "vehicle",
        "catid": 3,
        "name": "Moving Car",
        "ignoreInEval": False,
        "id": 7,
        "color": [64, 0, 128],
        "trainId": 7,
    },
]


@pytest.fixture()
def labels_info() -> list[dict]:
    return MINIMAL_INFO


@pytest.fixture()
def colour_map(labels_info):
    return build_colour_map(labels_info)


@pytest.fixture()
def lut(colour_map):
    return build_lut(colour_map)


@pytest.fixture()
def info_json_path(tmp_path) -> Path:
    p = tmp_path / "UAVid_info.json"
    p.write_text(json.dumps(MINIMAL_INFO))
    return p


# ---------------------------------------------------------------------------
# build_colour_map tests
# ---------------------------------------------------------------------------


class TestBuildColourMap:
    def test_clutter_maps_to_ignore(self, colour_map):
        """Clutter (ignoreInEval=True) must map to IGNORE_LABEL (255)."""
        assert colour_map[(0, 0, 0)] == IGNORE_LABEL

    def test_building_maps_to_zero(self, colour_map):
        """Building is the first eval class and should get YOLO id 0."""
        assert colour_map[(128, 0, 0)] == 0

    def test_road_maps_to_one(self, colour_map):
        """Road is the second eval class; id 1."""
        assert colour_map[(128, 64, 128)] == 1

    def test_moving_car_maps_to_six(self, colour_map):
        """Moving Car is the last eval class (trainId 7); should get id 6."""
        assert colour_map[(64, 0, 128)] == 6

    def test_all_eval_classes_have_unique_ids(self, colour_map):
        """No two eval colours should share the same YOLO class id."""
        ids = [v for v in colour_map.values() if v != IGNORE_LABEL]
        assert len(ids) == len(set(ids))

    def test_number_of_eval_classes(self, colour_map):
        """7 eval classes (all 8 minus Clutter)."""
        eval_ids = [v for v in colour_map.values() if v != IGNORE_LABEL]
        assert len(eval_ids) == 7


# ---------------------------------------------------------------------------
# build_lut tests
# ---------------------------------------------------------------------------


class TestBuildLUT:
    def test_lut_shape(self, lut):
        assert lut.shape == (256, 256, 256)

    def test_lut_dtype(self, lut):
        assert lut.dtype == np.uint8

    def test_lut_clutter_entry(self, lut):
        assert lut[0, 0, 0] == IGNORE_LABEL

    def test_lut_building_entry(self, lut):
        assert lut[128, 0, 0] == 0

    def test_lut_road_entry(self, lut):
        assert lut[128, 64, 128] == 1

    def test_unknown_colour_defaults_to_ignore(self, lut):
        """Any colour not in the palette must default to 255."""
        assert lut[1, 2, 3] == IGNORE_LABEL


# ---------------------------------------------------------------------------
# convert_mask tests
# ---------------------------------------------------------------------------


class TestConvertMask:
    def _make_rgb_mask(
        self, colours: list[tuple[int, int, int]], size=(4, 4)
    ) -> Image.Image:
        """Create a small RGB PIL image with blocks of specified colours."""
        h, w = size
        n = len(colours)
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        for i, colour in enumerate(colours):
            col_start = i * (w // n)
            col_end = (i + 1) * (w // n) if i < n - 1 else w
            arr[:, col_start:col_end] = colour
        return Image.fromarray(arr)

    def test_building_pixels_become_zero(self, lut, tmp_path):
        img = self._make_rgb_mask([(128, 0, 0)])
        src = tmp_path / "label.png"
        dst = tmp_path / "mask.png"
        img.save(src)
        convert_mask(src, dst, lut)
        result = np.array(Image.open(dst))
        assert (
            result == 0
        ).all(), f"Expected all 0, got unique values: {np.unique(result)}"

    def test_clutter_pixels_become_255(self, lut, tmp_path):
        img = self._make_rgb_mask([(0, 0, 0)])
        src = tmp_path / "clutter.png"
        dst = tmp_path / "mask.png"
        img.save(src)
        convert_mask(src, dst, lut)
        result = np.array(Image.open(dst))
        assert (result == 255).all()

    def test_unknown_colour_becomes_ignore(self, lut, tmp_path):
        """Pixels with an unrecognised colour must be treated as ignore."""
        img = self._make_rgb_mask([(7, 8, 9)])  # not in palette
        src = tmp_path / "unknown.png"
        dst = tmp_path / "mask.png"
        img.save(src)
        convert_mask(src, dst, lut)
        result = np.array(Image.open(dst))
        assert (result == 255).all()

    def test_mixed_classes_correct_ids(self, lut, tmp_path):
        """A mask with Building + Clutter should produce ids 0 and 255."""
        colours = [(128, 0, 0), (0, 0, 0)]  # Building, Clutter
        img = self._make_rgb_mask(colours)
        src = tmp_path / "mixed.png"
        dst = tmp_path / "mask.png"
        img.save(src)
        convert_mask(src, dst, lut)
        result = np.array(Image.open(dst))
        unique = set(result.flatten().tolist())
        assert unique == {0, 255}

    def test_output_is_single_channel(self, lut, tmp_path):
        img = self._make_rgb_mask([(128, 0, 0)])
        src = tmp_path / "src.png"
        dst = tmp_path / "dst.png"
        img.save(src)
        convert_mask(src, dst, lut)
        out = Image.open(dst)
        assert out.mode == "L", f"Expected mode L, got {out.mode}"

    def test_dry_run_does_not_write(self, lut, tmp_path):
        img = self._make_rgb_mask([(128, 0, 0)])
        src = tmp_path / "src.png"
        dst = tmp_path / "dst.png"
        img.save(src)
        convert_mask(src, dst, lut, dry_run=True)
        assert not dst.exists()

    def test_all_seven_eval_classes_round_trip(self, lut, colour_map, tmp_path):
        """Each of the 7 eval colours should survive a full round-trip."""
        eval_colours = [(c, v) for c, v in colour_map.items() if v != IGNORE_LABEL]
        n = len(eval_colours)
        w, h = n * 4, 4
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        expected_ids = []
        for i, (colour, cls_id) in enumerate(eval_colours):
            arr[:, i * 4 : (i + 1) * 4] = colour
            expected_ids.extend([cls_id] * 4)

        src = tmp_path / "all_classes.png"
        dst = tmp_path / "all_classes_mask.png"
        Image.fromarray(arr).save(src)
        convert_mask(src, dst, lut)

        result = np.array(Image.open(dst))  # (h, w)
        actual_ids = result[0, :].tolist()
        assert actual_ids == expected_ids


# ---------------------------------------------------------------------------
# get_yolo_class_names tests
# ---------------------------------------------------------------------------


class TestGetYoloClassNames:
    def test_seven_classes_returned(self, labels_info):
        names = get_yolo_class_names(labels_info)
        assert len(names) == 7

    def test_clutter_not_in_names(self, labels_info):
        names = get_yolo_class_names(labels_info)
        assert "Clutter" not in names.values()

    def test_building_is_class_zero(self, labels_info):
        names = get_yolo_class_names(labels_info)
        assert names[0] == "Building"

    def test_ids_are_consecutive(self, labels_info):
        names = get_yolo_class_names(labels_info)
        assert list(names.keys()) == list(range(len(names)))


# ---------------------------------------------------------------------------
# load_labels_info tests
# ---------------------------------------------------------------------------


class TestLoadLabelsInfo:
    def test_loads_eight_entries(self, info_json_path):
        info = load_labels_info(info_json_path)
        assert len(info) == 8

    def test_real_info_file(self):
        """Smoke-test against the actual project UAVid_info.json."""
        real_path = Path("configs/UAVid_info.json")
        if not real_path.exists():
            pytest.skip("UAVid_info.json not found (running outside project root)")
        info = load_labels_info(real_path)
        assert len(info) == 8
        clutter = next(c for c in info if c["name"] == "Clutter")
        assert clutter["ignoreInEval"] is True
        assert clutter["color"] == [0, 0, 0]
