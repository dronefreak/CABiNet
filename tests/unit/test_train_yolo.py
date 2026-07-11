"""Tests for src/scripts/train_yolo.py

Covers:
  - SUPPORTED_SEG_MODELS: all expected variants present
  - _resolve_dataset_path: absolute path passthrough, missing file raises FileNotFoundError
  - _resolve_resume_weights: returns None when resume=False or last.pt absent
  - _build_train_kwargs: key mapping, overlap_mask always False, aug keys flattened
  - _build_val_kwargs: key mapping, split fixed to 'val'
  - Hydra config: train_yolo.yaml loads without error; model override selects correct variant
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf

from src.scripts.train_yolo import (
    SUPPORTED_SEG_MODELS,
    _build_train_kwargs,
    _build_val_kwargs,
    _resolve_dataset_path,
    _resolve_resume_weights,
)

# ── SUPPORTED_SEG_MODELS ──────────────────────────────────────────────────────


def test_supported_models_contains_all_yolov8():
    for size in ("n", "s", "m", "l", "x"):
        assert f"yolov8{size}-seg" in SUPPORTED_SEG_MODELS


def test_supported_models_contains_yolov9_seg():
    assert "yolov9c-seg" in SUPPORTED_SEG_MODELS
    assert "yolov9e-seg" in SUPPORTED_SEG_MODELS


def test_supported_models_contains_all_yolo11():
    for size in ("n", "s", "m", "l", "x"):
        assert f"yolo11{size}-seg" in SUPPORTED_SEG_MODELS


def test_supported_models_excludes_detection_only():
    # YOLOv10 has no seg variants in Ultralytics
    for entry in SUPPORTED_SEG_MODELS:
        assert "v10" not in entry, f"YOLOv10 model found unexpectedly: {entry}"


# ── _resolve_dataset_path ─────────────────────────────────────────────────────


def test_resolve_dataset_path_absolute_passthrough(tmp_path):
    cfg_file = tmp_path / "uavid_yolo.yaml"
    cfg_file.write_text("nc: 7\n")
    result = _resolve_dataset_path(str(cfg_file))
    assert result == cfg_file.resolve()


def test_resolve_dataset_path_relative(tmp_path, monkeypatch):
    cfg_file = tmp_path / "uavid_yolo.yaml"
    cfg_file.write_text("nc: 7\n")
    monkeypatch.chdir(tmp_path)
    result = _resolve_dataset_path("uavid_yolo.yaml")
    assert result.exists()


def test_resolve_dataset_path_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset config not found"):
        _resolve_dataset_path(str(tmp_path / "nonexistent.yaml"))


# ── _resolve_resume_weights ───────────────────────────────────────────────────


def _make_cfg(resume: bool, experiments_path: str, run_name: str) -> MagicMock:
    cfg = MagicMock()
    cfg.training_config.get.side_effect = lambda key, default=None: (
        resume if key == "resume" else default
    )
    cfg.training_config.experiments_path = experiments_path
    cfg.model.run_name = run_name
    return cfg


def test_resolve_resume_weights_false_returns_none(tmp_path):
    cfg = _make_cfg(resume=False, experiments_path=str(tmp_path), run_name="yolov8m")
    assert _resolve_resume_weights(cfg) is None


def test_resolve_resume_weights_true_missing_last_pt_returns_none(tmp_path):
    cfg = _make_cfg(resume=True, experiments_path=str(tmp_path), run_name="yolov8m")
    assert _resolve_resume_weights(cfg) is None


def test_resolve_resume_weights_true_existing_last_pt(tmp_path):
    weights_dir = tmp_path / "yolov8m" / "weights"
    weights_dir.mkdir(parents=True)
    last_pt = weights_dir / "last.pt"
    last_pt.write_bytes(b"fake")
    cfg = _make_cfg(resume=True, experiments_path=str(tmp_path), run_name="yolov8m")
    result = _resolve_resume_weights(cfg)
    assert result == str(last_pt)


# ── _build_train_kwargs ───────────────────────────────────────────────────────


@pytest.fixture()
def minimal_cfg():
    """Minimal Hydra-like DictConfig for _build_train_kwargs."""
    raw = {
        "model": {"run_name": "yolov8m"},
        "training_config": {
            "epochs": 10,
            "batch_size": 4,
            "imgsz": 512,
            "nbs": 64,
            "amp": True,
            "optimizer": "SGD",
            "optimizer_lr_start": 0.01,
            "lrf": 0.01,
            "optimizer_momentum": 0.937,
            "optimizer_weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "cos_lr": True,
            "patience": 30,
            "cls_pw": 0.5,
            "overlap_mask": False,
            "mask_ratio": 1,
            "save_period": 10,
            "num_workers": 4,
            "resume": False,
            "exist_ok": False,
            "experiments_path": "runs/test",
            "augmentation": {
                "degrees": 10.0,
                "flipud": 0.2,
                "fliplr": 0.5,
                "mosaic": 0.8,
                "close_mosaic": 15,
            },
        },
        "runtime": {"seed": 0, "device": 0, "deterministic": True},
    }
    return OmegaConf.create(raw)


def test_build_train_kwargs_overlap_mask_always_false(minimal_cfg, tmp_path):
    dataset = tmp_path / "uavid_yolo.yaml"
    dataset.write_text("nc: 7\n")
    kwargs = _build_train_kwargs(minimal_cfg, dataset)
    assert kwargs["overlap_mask"] is False


def test_build_train_kwargs_augmentation_flattened(minimal_cfg, tmp_path):
    dataset = tmp_path / "uavid_yolo.yaml"
    dataset.write_text("nc: 7\n")
    kwargs = _build_train_kwargs(minimal_cfg, dataset)
    assert kwargs["degrees"] == 10.0
    assert kwargs["flipud"] == 0.2
    assert kwargs["mosaic"] == 0.8
    assert "augmentation" not in kwargs


def test_build_train_kwargs_epoch_passthrough(minimal_cfg, tmp_path):
    dataset = tmp_path / "uavid_yolo.yaml"
    dataset.write_text("nc: 7\n")
    kwargs = _build_train_kwargs(minimal_cfg, dataset)
    assert kwargs["epochs"] == 10
    assert kwargs["batch"] == 4
    assert kwargs["imgsz"] == 512


def test_build_train_kwargs_project_and_name(minimal_cfg, tmp_path):
    dataset = tmp_path / "uavid_yolo.yaml"
    dataset.write_text("nc: 7\n")
    kwargs = _build_train_kwargs(minimal_cfg, dataset)
    assert kwargs["project"] == "runs/test"
    assert kwargs["name"] == "yolov8m"


# ── _build_val_kwargs ─────────────────────────────────────────────────────────


def test_build_val_kwargs_split_fixed(minimal_cfg, tmp_path):
    dataset = tmp_path / "uavid_yolo.yaml"
    dataset.write_text("nc: 7\n")
    # patch validation_config into cfg
    OmegaConf.update(
        minimal_cfg,
        "validation_config",
        {"batch_size": 1, "overlap_mask": False, "save_json": True, "augment": False},
    )
    kwargs = _build_val_kwargs(minimal_cfg, dataset, "best.pt")
    assert kwargs["split"] == "val"
    assert kwargs["overlap_mask"] is False
    assert kwargs["batch"] == 1


# ── Hydra config composition ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model_override,expected_name,expected_model",
    [
        (None, "yolov8m", "yolov8m-seg.pt"),          # default
        ("yolov8n-seg", "yolov8n", "yolov8n-seg.pt"),
        ("yolov8x-seg", "yolov8x", "yolov8x-seg.pt"),
        ("yolov9c-seg", "yolov9c", "yolov9c-seg.pt"),
        ("yolov9e-seg", "yolov9e", "yolov9e-seg.pt"),
        ("yolo11n-seg", "yolo11n", "yolo11n-seg.pt"),
        ("yolo11m-seg", "yolo11m", "yolo11m-seg.pt"),
        ("yolo11x-seg", "yolo11x", "yolo11x-seg.pt"),
    ],
)
def test_hydra_config_model_override(model_override, expected_name, expected_model):
    overrides = []
    if model_override:
        overrides.append(f"yolo/model@model={model_override}")
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="train_yolo", overrides=overrides)
    assert cfg.model.run_name == expected_name
    assert cfg.model.model_name == expected_model


def test_hydra_config_default_overlap_mask_false():
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="train_yolo")
    assert cfg.training_config.overlap_mask is False


def test_hydra_config_mode_defaults_to_train():
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="train_yolo")
    assert cfg.mode == "train"
