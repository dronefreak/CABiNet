"""Tests for src/scripts/train_yolo.py

Covers:
  - SUPPORTED_SEMANTIC_MODELS: all expected yolo26*-sem variants present
  - _resolve_dataset_path: absolute path passthrough, missing file raises FileNotFoundError
  - _resolve_resume_weights: returns None when resume=False or last.pt absent
  - _build_train_kwargs: key mapping, task='semantic', aug keys flattened
  - _build_val_kwargs: key mapping, task='semantic', split fixed to 'val'
  - Hydra config: train_yolo.yaml loads without error; model override selects correct variant
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf

from src.scripts.train_yolo import (
    REPO_ROOT,
    SUPPORTED_SEMANTIC_MODELS,
    _build_train_kwargs,
    _build_val_kwargs,
    _resolve_dataset_path,
    _resolve_experiments_path,
    _resolve_resume_weights,
)

# ── SUPPORTED_SEMANTIC_MODELS ─────────────────────────────────────────────────


def test_supported_models_contains_all_yolo26_sem():
    for size in ("n", "s", "m", "l", "x"):
        assert f"yolo26{size}-sem" in SUPPORTED_SEMANTIC_MODELS


def test_supported_models_excludes_seg_variants():
    # '-seg' models are instance segmentation ('segment' task); not valid here.
    for entry in SUPPORTED_SEMANTIC_MODELS:
        assert "-seg" not in entry, f"Instance-seg model found unexpectedly: {entry}"


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


# ── _resolve_experiments_path ─────────────────────────────────────────────────


def test_resolve_experiments_path_relative_anchored_to_repo_root():
    result = _resolve_experiments_path("runs/uavid_yolo")
    assert result.is_absolute()
    assert result == REPO_ROOT / "runs" / "uavid_yolo"


def test_resolve_experiments_path_absolute_passthrough(tmp_path):
    result = _resolve_experiments_path(str(tmp_path))
    assert result == tmp_path


def test_resolve_experiments_path_independent_of_cwd(tmp_path, monkeypatch):
    """Regression test: a relative experiments_path must not resolve relative
    to whatever directory the shell happens to be in at invocation time (this
    was the mechanism — combined with Ultralytics' global 'runs_dir' fallback
    for relative paths — that sent training output into an unrelated repo)."""
    monkeypatch.chdir(tmp_path)
    result = _resolve_experiments_path("runs/uavid_yolo")
    assert result == REPO_ROOT / "runs" / "uavid_yolo"
    assert str(tmp_path) not in str(result)


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
    cfg = _make_cfg(resume=False, experiments_path=str(tmp_path), run_name="yolo26n")
    assert _resolve_resume_weights(cfg) is None


def test_resolve_resume_weights_true_missing_last_pt_returns_none(tmp_path):
    cfg = _make_cfg(resume=True, experiments_path=str(tmp_path), run_name="yolo26n")
    assert _resolve_resume_weights(cfg) is None


def test_resolve_resume_weights_true_existing_last_pt(tmp_path):
    weights_dir = tmp_path / "yolo26n" / "weights"
    weights_dir.mkdir(parents=True)
    last_pt = weights_dir / "last.pt"
    last_pt.write_bytes(b"fake")
    cfg = _make_cfg(resume=True, experiments_path=str(tmp_path), run_name="yolo26n")
    result = _resolve_resume_weights(cfg)
    assert result == str(last_pt)


# ── _build_train_kwargs ───────────────────────────────────────────────────────


@pytest.fixture()
def minimal_cfg():
    """Minimal Hydra-like DictConfig for _build_train_kwargs."""
    raw = {
        "model": {"run_name": "yolo26n"},
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


def test_build_train_kwargs_task_is_semantic(minimal_cfg, tmp_path):
    dataset = tmp_path / "uavid_yolo.yaml"
    dataset.write_text("nc: 8\n")
    kwargs = _build_train_kwargs(minimal_cfg, dataset)
    assert kwargs["task"] == "semantic"
    assert "overlap_mask" not in kwargs
    assert "mask_ratio" not in kwargs


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
    # project is resolved to an absolute, repo-anchored path (see
    # _resolve_experiments_path) so it can't be silently redirected by a
    # relative-path fallback onto Ultralytics' global 'runs_dir' setting.
    assert Path(kwargs["project"]).is_absolute()
    assert Path(kwargs["project"]).name == "test"
    assert kwargs["name"] == "yolo26n"


# ── _build_val_kwargs ─────────────────────────────────────────────────────────


def test_build_val_kwargs_split_fixed(minimal_cfg, tmp_path):
    dataset = tmp_path / "uavid_yolo.yaml"
    dataset.write_text("nc: 8\n")
    # patch validation_config into cfg
    OmegaConf.update(
        minimal_cfg,
        "validation_config",
        {"batch_size": 1, "save_json": True, "augment": False},
    )
    kwargs = _build_val_kwargs(minimal_cfg, dataset, "best.pt")
    assert kwargs["split"] == "val"
    assert kwargs["task"] == "semantic"
    assert "overlap_mask" not in kwargs
    assert kwargs["batch"] == 1


# ── Hydra config composition ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model_override,expected_name,expected_model",
    [
        (None, "yolo26n", "yolo26n-sem.pt"),  # default
        ("yolo26n-sem", "yolo26n", "yolo26n-sem.pt"),
        ("yolo26s-sem", "yolo26s", "yolo26s-sem.pt"),
        ("yolo26m-sem", "yolo26m", "yolo26m-sem.pt"),
        ("yolo26l-sem", "yolo26l", "yolo26l-sem.pt"),
        ("yolo26x-sem", "yolo26x", "yolo26x-sem.pt"),
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


def test_hydra_config_no_segment_only_keys():
    """overlap_mask/mask_ratio are 'segment'-task-only and must not linger."""
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="train_yolo")
    assert "overlap_mask" not in cfg.training_config
    assert "mask_ratio" not in cfg.training_config
    assert "overlap_mask" not in cfg.validation_config


def test_hydra_config_experiments_path_resolves():
    """Regression test: experiments_path previously referenced a non-existent
    interpolation key (${yolo.model.model}) and raised at access time.
    Only checks the interpolation resolves cleanly, not the literal value —
    that's a tunable config detail, not part of this contract."""
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="train_yolo")
    assert isinstance(cfg.training_config.experiments_path, str)
    assert "uavid_yolo" in cfg.training_config.experiments_path


def test_hydra_config_dotted_overrides():
    """The Hydra dotted-path override style (training_config.epochs=...) must work."""
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(
            config_name="train_yolo",
            overrides=["training_config.epochs=200", "training_config.batch_size=8"],
        )
    assert cfg.training_config.epochs == 200
    assert cfg.training_config.batch_size == 8


def test_hydra_config_mode_defaults_to_train():
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="train_yolo")
    assert cfg.mode == "train"
