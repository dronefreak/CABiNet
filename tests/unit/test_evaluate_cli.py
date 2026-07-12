"""Tests for the standalone evaluate.py CLI.

Covers:
  - _load_model_weights: raw state_dict passthrough vs full-checkpoint extraction
  - Hydra config: evaluate.yaml loads without error, checkpoint_path is mandatory,
    defaults match documented behaviour
"""

from __future__ import annotations

import pytest
import torch
from hydra import compose, initialize
from omegaconf.errors import MissingMandatoryValue

from src.scripts.evaluate import _load_model_weights

# ── _load_model_weights ───────────────────────────────────────────────────────


def test_load_model_weights_raw_state_dict_passthrough(tmp_path):
    """A raw model state_dict (as saved for *_best.pth / the final .pth) must
    be returned as-is."""
    state_dict = {"conv.weight": torch.randn(3, 3)}
    ckpt_path = tmp_path / "raw.pth"
    torch.save(state_dict, ckpt_path)

    loaded = _load_model_weights(ckpt_path, torch.device("cpu"))

    assert set(loaded.keys()) == {"conv.weight"}
    assert torch.allclose(loaded["conv.weight"], state_dict["conv.weight"])


def test_load_model_weights_extracts_from_full_checkpoint(tmp_path):
    """A full training checkpoint (checkpoint_last.pth) wraps weights under
    'model_state' alongside optimizer/EMA/scaler state — only the weights
    should be returned."""
    state_dict = {"conv.weight": torch.randn(3, 3)}
    full_checkpoint = {
        "epoch": 5,
        "model_state": state_dict,
        "optimizer_state": {},
        "ema_state": {},
    }
    ckpt_path = tmp_path / "checkpoint_last.pth"
    torch.save(full_checkpoint, ckpt_path)

    loaded = _load_model_weights(ckpt_path, torch.device("cpu"))

    assert set(loaded.keys()) == {"conv.weight"}
    assert torch.allclose(loaded["conv.weight"], state_dict["conv.weight"])


def test_load_model_weights_raw_dict_without_model_state_key_untouched(tmp_path):
    """A raw state_dict that happens not to contain a 'model_state' key must
    not be misidentified as a full checkpoint."""
    state_dict = {"linear.weight": torch.randn(2, 2), "linear.bias": torch.randn(2)}
    ckpt_path = tmp_path / "raw2.pth"
    torch.save(state_dict, ckpt_path)

    loaded = _load_model_weights(ckpt_path, torch.device("cpu"))

    assert set(loaded.keys()) == {"linear.weight", "linear.bias"}


# ── Hydra config composition ───────────────────────────────────────────────────


def test_hydra_config_requires_checkpoint_path():
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="evaluate")
    with pytest.raises(MissingMandatoryValue):
        _ = cfg.checkpoint_path


def test_hydra_config_defaults():
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="evaluate", overrides=["checkpoint_path=/tmp/x.pth"])
    assert cfg.split == "val"
    assert cfg.dataset.name == "cityscapes"
    assert cfg.validation_config.batch_size == 2
    assert cfg.validation_config.flip is True
    assert cfg.device == "cuda"


def test_hydra_config_dataset_override():
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(
            config_name="evaluate",
            overrides=["checkpoint_path=/tmp/x.pth", "dataset=uavid"],
        )
    assert cfg.dataset.name == "uavid"
    assert cfg.dataset.num_classes == 8


def test_hydra_config_split_override():
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(
            config_name="evaluate",
            overrides=["checkpoint_path=/tmp/x.pth", "split=test"],
        )
    assert cfg.split == "test"
