"""Pytest configuration and shared fixtures."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.cabinet import CABiNet


@pytest.fixture
def device():
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_image():
    """Create a sample RGB image tensor."""
    return torch.randn(1, 3, 512, 512)


@pytest.fixture
def sample_label():
    """Create a sample label tensor."""
    return torch.randint(0, 19, (1, 512, 512))


@pytest.fixture
def batch_image():
    """Create a batch of RGB images."""
    return torch.randn(4, 3, 512, 512)


@pytest.fixture
def batch_label():
    """Create a batch of labels."""
    return torch.randint(0, 19, (4, 512, 512))


@pytest.fixture
def num_classes():
    """Return number of classes for Cityscapes."""
    return 19


@pytest.fixture
def project_root():
    """Return path to project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    seed_value = 42
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed_value


@pytest.fixture
def mock_config():
    """Create a mock configuration dictionary."""
    return {
        "dataset": {
            "name": "cityscapes",
            "num_classes": 19,
            "ignore_idx": 255,
            "cropsize": [512, 512],
            "seed": 42,
        },
        "model": {
            "mode": "large",
            "pretrained_weights": None,
        },
        "training_config": {
            "batch_size": 4,
            "num_workers": 2,
            "epochs": 1,
            "optimizer_lr_start": 1e-3,
            "optimizer_momentum": 0.9,
            "optimizer_weight_decay": 5e-4,
            "class_balancing": False,
        },
    }


@pytest.fixture
def mock_small_model():
    """Factory fixture to create CABiNet models with configurable mode and number of classes."""

    def _make_model(num_classes=19, mode="small"):
        # Define model configuration: [k, t, c, SE, HS, s]
        cfgs = [
            [3, 1, 16, 1, 0, 2],
            [3, 4.5, 24, 0, 0, 2],
            [3, 3.67, 24, 0, 0, 1],
            [5, 4, 40, 1, 1, 2],
            [5, 6, 40, 1, 1, 1],
            [5, 6, 40, 1, 1, 1],
            [5, 3, 48, 1, 1, 1],
            [5, 3, 48, 1, 1, 1],
            [5, 6, 96, 1, 1, 2],
            [5, 6, 96, 1, 1, 1],
            [5, 6, 96, 1, 1, 1],
        ]
        model = CABiNet(cfgs=cfgs, n_classes=num_classes, mode=mode)
        return model

    return _make_model


@pytest.fixture
def mock_large_model():
    """Factory fixture to create CABiNet models with configurable mode and number of classes."""

    def _make_model(num_classes=19, mode="large"):
        # Define model configuration: [k, t, c, SE, HS, s]
        cfgs = [
            # k, t, c, SE, HS, s
            [3, 1, 16, 0, 0, 1],
            [3, 4, 24, 0, 0, 2],
            [3, 3, 24, 0, 0, 1],
            [5, 3, 40, 1, 0, 2],
            [5, 3, 40, 1, 0, 1],
            [5, 3, 40, 1, 0, 1],
            [3, 6, 80, 0, 1, 2],
            [3, 2.5, 80, 0, 1, 1],
            [3, 2.3, 80, 0, 1, 1],
            [3, 2.3, 80, 0, 1, 1],
            [3, 6, 112, 1, 1, 1],
            [3, 6, 112, 1, 1, 1],
            [5, 6, 160, 1, 1, 2],
            [5, 6, 160, 1, 1, 1],
            [5, 6, 160, 1, 1, 1],
        ]
        model = CABiNet(cfgs=cfgs, n_classes=num_classes, mode=mode)
        return model

    return _make_model
