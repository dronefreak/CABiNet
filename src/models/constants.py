"""Constants and configuration for CABiNet models."""

from typing import Any, Dict

# MobileNetV3 backbone configuration
MOBILENET_LARGE_FEATURES = 960
MOBILENET_SMALL_FEATURES = 576

# Model output channel configuration
MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    "large": {
        "attention_planes": MOBILENET_LARGE_FEATURES,
        "output_channel": 1280,
    },
    "small": {
        "attention_planes": MOBILENET_SMALL_FEATURES,
        "output_channel": 1024,
    },
}

# Training constants
OHEM_DIVISOR = 16  # Divisor for computing n_min in OHEM loss
DEFAULT_SCORE_THRESHOLD = 0.7  # Default threshold for OHEM loss

# Evaluation constants
EVAL_STRIDE_RATE = 5 / 6.0  # Stride rate for sliding window evaluation
DEFAULT_EVAL_SCALES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

# Dataset-specific constants
CITYSCAPES_NUM_CLASSES = 19
UAVID_NUM_CLASSES = 8
DEFAULT_IGNORE_LABEL = 255

# Visualization constants
VISUALIZATION_SAMPLE_LIMIT = 50  # Maximum number of samples to visualize
