"""Dataset registry shared by train.py and evaluate.py.

Kept separate from both scripts (rather than defined in train.py and
imported by evaluate.py) to avoid a circular import — train.py already
imports MscEvalV0 from evaluate.py.
"""

from src.datasets.aeroscapes import AeroScapes
from src.datasets.cityscapes import CityScapes
from src.datasets.uavid import UAVid
from src.datasets.vdd import VDD

DATASET_REGISTRY = {
    "cityscapes": CityScapes,
    "uavid": UAVid,
    "aeroscapes": AeroScapes,
    "vdd": VDD,
}

# Per-dataset constructor kwargs. UAVid/AeroScapes/VDD consume a
# pre-converted images/+masks/ directory (see convert_uavid_to_yolo.py /
# convert_aeroscapes_to_yolo.py / convert_vdd_to_yolo.py) and no longer need
# a colour-palette config_file, unlike CityScapes — the two datasets have
# genuinely different constructors, so they can't share one flat kwargs dict.
DATASET_KWARGS_BUILDERS = {
    "cityscapes": lambda cfg, ignore_idx, cropsize: dict(
        config_file=cfg.dataset.config_file,
        ignore_lb=ignore_idx,
        rootpth=cfg.dataset.dataset_path,
        cropsize=cropsize,
    ),
    "uavid": lambda cfg, ignore_idx, cropsize: dict(
        ignore_lb=ignore_idx,
        rootpth=cfg.dataset.dataset_path,
        cropsize=cropsize,
        augmentation=cfg.dataset.get("augmentation"),
    ),
    "aeroscapes": lambda cfg, ignore_idx, cropsize: dict(
        ignore_lb=ignore_idx,
        rootpth=cfg.dataset.dataset_path,
        cropsize=cropsize,
        augmentation=cfg.dataset.get("augmentation"),
    ),
    "vdd": lambda cfg, ignore_idx, cropsize: dict(
        ignore_lb=ignore_idx,
        rootpth=cfg.dataset.dataset_path,
        cropsize=cropsize,
        augmentation=cfg.dataset.get("augmentation"),
    ),
}
