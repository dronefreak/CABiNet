# CABiNet: Efficient Context Aggregation Network for Semantic Segmentation

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![CI](https://github.com/dronefreak/CABiNet/actions/workflows/ci.yml/badge.svg)](https://github.com/dronefreak/CABiNet/actions/workflows/ci.yml)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![Maintained](https://img.shields.io/badge/Maintained-yes-2ea44f.svg)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](.github/CONTRIBUTING.md)

CABiNet (Context Aggregation Network) is a dual-branch convolutional neural network designed for real-time semantic segmentation with significantly lower computational costs compared to state-of-the-art methods while maintaining competitive accuracy. The architecture is specifically optimized for autonomous systems and real-time applications.

<p align="center">
  <img src="assets/showcase_mosaic.gif" alt="UAVid Semantic Segmentation Demo">
</p>

## Architecture

The CABiNet architecture employs a dual-branch design that balances spatial detail preservation and contextual understanding:

- **Spatial Branch**: Maintains high-resolution features for precise boundary detection
- **Context Branch**: Lightweight global aggregation and local distribution blocks for capturing long-range and local dependencies
- **Feature Fusion Module (FFM)**: Normalizes and selects optimal features for scene segmentation
- **Deep Supervision**: Bottleneck in context branch enables better representational learning

![CABiNet Architecture](imgs/cabinet.jpg)

## Results

### Cityscapes Dataset

Comparison of semantic segmentation results on the Cityscapes validation set:

![Cityscapes Results](imgs/citys.jpg)

_From top to bottom: Input RGB images, SwiftNet predictions, CABiNet predictions (red boxes highlight improvements), ground truth_

### UAVid Dataset

Performance on the UAVid validation set for aerial imagery:

![UAVid Results](imgs/uavid_r.jpg)

_Columns: Input images, State-of-the-art predictions, CABiNet predictions (white boxes show improvements)_

## UAVid Model Zoo

CABiNet and Ultralytics YOLO26 (`semantic` task, dense per-pixel — not `-seg`) are trained and evaluated under one shared UAVid pipeline (`images/`+`masks/` format, 8 classes incl. Clutter).

All numbers below are test-split mIoU (not val — see [`train_yolo.py`](src/scripts/train_yolo.py)'s `validation_config.split`). Params/FLOPs are architecture-only, measured at 1024×1024 via `thop` (CABiNet) / Ultralytics' built-in profiler (YOLO26, also `thop`-backed — both report FLOPs as 2× MACs, so the two families are directly comparable).

| Model                       | mIoU (%) | Params (M) | FLOPs (GFLOPs) | HF Weights                                                                    |
| --------------------------- | -------- | ---------- | -------------- | ----------------------------------------------------------------------------- |
| CABiNet (MobileNetV3-Large) | 68.60    | 9.17       | 54.8           | [HF Model](https://huggingface.co/dronefreak/cabinet-mobilenetv3-large-uavid) |
| CABiNet (MobileNetV3-Small) | 66.84    | 5.36       | 44.1           | [HF Model](https://huggingface.co/dronefreak/cabinet-mobilenetv3-small-uavid) |
| YOLO26x-sem                 | 64.41    | 40.16      | 430.9          | [HF Model](https://huggingface.co/dronefreak/uavid-yolo26x-sem)               |
| YOLO26l-sem                 | 63.28    | 17.87      | 192.4          | [HF Model](https://huggingface.co/dronefreak/uavid-yolo26l-sem)               |
| YOLO26m-sem                 | 61.98    | 14.32      | 152.3          | [HF Model](https://huggingface.co/dronefreak/uavid-yolo26m-sem)               |
| YOLO26s-sem                 | 61.69    | 6.50       | 44.4           | [HF Model](https://huggingface.co/dronefreak/uavid-yolo26s-sem)               |
| YOLO26n-sem                 | 58.17    | 1.63       | 11.4           | [HF Model](https://huggingface.co/dronefreak/uavid-yolo26n-sem)               |

CABiNet (both backbones) outperforms **every** YOLO26-sem variant on UAVid, including the largest (YOLO26x) — and does so at a fraction of the compute: CABiNet-Large (54.8 GFLOPs) beats YOLO26x (430.9 GFLOPs, ~8× more) and YOLO26m (152.3 GFLOPs, ~3× more) on both mIoU and FLOPs simultaneously. CABiNet-Small also improves substantially over the numbers originally reported in the [CABiNet paper](#citation) on this dataset.

## VDD Model Zoo

CABiNet and YOLO26-sem are also trained and evaluated under one shared [VDD (Varied Drone Dataset)](https://github.com/RussRobin/VDD) pipeline (`images/`+`masks/` format, 7 classes). VDD is a much smaller dataset than UAVid (280 train images) spanning varied altitudes and viewpoints; heavier augmentation (mosaic/mixup/copy-paste) is used to help offset the small training set — see the [VDD Dataset](#vdd-dataset) section below.

All numbers below are test-split mIoU, same methodology as the [UAVid Model Zoo](#uavid-model-zoo) table above (Params/FLOPs are architecture-only and identical across datasets, since both are measured at 1024×1024 regardless of what the model was finetuned on).

| Model                       | mIoU (%) | Params (M) | FLOPs (GFLOPs) | HF Weights                                                                  |
| --------------------------- | -------- | ---------- | -------------- | --------------------------------------------------------------------------- |
| YOLO26x-sem                 | 78.83    | 40.16      | 430.9          | [HF Model](https://huggingface.co/dronefreak/vdd-yolo26x-sem)               |
| YOLO26l-sem                 | 78.57    | 17.87      | 192.4          | [HF Model](https://huggingface.co/dronefreak/vdd-yolo26l-sem)               |
| CABiNet (MobileNetV3-Large) | 77.76    | 9.17       | 54.8           | [HF Model](https://huggingface.co/dronefreak/cabinet-mobilenetv3-large-vdd) |
| YOLO26m-sem                 | 77.02    | 14.32      | 152.3          | [HF Model](https://huggingface.co/dronefreak/vdd-yolo26m-sem)               |
| YOLO26s-sem                 | 76.35    | 6.50       | 44.4           | [HF Model](https://huggingface.co/dronefreak/vdd-yolo26s-sem)               |
| YOLO26n-sem                 | 73.99    | 1.63       | 11.4           | [HF Model](https://huggingface.co/dronefreak/vdd-yolo26n-sem)               |

Unlike UAVid, the two largest YOLO26-sem variants (x, l) edge out CABiNet-Large on raw mIoU here, by a small margin (≤1.1 pts) — VDD's much smaller training set and different 7-class scheme make this a genuine dataset effect, not a regression. CABiNet-Large still beats YOLO26m on mIoU while using ~64% less compute (54.8 vs 152.3 GFLOPs), and outperforms YOLO26s/YOLO26n outright on both mIoU and FLOPs.

## AeroScapes Model Zoo

CABiNet and YOLO26-sem are also trained and evaluated under one shared [AeroScapes](https://github.com/ishann/aeroscapes) pipeline (`images/`+`masks/` format, 12 classes). AeroScapes has no source test split, so numbers below are **val-split** mIoU, unlike the test-split numbers above for UAVid/VDD — see the [AeroScapes Dataset](#aeroscapes-dataset) section below.

Params/FLOPs are measured at **720×720**, not 1024×1024 like the UAVid/VDD tables above (chosen close to AeroScapes' native 1280×720 resolution) — **these FLOPs figures are not directly comparable to the UAVid/VDD ones**, only to each other within this table.

| Model                       | mIoU (%) | Params (M) | FLOPs (GFLOPs) | HF Weights                                                                         |
| --------------------------- | -------- | ---------- | -------------- | ---------------------------------------------------------------------------------- |
| YOLO26x-sem                 | 68.36    | 40.15      | 213.0          | [HF Model](https://huggingface.co/dronefreak/aeroscapes-yolo26x-sem)               |
| YOLO26l-sem                 | 68.00    | 17.86      | 95.1           | [HF Model](https://huggingface.co/dronefreak/aeroscapes-yolo26l-sem)               |
| CABiNet (MobileNetV3-Large) | 67.83    | 9.18       | 27.4           | [HF Model](https://huggingface.co/dronefreak/aeroscapes-cabinet-mobilenetv3-large) |
| YOLO26m-sem                 | 66.97    | 14.31      | 75.2           | [HF Model](https://huggingface.co/dronefreak/aeroscapes-yolo26m-sem)               |
| YOLO26s-sem                 | 65.44    | 6.50       | 21.9           | [HF Model](https://huggingface.co/dronefreak/aeroscapes-yolo26s-sem)               |
| YOLO26n-sem                 | 64.86    | 1.63       | 5.6            | [HF Model](https://huggingface.co/dronefreak/aeroscapes-yolo26n-sem)               |

Same pattern as VDD: the top two YOLO26-sem variants edge out CABiNet-Large on raw mIoU, here by an even smaller margin (0.53 pts) — CABiNet is effectively tied with the top of the leaderboard while using ~7.8x less compute than YOLO26x (27.4 vs 213.0 GFLOPs) and ~2.7x less than YOLO26m, which it still beats outright.

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended)
- Conda or pip for package management

### Quick Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/dronefreak/CABiNet.git
   cd CABiNet
   ```

2. **Create and activate environment**:

   ```bash
   # Using conda with provided environment file
   conda env create -f environment.yml
   conda activate cabinet

   # Or install in local environment
   mkdir env/
   conda env create -f environment.yml --prefix env/cabinet
   conda activate env/cabinet
   ```

3. **Install package**:
   ```bash
   pip install -e .
   ```

### Alternative Setup with pip

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .[dev]
```

## Project Structure

```
CABiNet/
├── src/
│   ├── models/              # Neural network architectures
│   │   ├── cabinet.py       # Main CABiNet model implementation
│   │   ├── cab.py           # Context Aggregation Block (plug-and-play module)
│   │   ├── mobilenetv3.py   # MobileNetV3 backbone implementations
│   │   ├── layers/          # Shared layer components
│   │   └── constants.py     # Model configuration constants
│   ├── datasets/            # Data loading and preprocessing
│   │   ├── cityscapes.py    # Cityscapes dataset loader
│   │   ├── uavid.py         # UAVid dataset loader
│   │   ├── aeroscapes.py    # AeroScapes dataset loader
│   │   ├── vdd.py           # VDD (Varied Drone Dataset) loader
│   │   ├── registry.py      # Shared dataset registry (train.py + evaluate.py)
│   │   └── transform.py     # Data augmentation pipeline
│   ├── scripts/             # Training, evaluation, inference scripts
│   │   ├── train.py                       # CABiNet training
│   │   ├── evaluate.py                    # CABiNet standalone checkpoint evaluation
│   │   ├── visualize.py                   # Prediction visualization (Cityscapes only)
│   │   ├── train_yolo.py                  # YOLO26-sem training/validation (Hydra CLI)
│   │   ├── infer_yolo.py                  # YOLO26-sem inference + showcase mosaic
│   │   ├── convert_uavid_to_yolo.py       # UAVid RGB masks -> single-channel format
│   │   ├── convert_aeroscapes_to_yolo.py  # AeroScapes -> shared images/+masks/ format
│   │   └── convert_vdd_to_yolo.py         # VDD -> shared images/+masks/ format
│   └── utils/               # Utility functions
│       ├── loss.py           # OHEM Cross Entropy + Focal Loss
│       ├── optimizer.py      # Custom optimizer with warmup
│       ├── ema.py            # Exponential moving average of model weights
│       ├── early_stopping.py # Patience-based early stopping on mIoU
│       ├── class_weights.py  # Dynamic cls_pw inverse-frequency class weighting
│       ├── logger.py         # Logging utilities
│       ├── profiler.py       # Inference timing / memory profiling
│       └── exceptions.py     # Custom exception classes
├── configs/                 # Configuration files
│   ├── train.yaml           # CABiNet training configuration (Hydra)
│   ├── evaluate.yaml        # Standalone evaluate.py CLI configuration
│   ├── train_yolo.yaml      # YOLO26-sem training configuration (Hydra)
│   ├── dataset/             # Dataset-specific configs
│   │   ├── cityscapes.yaml / uavid.yaml / aeroscapes.yaml / vdd.yaml  # CABiNet training configs
│   │   └── uavid_yolo.yaml / aeroscapes_yolo.yaml / vdd_yolo.yaml     # Ultralytics dataset YAMLs
│   ├── model/               # CABiNet backbone configs (mobilenetv3_{large,small}.yaml)
│   ├── yolo/                # Ultralytics YOLO configs
│   │   ├── uavid_train.yaml / uavid_val.yaml
│   │   ├── aeroscapes_train.yaml / aeroscapes_val.yaml
│   │   ├── vdd_train.yaml / vdd_val.yaml
│   │   └── model/           # Per-variant configs (yolo26{n,s,m,l,x}-sem.yaml, legacy -seg)
│   ├── train_yolo_aeroscapes.yaml # Hydra root config for train_yolo.py on AeroScapes
│   ├── train_yolo_vdd.yaml        # Hydra root config for train_yolo.py on VDD
│   ├── cityscapes_info.json # Cityscapes label information
│   ├── UAVid_info.json      # UAVid label information
│   └── AeroScapes_info.json # AeroScapes label information
├── hf_modelcards/           # Hugging Face model card generator (template + per-model metrics)
├── tests/                   # Test suite
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── conftest.py          # Shared test fixtures
├── legacy/                  # Legacy configuration files
└── .github/                 # GitHub workflows and documentation
    └── workflows/           # CI/CD pipelines
```

### Key Files and Directories

#### Models ([`src/models/`](src/models/))

- **[`cabinet.py`](src/models/cabinet.py)**: Complete implementation of the CABiNet architecture including spatial branch, context branch, and feature fusion modules
- **[`cab.py`](src/models/cab.py)**: Context Aggregation Block - a modular component that can be integrated into other PyTorch models
- **[`mobilenetv3.py`](src/models/mobilenetv3.py)**: MobileNetV3-Large and MobileNetV3-Small backbone implementations with pretrained weight loading
- **[`layers/common.py`](src/models/layers/common.py)**: Reusable layer components (DepthwiseConv, DepthwiseSeparableConv)

#### Datasets ([`src/datasets/`](src/datasets/))

- **[`cityscapes.py`](src/datasets/cityscapes.py)**: Cityscapes dataset loader with label remapping and thread-safe preprocessing
- **[`uavid.py`](src/datasets/uavid.py)**: UAVid dataset loader — consumes the pre-converted `images/+masks/` format (same as the YOLO pipeline), `RandomCrop`-based training
- **[`aeroscapes.py`](src/datasets/aeroscapes.py)**: AeroScapes dataset loader — same pre-converted `images/+masks/` format and `RandomCrop`-based training as `uavid.py`; source images are uniformly 1280x720, so unlike UAVid there's no mixed-resolution batching constraint
- **[`vdd.py`](src/datasets/vdd.py)**: VDD (Varied Drone Dataset) loader — same pre-converted `images/+masks/` format; source images are uniformly 4000x3000 and ship with a real train/val/test split already
- **[`transform.py`](src/datasets/transform.py)**: Comprehensive data augmentation including geometric, photometric, and regularization transforms

#### Scripts ([`src/scripts/`](src/scripts/))

- **[`train.py`](src/scripts/train.py)**: CABiNet training — Hydra config, mixed precision, EMA, early stopping, per-epoch mIoU eval
- **[`evaluate.py`](src/scripts/evaluate.py)**: Standalone checkpoint evaluation CLI (Hydra config) — multi-scale, sliding-window inference; also used internally by `train.py`
- **[`visualize.py`](src/scripts/visualize.py)**: Prediction visualization overlays (Cityscapes only)
- **[`train_yolo.py`](src/scripts/train_yolo.py)**: YOLO26 semantic segmentation training/validation (Hydra CLI wrapping Ultralytics)
- **[`infer_yolo.py`](src/scripts/infer_yolo.py)**: YOLO26-sem inference over images/videos/folders, plus a `--showcase-videos` 2x2 mosaic mode
- **[`convert_uavid_to_yolo.py`](src/scripts/convert_uavid_to_yolo.py)**: Converts UAVid RGB colour masks → single-channel class-ID format shared by both pipelines
- **[`convert_aeroscapes_to_yolo.py`](src/scripts/convert_aeroscapes_to_yolo.py)**: Converts AeroScapes (already single-channel class-ID masks) into the shared `images/+masks/` layout — copies (not symlinks) so the result is redistributable
- **[`convert_vdd_to_yolo.py`](src/scripts/convert_vdd_to_yolo.py)**: Converts VDD (already single-channel class-ID masks, already split into train/val/test) into the shared `images/+masks/` layout — symlinks (VDD is already published on HF, so no redistribution concern)

#### Utilities ([`src/utils/`](src/utils/))

- **[`loss.py`](src/utils/loss.py)**: OHEM Cross Entropy and Focal Loss implementations
- **[`optimizer.py`](src/utils/optimizer.py)**: Custom optimizer with polynomial learning rate decay and warmup
- **[`ema.py`](src/utils/ema.py)**: Exponential moving average of model weights (best/final checkpoints use the EMA model)
- **[`early_stopping.py`](src/utils/early_stopping.py)**: Patience-based early stopping on mIoU
- **[`class_weights.py`](src/utils/class_weights.py)**: Dynamic `cls_pw` inverse-frequency class weighting (ENet formula)
- **[`profiler.py`](src/utils/profiler.py)**: Inference timing and memory profiling (not analytic FLOPs — see [Performance Profiling](#performance-profiling))

#### Configuration ([`configs/`](configs/))

- **[`train.yaml`](configs/train.yaml)**: Main CABiNet training configuration (Hydra)
- **[`evaluate.yaml`](configs/evaluate.yaml)**: Standalone `evaluate.py` CLI configuration
- **`dataset/*.yaml`**: Dataset-specific configurations (paths, preprocessing parameters)
  - `cityscapes.yaml` / `uavid.yaml` / `aeroscapes.yaml` / `vdd.yaml` — CABiNet training configs
  - `uavid_yolo.yaml` — Ultralytics dataset YAML (8 classes incl. Clutter, single-channel masks, 255=ignore)
  - `aeroscapes_yolo.yaml` — Ultralytics dataset YAML (12 classes, single-channel masks, no test split)
  - `vdd_yolo.yaml` — Ultralytics dataset YAML (7 classes, single-channel masks, train/val/test)
- **`model/*.yaml`**: CABiNet backbone configurations
- **`yolo/*.yaml`**: Ultralytics YOLO training/evaluation configs
  - [`uavid_train.yaml`](configs/yolo/uavid_train.yaml) — full training config (AMP, EMA, grad accum, resume, augmentation)
  - [`uavid_val.yaml`](configs/yolo/uavid_val.yaml) — validation / benchmark config
  - [`aeroscapes_train.yaml`](configs/yolo/aeroscapes_train.yaml) / [`aeroscapes_val.yaml`](configs/yolo/aeroscapes_val.yaml) — same, for AeroScapes
  - [`vdd_train.yaml`](configs/yolo/vdd_train.yaml) / [`vdd_val.yaml`](configs/yolo/vdd_val.yaml) — same, for VDD
  - `model/yolo26{n,s,m,l,x}-sem.yaml` — per-size model configs, selected via `'yolo/model@model=yolo26s-sem'`
- **[`train_yolo_aeroscapes.yaml`](configs/train_yolo_aeroscapes.yaml)** / **[`train_yolo_vdd.yaml`](configs/train_yolo_vdd.yaml)**: Hydra root configs for `train_yolo.py` on AeroScapes/VDD (`--config-name train_yolo_aeroscapes` / `train_yolo_vdd`) — `train_yolo.yaml` remains the UAVid default

## Usage

### Training

#### Cityscapes Dataset

1. **Download the dataset** from [Cityscapes website](https://www.cityscapes-dataset.com/downloads/):
   - `gtFine_trainvaltest.zip` (241MB) - Ground truth labels
   - `leftImg8bit_trainvaltest.zip` (11GB) - RGB images

2. **Extract and configure**:

   ```bash
   # Extract datasets
   unzip gtFine_trainvaltest.zip -d data/cityscapes/
   unzip leftImg8bit_trainvaltest.zip -d data/cityscapes/

   # Update dataset path in configs/dataset/cityscapes.yaml
   ```

3. **Start training**:
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   python src/scripts/train.py
   ```

#### UAVid Dataset

CABiNet's UAVid training now consumes the same pre-converted `images/+masks/`
dataset format as the [YOLO26 semantic segmentation pipeline](#uavid--yolo-format)
below — one conversion step serves both pipelines, and there's no more raw
RGB-mask parsing at load time.

1. **Download** from [UAVid website](https://uavid.nl/) under Downloads section

2. **Convert** the raw dataset (see [UAVid → YOLO Format](#uavid--yolo-format)
   for full details):

   ```bash
   python src/scripts/convert_uavid_to_yolo.py \
       --src /path/to/raw/uavid --dst /path/to/converted --info configs/UAVid_info.json
   ```

3. **Configure and train**:

   ```bash
   export UAVID_YOLO_ROOT=/path/to/converted
   # or edit configs/dataset/uavid.yaml and set 'dataset_path:' directly
   #
   # UAVid source images are not uniform resolution, and validation applies
   # no crop, so validation_config.batch_size must be 1 (train.py raises a
   # clear error otherwise):
   python src/scripts/train.py dataset=uavid validation_config.batch_size=1
   ```

   > **Breaking change**: `dataset_path` must now point at the _converted_
   > directory (`convert_uavid_to_yolo.py`'s `--dst`), not the raw UAVid
   > distribution — pointing it at raw `uavid_train/` will fail with a
   > missing-directory error.

Key `training_config` knobs (see [`configs/train.yaml`](configs/train.yaml) for full comments):

| Setting                 | Purpose                                                                               |
| ----------------------- | ------------------------------------------------------------------------------------- |
| `cls_pw`                | 0=uniform, 1=full inverse-frequency class weighting (rare classes: Human, Moving Car) |
| `ema_decay` / `ema_tau` | Exponential moving average of weights; best/final checkpoints use the EMA model       |
| `patience`              | Early stopping on mIoU (epochs with no improvement); 0=disabled                       |

`configs/dataset/uavid.yaml`'s `augmentation:` block (degrees, translate, scale, flips, HSV, mixup) mirrors the YOLO26 pipeline's recipe below — see [`transform.py`](src/datasets/transform.py).

#### AeroScapes Dataset

A second aerial semantic segmentation dataset, wired in the same way as UAVid — a pre-converted `images/+masks/` layout consumed by both CABiNet and the YOLO26 pipeline. Unlike UAVid, AeroScapes source images are already single-channel class-ID masks (no RGB decoding step) and uniformly 1280x720 (no mixed-resolution batching constraint).

1. **Download** the AeroScapes dataset (`JPEGImages/`, `SegmentationClass/`, `ImageSets/{trn,val}.txt`)

2. **Convert** the raw dataset (see [AeroScapes → YOLO Format](#aeroscapes--yolo-format) for full details):

   ```bash
   python src/scripts/convert_aeroscapes_to_yolo.py \
       --src /path/to/raw/aeroscapes --dst /path/to/converted --workers 8
   ```

   Unlike `convert_uavid_to_yolo.py`, this always **copies** files (no symlink option) — the converted output is meant to be redistributable as-is.

3. **Configure and train**:

   ```bash
   export AEROSCAPES_YOLO_ROOT=/path/to/converted
   # or edit configs/dataset/aeroscapes.yaml and set 'dataset_path:' directly

   python src/scripts/train.py dataset=aeroscapes
   ```

   Source images are uniformly 1280x720, so — unlike UAVid — `validation_config.batch_size` can be left at its default; no special-casing is required.

There is no source test split for AeroScapes (only `trn.txt`/`val.txt`); the converter only ever produces `train`/`val`.

#### VDD Dataset

A third aerial semantic segmentation dataset — [VDD (Varied Drone Dataset)](https://github.com/RussRobin/VDD), already published on [Hugging Face](https://huggingface.co/datasets/RussRobin/VDD). Wired in the same way as UAVid/AeroScapes. Unlike both, VDD ships with a real train/val/test split already defined by its own directory layout (`train/`, `val/`, `test/`, each with `src/` + `gt/`), and its masks are already single-channel class-ID PNGs like AeroScapes (no RGB decoding needed).

1. **Download** the VDD dataset (e.g. `git clone` the HF dataset repo) — this gives `train/{src,gt}`, `val/{src,gt}`, `test/{src,gt}`.

2. **Convert** the raw dataset (see [VDD → YOLO Format](#vdd--yolo-format) for full details):

   ```bash
   python src/scripts/convert_vdd_to_yolo.py \
       --src /path/to/VDD --dst /path/to/converted
   ```

   Unlike `convert_aeroscapes_to_yolo.py`, this **symlinks** by default (like `convert_uavid_to_yolo.py`) — VDD is already redistributed on HF, so there's no need to produce a standalone copy.

3. **Configure and train**:

   ```bash
   export VDD_YOLO_ROOT=/path/to/converted
   # or edit configs/dataset/vdd.yaml and set 'dataset_path:' directly

   python src/scripts/train.py dataset=vdd
   ```

   Source images are uniformly 4000x3000, so — like AeroScapes and unlike UAVid — `validation_config.batch_size` can be left at its default.

VDD is small (280 train / 80 val / 40 test images) — heavier augmentation (`mosaic`, `mixup`, `copy_paste`) is enabled by default in the YOLO configs to help offset this.

#### Finetuning CABiNet from a pretrained checkpoint

All CABiNet aerial datasets (UAVid, AeroScapes, VDD) share the same backbone/CAB/spatial/fusion
architecture and only differ in the final classifier heads (sized by `num_classes`). Rather than
always starting from ImageNet backbone weights, `training_config.pretrained_ckpt_path` warm-starts
the **whole model** from a checkpoint trained on a different aerial dataset — converges
substantially faster than backbone-only init, since the CAB/fusion layers are already adapted to
aerial imagery, not just the backbone.

```bash
# Finetune on AeroScapes starting from a UAVid-trained checkpoint
python src/scripts/train.py dataset=aeroscapes \
    training_config.pretrained_ckpt_path=experiments/uavid/.../checkpoint_last.pth
```

Only parameters whose name **and shape** match are loaded — the two classifier heads
(`ab.b4`, `conv_out.conv_out`) are automatically skipped (left at fresh init) whenever
`num_classes` differs between the source checkpoint and the target dataset; everything else
transfers. This is independent of `resume` (which restores optimizer/EMA/epoch state to continue
an interrupted run of the _same_ experiment) — `pretrained_ckpt_path` always starts a fresh run
(epoch 0, fresh optimizer/EMA) with warm-started weights. Accepts either a full training
checkpoint (`checkpoint_last.pth`) or a raw model state_dict (`*_best.pth`).

### UAVid → YOLO Format

UAVid distributes ground-truth labels as **3-channel RGB colour-coded masks** (in each sequence's `Labels/` directory). YOLO's semantic segmentation task requires **single-channel PNG masks** where each pixel value is a class index (0 – N-1); pixel value `255` is reserved for pixels whose colour doesn't match any known class (corrupted/anti-aliased data) and is excluded from training/eval. Per the original UAVid paper, Clutter is a valid class — it is **not** mapped to the ignore label.

#### Why the conversion is needed

| Aspect   | UAVid raw (`Labels/`) | YOLO expected format |
| -------- | --------------------- | -------------------- |
| Channels | 3 (RGB)               | 1 (grayscale)        |
| Encoding | RGB colour per class  | Integer class ID     |
| Ignore   | None explicitly       | Pixel value = 255    |

#### Class mapping after conversion

All 8 UAVid classes are valid and active — none are mapped to the ignore label (this matches CABiNet's own trainId scheme exactly):

| YOLO ID | Class      | Original RGB     |
| ------- | ---------- | ---------------- |
| 0       | Clutter    | `[ 0, 0, 0]`     |
| 1       | Building   | `[128, 0, 0]`    |
| 2       | Road       | `[128, 64, 128]` |
| 3       | Static Car | `[192, 0, 192]`  |
| 4       | Tree       | `[ 0, 128, 0]`   |
| 5       | Vegetation | `[128, 128, 0]`  |
| 6       | Human      | `[ 64, 64, 0]`   |
| 7       | Moving Car | `[ 64, 0, 128]`  |

`255` is reserved for pixels whose colour doesn't match any of the 8 known classes (corrupted/anti-aliased data) and is excluded from training/eval — it does not apply to any defined class.

#### Step-by-step workflow

1. **Convert masks** (one-time, ~2 min on 8 CPU cores):

   ```bash
   python src/scripts/convert_uavid_to_yolo.py \
       --src  /data/uavid \
       --dst  /data/uavid_yolo \
       --info configs/UAVid_info.json \
       --split both \
       --workers 8
   ```

   Output layout:

   ```
   /data/uavid_yolo/
   ├── images/
   │   ├── train/   ← symlinks to original RGB PNGs
   │   └── val/
   └── masks/
       ├── train/   ← single-channel class-ID PNGs (0-7, all 8 classes valid)
       └── val/
   ```

2. **Point the dataset YAML** at the converted data:

   ```bash
   export UAVID_YOLO_ROOT=/data/uavid_yolo
   # or edit configs/dataset/uavid_yolo.yaml and set 'path:' directly
   ```

3. **Install Ultralytics**:

   ```bash
   pip install ultralytics
   ```

4. **Train** — via the Hydra CLI wrapper, not raw `yolo` commands (the config uses the dense `semantic` task, not `-seg`):

   ```bash
   # Default (yolo26n-sem):
   python src/scripts/train_yolo.py

   # Swap size — only override the model group:
   python src/scripts/train_yolo.py 'yolo/model@model=yolo26s-sem'

   # Override hyperparameters (dotted Hydra overrides):
   python src/scripts/train_yolo.py training_config.epochs=200 training_config.batch_size=8

   # Resume an interrupted run:
   python src/scripts/train_yolo.py training_config.resume=true

   # Multi-GPU (DDP):
   python src/scripts/train_yolo.py runtime.device="0,1"
   ```

   > **Key parameters** (see [`configs/train_yolo.yaml`](configs/train_yolo.yaml) for all options with comments):
   >
   > | Parameter                         | Purpose                                                                     |
   > | --------------------------------- | --------------------------------------------------------------------------- |
   > | `nbs`                             | Gradient accumulation target: `accum_steps = nbs / batch`                   |
   > | `amp`                             | Mixed precision (`torch.cuda.amp`)                                          |
   > | `patience`                        | Early stopping — stops if val mIoU doesn't improve for `patience` epochs    |
   > | `cls_pw`                          | Class-imbalance weighting (upweights rare classes: Human, Moving Car)       |
   > | `flipud`                          | Vertical flip — valid for top-down UAV imagery                              |
   > | `mosaic` / `mixup` / `copy_paste` | Aerial-tuned augmentation for small-object diversity (cars, humans)         |
   > | EMA                               | Always on — `best.pt` / `last.pt` are EMA-averaged; no separate flag needed |

5. **Evaluate** (val split by default; UAVid also has a real test split) — use the Hydra wrapper, not
   the raw `yolo semantic val cfg=...` CLI (the `semantic` task isn't reliably invokable that way;
   `train_yolo.py` drives it correctly via the Ultralytics Python API):

   ```bash
   # val split
   python src/scripts/train_yolo.py mode=val
   python src/scripts/train_yolo.py mode=val validation_config.weights=path/to/best.pt

   # test split
   python src/scripts/train_yolo.py mode=val validation_config.split=test \
       validation_config.weights=path/to/best.pt
   ```

6. **Inference & showcase mosaic**:

   ```bash
   # Images/videos/folders -> colorized masks + overlays
   python src/scripts/infer_yolo.py --weights <best.pt> --source <path> --output outputs/inference

   # 2x2 showcase mosaic from 4 clips: each blends raw frame -> full mask
   # over its own duration, tiled into one video
   python src/scripts/infer_yolo.py --weights <best.pt> \
       --showcase-videos clip1.mp4 clip2.mp4 clip3.mp4 clip4.mp4 --output outputs/inference
   ```

### AeroScapes → YOLO Format

AeroScapes distributes ground-truth labels as **already single-channel PNG masks** (`SegmentationClass/`, pixel value = class index 0-11) — unlike UAVid, no RGB colour decoding step is needed. `convert_aeroscapes_to_yolo.py` reads the split membership from `ImageSets/{trn,val}.txt` and copies each `(image, mask)` pair into the shared layout, validating that every mask pixel is a known class ID (or `255`) along the way.

#### Class mapping

All 12 AeroScapes classes are valid and active — none are mapped to the ignore label:

| YOLO ID | Class        | Original RGB (Visualizations) |
| ------- | ------------ | ----------------------------- |
| 0       | Background   | `[  0,   0,   0]`             |
| 1       | Person       | `[192, 128, 128]`             |
| 2       | Bike         | `[  0, 128,   0]`             |
| 3       | Car          | `[128, 128, 128]`             |
| 4       | Drone        | `[128,   0,   0]`             |
| 5       | Boat         | `[  0,   0, 128]`             |
| 6       | Animal       | `[192,   0, 128]`             |
| 7       | Obstacle     | `[192,   0,   0]`             |
| 8       | Construction | `[192, 128,   0]`             |
| 9       | Vegetation   | `[  0,  64,   0]`             |
| 10      | Road         | `[128, 128,   0]`             |
| 11      | Sky          | `[  0, 128, 128]`             |

`255` is reserved for genuinely unrecognized pixel values, which should not occur in a clean copy of the dataset — the converter validates and warns on any mask that contains one.

#### Step-by-step workflow

1. **Convert** (copies, not symlinks — the result is meant to be redistributable):

   ```bash
   python src/scripts/convert_aeroscapes_to_yolo.py \
       --src /path/to/aeroscapes \
       --dst /path/to/aeroscapes_yolo \
       --workers 8
   ```

   Output layout:

   ```
   /path/to/aeroscapes_yolo/
   ├── images/
   │   ├── train/   ← copies of JPEGImages/*.jpg (2621 images)
   │   └── val/     ← copies of JPEGImages/*.jpg (648 images)
   └── masks/
       ├── train/   ← copies of SegmentationClass/*.png (already single-channel)
       └── val/
   ```

   There is no source test split — only `train`/`val` are produced.

2. **Point the dataset YAML** at the converted data:

   ```bash
   export AEROSCAPES_YOLO_ROOT=/path/to/aeroscapes_yolo
   # or edit configs/dataset/aeroscapes_yolo.yaml and set 'path:' directly
   ```

3. **Train** — either the direct Ultralytics CLI:

   ```bash
   yolo semantic train cfg=configs/yolo/aeroscapes_train.yaml
   ```

   or the repo's Hydra wrapper, which also takes the same dotted-path overrides as UAVid's:

   ```bash
   # Default (yolo26n-sem):
   python src/scripts/train_yolo.py --config-name train_yolo_aeroscapes

   # Swap size — only override the model group:
   python src/scripts/train_yolo.py --config-name train_yolo_aeroscapes 'yolo/model@model=yolo26s-sem'

   # Override hyperparameters:
   python src/scripts/train_yolo.py --config-name train_yolo_aeroscapes \
       training_config.epochs=200 training_config.batch_size=8

   # Resume an interrupted run:
   python src/scripts/train_yolo.py --config-name train_yolo_aeroscapes training_config.resume=true

   # Multi-GPU (DDP):
   python src/scripts/train_yolo.py --config-name train_yolo_aeroscapes runtime.device="0,1"
   ```

4. **Evaluate** (val split only — AeroScapes has no test split, see above) — use the Hydra wrapper, not
   the raw `yolo semantic val cfg=...` CLI (the `semantic` task isn't reliably invokable that way):

   ```bash
   python src/scripts/train_yolo.py --config-name train_yolo_aeroscapes mode=val \
       validation_config.weights=runs/aeroscapes/yolo26n/weights/best.pt
   ```

### VDD → YOLO Format

VDD distributes ground-truth labels as **already single-channel PNG masks** (`gt/`, pixel value = class index 0-6) — like AeroScapes and unlike UAVid, no RGB colour decoding step is needed. `convert_vdd_to_yolo.py` discovers stems present in both `<split>/src/` and `<split>/gt/` and symlinks each `(image, mask)` pair into the shared layout (renaming the image extension to lowercase `.jpg` along the way), validating that every mask pixel is a known class ID (or `255`).

#### Class mapping

All 7 VDD classes are valid and active — none are mapped to the ignore label:

| YOLO ID | Class      |
| ------- | ---------- |
| 0       | Other      |
| 1       | Wall       |
| 2       | Road       |
| 3       | Vegetation |
| 4       | Vehicle    |
| 5       | Roof       |
| 6       | Water      |

`255` is reserved for genuinely unrecognized pixel values, which should not occur in a clean copy of the dataset — the converter validates and warns on any mask that contains one.

#### Step-by-step workflow

1. **Convert** (symlinks — VDD is already published on HF, no redistribution copy needed):

   ```bash
   python src/scripts/convert_vdd_to_yolo.py \
       --src /path/to/VDD \
       --dst /path/to/vdd_yolo
   ```

   Output layout:

   ```
   /path/to/vdd_yolo/
   ├── images/
   │   ├── train/   ← symlinks to src/*.JPG (280 images), renamed to lowercase .jpg
   │   ├── val/     ← 80 images
   │   └── test/    ← 40 images
   └── masks/
       ├── train/   ← symlinks to gt/*.png (already single-channel)
       ├── val/
       └── test/
   ```

2. **Point the dataset YAML** at the converted data:

   ```bash
   export VDD_YOLO_ROOT=/path/to/vdd_yolo
   # or edit configs/dataset/vdd_yolo.yaml and set 'path:' directly
   ```

3. **Train** — either the direct Ultralytics CLI:

   ```bash
   yolo semantic train cfg=configs/yolo/vdd_train.yaml
   ```

   or the repo's Hydra wrapper, which also takes the same dotted-path overrides as UAVid's:

   ```bash
   # Default (yolo26n-sem):
   python src/scripts/train_yolo.py --config-name train_yolo_vdd

   # Swap size — only override the model group:
   python src/scripts/train_yolo.py --config-name train_yolo_vdd 'yolo/model@model=yolo26s-sem'

   # Override hyperparameters:
   python src/scripts/train_yolo.py --config-name train_yolo_vdd \
       training_config.epochs=200 training_config.batch_size=8

   # Resume an interrupted run:
   python src/scripts/train_yolo.py --config-name train_yolo_vdd training_config.resume=true

   # Multi-GPU (DDP):
   python src/scripts/train_yolo.py --config-name train_yolo_vdd runtime.device="0,1"
   ```

4. **Evaluate** (val split by default; VDD also has a real test split) — use the Hydra wrapper, not
   the raw `yolo semantic val cfg=...` CLI (the `semantic` task isn't reliably invokable that way):

   ```bash
   # val split
   python src/scripts/train_yolo.py --config-name train_yolo_vdd mode=val \
       validation_config.weights=runs/vdd/yolo26n/weights/best.pt

   # test split
   python src/scripts/train_yolo.py --config-name train_yolo_vdd mode=val \
       validation_config.split=test validation_config.weights=runs/vdd/yolo26n/weights/best.pt
   ```

### Evaluation

Evaluate a trained checkpoint standalone, independent of a training run (Hydra config: [`configs/evaluate.yaml`](configs/evaluate.yaml)). Accepts either a raw model state_dict (`*_best.pth` / the final saved `.pth`) or a full training checkpoint (`checkpoint_last.pth`):

```bash
# Cityscapes, val split, multi-scale + flip (default)
python src/scripts/evaluate.py checkpoint_path=experiments/.../model_best.pth

# UAVid — validation_config.batch_size=1 is required (source images are not
# uniform resolution); dataset_path comes from configs/dataset/uavid.yaml
python src/scripts/evaluate.py checkpoint_path=... dataset=uavid validation_config.batch_size=1

# UAVid test split
python src/scripts/evaluate.py checkpoint_path=... dataset=uavid validation_config.batch_size=1 split=test

# AeroScapes — source images are uniformly 1280x720, so no batch_size=1
# constraint is needed; dataset_path comes from configs/dataset/aeroscapes.yaml
# (no test split exists for this dataset)
python src/scripts/evaluate.py checkpoint_path=... dataset=aeroscapes

# VDD — source images are uniformly 4000x3000, so no batch_size=1 constraint
# is needed; dataset_path comes from configs/dataset/vdd.yaml
python src/scripts/evaluate.py checkpoint_path=... dataset=vdd

# VDD test split
python src/scripts/evaluate.py checkpoint_path=... dataset=vdd split=test

# Fast single-scale evaluation (no flip)
python src/scripts/evaluate.py checkpoint_path=... \
    validation_config.eval_scales=[1.0] validation_config.flip=false
```

`split=train` is intentionally rejected — the dataset classes apply training augmentation whenever `mode='train'`, which would corrupt evaluation metrics. Use `val` or `test`.

### Visualization

Generate prediction visualizations (Cityscapes only — `visualize.py` does not yet use the shared dataset registry, so UAVid is unsupported here):

```bash
python src/scripts/visualize.py
```

### Performance Profiling

Benchmark inference timing and memory usage (`profile_model_flops` measures `torch.profiler` CPU/CUDA time, not analytic FLOP counts — the analytic Params/FLOPs in the [UAVid Model Zoo](#uavid-model-zoo) table above were computed separately via `thop`):

```python
from src.utils.profiler import PerformanceProfiler
from src.models.cabinet import CABiNet

model = CABiNet(n_classes=19, mode="large")
profiler = PerformanceProfiler(model)

# Run comprehensive benchmark
results = profiler.run_full_benchmark(
    input_size=(1, 3, 512, 512),
    num_iterations=100
)

print(f"Average FPS: {results['timing']['fps']:.2f}")
print(f"Peak Memory: {results['memory']['peak_mb']:.2f} MB")
```

## Pretrained Models

Pretrained MobileNetV3 backbone weights: [`src/models/pretrained_backbones/`](src/models/pretrained_backbones/).

Full CABiNet + YOLO26-sem checkpoints are being published to Hugging Face UAVid and VDD model zoos — model cards are generated from [`hf_modelcards/`](hf_modelcards/) (one Jinja template shared across datasets + per-model `metrics.json`; run `python hf_modelcards/generate_hf_model_zoo.py` to regenerate all cards). See [UAVid Model Zoo](#uavid-model-zoo) / [VDD Model Zoo](#vdd-model-zoo) above for current numbers.

## Testing

Run the test suite to verify installation:

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test category
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only
```

See [`tests/README.md`](tests/README.md) for detailed testing documentation.

## Development

### Code Quality

- **Ruff**: Linting and formatting
- **mypy**: Static type checking
- **bandit**: Security linting
- **pytest**: Testing framework

### Pre-commit Hooks

Install pre-commit hooks to automatically check code quality:

```bash
pip install pre-commit
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

### Contributing

Contributions are welcome! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines on:

- Setting up the development environment
- Code style and conventions
- Testing requirements
- Pull request process

## Citation

If you find this work helpful, please consider citing:

```bibtex
@article{LYU2020108,
    author = "Ye Lyu and George Vosselman and Gui-Song Xia and Alper Yilmaz and Michael Ying Yang",
    title = "UAVid: A semantic segmentation dataset for UAV imagery",
    journal = "ISPRS Journal of Photogrammetry and Remote Sensing",
    volume = "165",
    pages = "108 - 119",
    year = "2020",
    issn = "0924-2716",
    doi = "https://doi.org/10.1016/j.isprsjprs.2020.05.009",
    url = "http://www.sciencedirect.com/science/article/pii/S0924271620301295",
}

@INPROCEEDINGS{9560977,
  author={Kumaar, Saumya and Lyu, Ye and Nex, Francesco and Yang, Michael Ying},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)},
  title={CABiNet: Efficient Context Aggregation Network for Low-Latency Semantic Segmentation},
  year={2021},
  pages={13517-13524},
  doi={10.1109/ICRA48506.2021.9560977}
}

@article{Kumaar_Real-time_Semantic_Segmentation_2021,
  author = {Kumaar, Saumya and Lyu, Ye and Nex, Francesco and Yang, Michael Ying},
  doi = {10.1016/j.isprsjprs.2021.06.006},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  pages = {124--134},
  title = {{Real-time Semantic Segmentation with Context Aggregation Network}},
  url = {https://www.sciencedirect.com/science/article/pii/S0924271621001647},
  volume = {178},
  year = {2021}
}

@article{jocher2026ultralytics,
  title={Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models},
  author={Jocher, Glenn and Qiu, Jing and Liu, Mengyu and Lyu, Shuai and Akyon, Fatih Cagatay and Kalfaoglu, Muhammet Esat},
  journal={arXiv preprint arXiv:2606.03748},
  year={2026}
}

@software{cabinet_uavid_benchmark,
  author = {Kumaar, Saumya},
  title = {CABiNet: Semantic Segmentation Benchmarking on UAVid (CABiNet vs. YOLO26)},
  url = {https://github.com/dronefreak/CABiNet},
  year = {2026}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, issues, or collaboration opportunities:

- **Email**: kumaar324@gmail.com
- **Issues**: Please use the [GitHub issue tracker](https://github.com/dronefreak/CABiNet/issues)
- **Pull Requests**: Contributions are welcome via [pull requests](https://github.com/dronefreak/CABiNet/pulls)

## Acknowledgments

This work was conducted at the University of Twente, Faculty of Geo-Information Science and Earth Observation (ITC).
