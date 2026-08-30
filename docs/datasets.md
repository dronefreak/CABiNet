# Datasets — conversion, class mappings, and layouts

Detailed reference for preparing each dataset. The [main README](../README.md#datasets)
has the short version; this file has the full walkthroughs, class-ID/RGB mappings, and
output-directory layouts.

All aerial datasets (UAVid, AeroScapes, VDD) are normalized to **one** pre-converted
layout that both pipelines consume — CABiNet's dataset classes and the Ultralytics YOLO26
`semantic` pipeline read the same directory:

```
<root>/
├── images/
│   ├── train/
│   ├── val/
│   └── test/          # when the dataset has one
└── masks/
    ├── train/         # single-channel PNGs, pixel value = class ID (0..N-1), 255 = ignore
    ├── val/
    └── test/
```

Contents:

- [Cityscapes](#cityscapes)
- [UAVid → shared format](#uavid--shared-format)
- [AeroScapes → shared format](#aeroscapes--shared-format)
- [VDD → shared format](#vdd--shared-format)
- [Finetuning CABiNet from a pretrained checkpoint](#finetuning-cabinet-from-a-pretrained-checkpoint)
- [Per-dataset evaluation notes](#per-dataset-evaluation-notes)

---

## Cityscapes

No conversion step — `src/datasets/cityscapes.py` reads the raw distribution directly and
remaps label IDs → trainIds via `configs/cityscapes_info.json`.

1. **Download** from the [Cityscapes website](https://www.cityscapes-dataset.com/downloads/):
   - `gtFine_trainvaltest.zip` (241 MB) — ground-truth labels
   - `leftImg8bit_trainvaltest.zip` (11 GB) — RGB images

2. **Extract**:

   ```bash
   unzip gtFine_trainvaltest.zip     -d data/cityscapes/
   unzip leftImg8bit_trainvaltest.zip -d data/cityscapes/
   ```

3. **Point the config** at it — set `dataset_path:` in `configs/dataset/cityscapes.yaml`.

4. **Train**:

   ```bash
   export CUDA_VISIBLE_DEVICES=0
   python src/scripts/train.py          # dataset=cityscapes is the default
   ```

---

## UAVid → shared format

UAVid distributes ground-truth labels as **3-channel RGB colour-coded masks** (in each
sequence's `Labels/` directory). Both pipelines require **single-channel PNG masks** where
each pixel value is a class index.

### Why the conversion is needed

| Aspect   | UAVid raw (`Labels/`) | Expected format  |
| -------- | --------------------- | ---------------- |
| Channels | 3 (RGB)               | 1 (grayscale)    |
| Encoding | RGB colour per class  | Integer class ID |
| Ignore   | none explicit         | pixel value 255  |

### Class mapping

All 8 UAVid classes are valid and active — none are mapped to the ignore label (matches
CABiNet's own trainId scheme and the original UAVid paper, in which Clutter is a real class).

| ID  | Class      | Original RGB     |
| --- | ---------- | ---------------- |
| 0   | Clutter    | `[0, 0, 0]`      |
| 1   | Building   | `[128, 0, 0]`    |
| 2   | Road       | `[128, 64, 128]` |
| 3   | Static Car | `[192, 0, 192]`  |
| 4   | Tree       | `[0, 128, 0]`    |
| 5   | Vegetation | `[128, 128, 0]`  |
| 6   | Human      | `[64, 64, 0]`    |
| 7   | Moving Car | `[64, 0, 128]`   |

`255` is reserved for pixels whose colour matches none of the 8 classes
(corrupted/anti-aliased source data) and is excluded from training/eval.

### Workflow

1. **Convert** (one-time, ~2 min on 8 CPU cores):

   ```bash
   python src/scripts/convert_uavid_to_yolo.py \
       --src  /data/uavid \
       --dst  /data/uavid_yolo \
       --info configs/UAVid_info.json \
       --split both \
       --workers 8
   ```

   Output layout (images are **symlinked** to the originals; masks are freshly written):

   ```
   /data/uavid_yolo/
   ├── images/
   │   ├── train/   ← symlinks to original RGB PNGs
   │   └── val/
   └── masks/
       ├── train/   ← single-channel class-ID PNGs (0-7)
       └── val/
   ```

2. **Point the dataset configs** at the converted directory:

   ```bash
   export UAVID_YOLO_ROOT=/data/uavid_yolo
   # CABiNet: or edit configs/dataset/uavid.yaml       -> dataset_path:
   # YOLO26:  or edit configs/dataset/uavid_yolo.yaml  -> path:
   ```

   > **Breaking change** (vs. older revisions): `dataset_path` must point at the
   > _converted_ directory, not the raw UAVid distribution.

3. **Train CABiNet** — `validation_config.batch_size=1` is required (UAVid source images
   are mixed-resolution — 3840×2160 and 4096×2160 both occur — and val/test mode applies
   no crop, so a larger batch can't be stacked; `train.py` raises a clear error otherwise):

   ```bash
   python src/scripts/train.py dataset=uavid validation_config.batch_size=1
   ```

   `configs/dataset/uavid.yaml`'s `augmentation:` block (degrees, translate, scale, flips,
   HSV, mixup) mirrors the YOLO26 recipe — see `src/datasets/transform.py`.

4. **Train YOLO26-sem** — via the Hydra wrapper, not raw `yolo` commands (the config uses
   the dense `semantic` task, not `-seg`):

   ```bash
   python src/scripts/train_yolo.py                              # default yolo26n-sem
   python src/scripts/train_yolo.py 'yolo/model@model=yolo26s-sem'
   python src/scripts/train_yolo.py training_config.epochs=200 training_config.batch_size=8
   python src/scripts/train_yolo.py training_config.resume=true
   python src/scripts/train_yolo.py runtime.device="0,1"         # DDP
   ```

   Key YOLO parameters (see `configs/train_yolo.yaml` for all options with comments):

   | Parameter                         | Purpose                                                                  |
   | --------------------------------- | ------------------------------------------------------------------------ |
   | `nbs`                             | gradient-accumulation target: `accum_steps = nbs / batch`                |
   | `amp`                             | mixed precision (`torch.cuda.amp`)                                       |
   | `patience`                        | early stopping — stops if val mIoU doesn't improve for `patience` epochs |
   | `cls_pw`                          | class-imbalance weighting (upweights rare classes: Human, Moving Car)    |
   | `flipud`                          | vertical flip — valid for top-down UAV imagery                           |
   | `mosaic` / `mixup` / `copy_paste` | aerial-tuned augmentation for small-object diversity                     |
   | EMA                               | always on — `best.pt` / `last.pt` are EMA-averaged                       |

5. **Evaluate** (val split by default; UAVid also has a real test split) — use the Hydra
   wrapper, not the raw `yolo semantic val cfg=...` CLI (the `semantic` task isn't reliably
   invokable that way):

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
   # images / videos / folders -> colorized masks + overlays
   python src/scripts/infer_yolo.py --weights <best.pt> --source <path> --output outputs/inference

   # 2x2 showcase mosaic from 4 clips: each blends raw frame -> full mask over its own duration
   python src/scripts/infer_yolo.py --weights <best.pt> \
       --showcase-videos clip1.mp4 clip2.mp4 clip3.mp4 clip4.mp4 --output outputs/inference
   ```

---

## AeroScapes → shared format

AeroScapes distributes ground-truth labels as **already single-channel PNG masks**
(`SegmentationClass/`, pixel value = class index 0-11) — no RGB decoding step.
`convert_aeroscapes_to_yolo.py` reads split membership from `ImageSets/{trn,val}.txt` and
**copies** each `(image, mask)` pair into the shared layout, validating that every mask
pixel is a known class ID (or `255`).

### Class mapping

All 12 classes are valid and active — none mapped to ignore.

| ID  | Class        | RGB (visualizations) |
| --- | ------------ | -------------------- |
| 0   | Background   | `[0, 0, 0]`          |
| 1   | Person       | `[192, 128, 128]`    |
| 2   | Bike         | `[0, 128, 0]`        |
| 3   | Car          | `[128, 128, 128]`    |
| 4   | Drone        | `[128, 0, 0]`        |
| 5   | Boat         | `[0, 0, 128]`        |
| 6   | Animal       | `[192, 0, 128]`      |
| 7   | Obstacle     | `[192, 0, 0]`        |
| 8   | Construction | `[192, 128, 0]`      |
| 9   | Vegetation   | `[0, 64, 0]`         |
| 10  | Road         | `[128, 128, 0]`      |
| 11  | Sky          | `[0, 128, 128]`      |

`255` should not occur in a clean copy of the dataset — the converter validates and warns
on any mask that contains one.

### Workflow

1. **Download** the AeroScapes dataset (`JPEGImages/`, `SegmentationClass/`,
   `ImageSets/{trn,val}.txt`).

2. **Convert** (always copies, never symlinks — the output is meant to be redistributable):

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

   There is **no source test split** — only `train`/`val` are produced.

3. **Point the dataset configs**:

   ```bash
   export AEROSCAPES_YOLO_ROOT=/path/to/aeroscapes_yolo
   # CABiNet: or edit configs/dataset/aeroscapes.yaml       -> dataset_path:
   # YOLO26:  or edit configs/dataset/aeroscapes_yolo.yaml  -> path:
   ```

4. **Train CABiNet** — source images are uniformly 1280×720, so no `batch_size=1`
   constraint:

   ```bash
   python src/scripts/train.py dataset=aeroscapes
   ```

5. **Train YOLO26-sem** — either the direct Ultralytics CLI:

   ```bash
   yolo semantic train cfg=configs/yolo/aeroscapes_train.yaml
   ```

   or the Hydra wrapper (same dotted-path overrides as UAVid):

   ```bash
   python src/scripts/train_yolo.py --config-name train_yolo_aeroscapes
   python src/scripts/train_yolo.py --config-name train_yolo_aeroscapes 'yolo/model@model=yolo26s-sem'
   python src/scripts/train_yolo.py --config-name train_yolo_aeroscapes training_config.resume=true
   python src/scripts/train_yolo.py --config-name train_yolo_aeroscapes runtime.device="0,1"
   ```

6. **Evaluate** (val split only — AeroScapes has no test split):

   ```bash
   python src/scripts/train_yolo.py --config-name train_yolo_aeroscapes mode=val \
       validation_config.weights=runs/aeroscapes/yolo26n/weights/best.pt
   ```

---

## VDD → shared format

[VDD (Varied Drone Dataset)](https://github.com/RussRobin/VDD), also published on
[Hugging Face](https://huggingface.co/datasets/RussRobin/VDD). Ships with a real
train/val/test split already defined by its directory layout (`train/`, `val/`, `test/`,
each with `src/` + `gt/`), and its masks are already single-channel class-ID PNGs (no RGB
decoding). `convert_vdd_to_yolo.py` discovers stems present in both `<split>/src/` and
`<split>/gt/` and **symlinks** each `(image, mask)` pair into the shared layout (renaming
the image extension to lowercase `.jpg`), validating that every mask pixel is a known
class ID (or `255`).

### Class mapping

All 7 classes are valid and active — none mapped to ignore.

| ID  | Class      |
| --- | ---------- |
| 0   | Other      |
| 1   | Wall       |
| 2   | Road       |
| 3   | Vegetation |
| 4   | Vehicle    |
| 5   | Roof       |
| 6   | Water      |

`255` should not occur in a clean copy — the converter validates and warns.

### Workflow

1. **Download** the VDD dataset (e.g. `git clone` the HF dataset repo) — this gives
   `train/{src,gt}`, `val/{src,gt}`, `test/{src,gt}`.

2. **Convert** (symlinks by default — VDD is already redistributed on HF, so no standalone
   copy is needed):

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

3. **Point the dataset configs**:

   ```bash
   export VDD_YOLO_ROOT=/path/to/vdd_yolo
   # CABiNet: or edit configs/dataset/vdd.yaml       -> dataset_path:
   # YOLO26:  or edit configs/dataset/vdd_yolo.yaml  -> path:
   ```

4. **Train CABiNet** — source images are uniformly 4000×3000, so no `batch_size=1`
   constraint:

   ```bash
   python src/scripts/train.py dataset=vdd
   ```

   VDD is small (280 train / 80 val / 40 test) — heavier augmentation (`mosaic`, `mixup`,
   `copy_paste`) is enabled by default in the YOLO configs to help offset this.

5. **Train YOLO26-sem** — direct Ultralytics CLI:

   ```bash
   yolo semantic train cfg=configs/yolo/vdd_train.yaml
   ```

   or the Hydra wrapper:

   ```bash
   python src/scripts/train_yolo.py --config-name train_yolo_vdd
   python src/scripts/train_yolo.py --config-name train_yolo_vdd 'yolo/model@model=yolo26s-sem'
   python src/scripts/train_yolo.py --config-name train_yolo_vdd training_config.resume=true
   python src/scripts/train_yolo.py --config-name train_yolo_vdd runtime.device="0,1"
   ```

6. **Evaluate** (val split by default; VDD also has a real test split):

   ```bash
   # val split
   python src/scripts/train_yolo.py --config-name train_yolo_vdd mode=val \
       validation_config.weights=runs/vdd/yolo26n/weights/best.pt

   # test split
   python src/scripts/train_yolo.py --config-name train_yolo_vdd mode=val \
       validation_config.split=test validation_config.weights=runs/vdd/yolo26n/weights/best.pt
   ```

---

## Finetuning CABiNet from a pretrained checkpoint

All CABiNet aerial datasets (UAVid, AeroScapes, VDD) share the same
backbone/CAB/spatial/fusion architecture and differ only in the final classifier heads
(sized by `num_classes`). Rather than always starting from ImageNet backbone weights,
`training_config.pretrained_ckpt_path` warm-starts the **whole model** from a checkpoint
trained on a different aerial dataset — this converges substantially faster than
backbone-only init, since the CAB/fusion layers are already adapted to aerial imagery.

```bash
# Finetune on AeroScapes starting from a UAVid-trained checkpoint
python src/scripts/train.py dataset=aeroscapes \
    training_config.pretrained_ckpt_path=experiments/uavid/.../checkpoint_last.pth
```

Only parameters whose name **and** shape match are loaded — the two classifier heads
(`ab.b4`, `conv_out.conv_out`) are automatically skipped (left at fresh init) whenever
`num_classes` differs between source and target; everything else transfers. This is
independent of `resume` (which restores optimizer/EMA/epoch state to continue an
interrupted run of the _same_ experiment) — `pretrained_ckpt_path` always starts a fresh
run (epoch 0, fresh optimizer/EMA) with warm-started weights. Accepts either a full
training checkpoint (`checkpoint_last.pth`) or a raw model state_dict (`*_best.pth`).

---

## Per-dataset evaluation notes

`src/scripts/evaluate.py` evaluates a trained checkpoint independently of a training run
(Hydra config: `configs/evaluate.yaml`). Accepts a raw model state_dict (`*_best.pth`) or
a full training checkpoint (`checkpoint_last.pth`).

```bash
# Cityscapes, val split, multi-scale + flip (default)
python src/scripts/evaluate.py checkpoint_path=experiments/.../model_best.pth

# UAVid — validation_config.batch_size=1 is required (mixed-resolution source images)
python src/scripts/evaluate.py checkpoint_path=... dataset=uavid validation_config.batch_size=1
python src/scripts/evaluate.py checkpoint_path=... dataset=uavid validation_config.batch_size=1 split=test

# AeroScapes — uniformly 1280x720, no batch_size constraint; no test split exists
python src/scripts/evaluate.py checkpoint_path=... dataset=aeroscapes

# VDD — uniformly 4000x3000, no batch_size constraint
python src/scripts/evaluate.py checkpoint_path=... dataset=vdd
python src/scripts/evaluate.py checkpoint_path=... dataset=vdd split=test

# Fast single-scale evaluation (no flip)
python src/scripts/evaluate.py checkpoint_path=... \
    validation_config.eval_scales=[1.0] validation_config.flip=false
```

`split=train` is intentionally rejected — the dataset classes apply training augmentation
whenever `mode='train'`, which would corrupt evaluation metrics. Use `val` or `test`.
