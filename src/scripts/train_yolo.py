#!/usr/bin/python
# -*- encoding: utf-8 -*-

import json
import logging
from pathlib import Path

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

from src.utils.logger import RichConsoleManager

logger = logging.getLogger(__name__)

# Repo root: src/scripts/train_yolo.py -> src/scripts -> src -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]

# YOLO model variants that support the dedicated 'semantic' task via Ultralytics.
# Only the YOLO26 family ships '-sem' checkpoints — the '-seg' family (yolov8/9/11)
# is instance segmentation ('segment' task) and does NOT apply here.
SUPPORTED_SEMANTIC_MODELS = {
    "yolo26n-sem",
    "yolo26s-sem",
    "yolo26m-sem",
    "yolo26l-sem",
    "yolo26x-sem",
}


def _resolve_dataset_path(config_file: str) -> Path:
    """Return an absolute path to the Ultralytics dataset YAML.

    Ultralytics resolves data= paths relative to the CWD at train() call time,
    not relative to the package. Using an absolute path avoids ambiguity when
    Hydra changes the working directory via hydra.run.dir.
    """
    p = Path(config_file)
    if not p.is_absolute():
        # Resolve relative to original CWD (captured before Hydra may change it)
        p = Path.cwd() / p
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {p}\n"
            "Generate it with: python src/scripts/convert_uavid_to_yolo.py --help"
        )
    return p.resolve()


def _load_dataset_class_names(dataset_path: Path) -> list[str]:
    """Read the ``names:`` mapping out of the Ultralytics dataset YAML.

    Used only to label mode=val's per-class IoU printout in a form directly
    pastable into a hf_modelcards/model_metrics/<model>/metrics.json — reads
    from the dataset config itself rather than a hardcoded per-dataset list
    so this works for any dataset (UAVid, AeroScapes, …) without edits here.
    """
    with open(dataset_path) as f:
        data = yaml.safe_load(f)
    names = data.get("names", {})
    return [names[i] for i in sorted(names, key=int)]


def _resolve_experiments_path(experiments_path: str) -> Path:
    """Resolve experiments_path to an absolute path anchored to this repo.

    Two independent footguns otherwise apply to a relative path:
      1. Ultralytics' get_save_dir() silently prefixes any *relative*
         project= path with the global 'runs_dir' from
         ~/.config/Ultralytics/settings.json — a machine-wide setting that
         may point at a completely different project (this is what actually
         happened: it was left pointing at another repo from earlier work).
      2. A relative path is also ambiguous with respect to whatever
         directory the shell happened to be in at invocation time.
    Anchoring to the repo root sidesteps both — output always lands under
    this repo regardless of global Ultralytics settings or launch CWD.
    """
    p = Path(experiments_path)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _resolve_resume_weights(cfg: DictConfig) -> str | None:
    """Return the path to last.pt when resume=True, or None otherwise."""
    if not cfg.training_config.get("resume", False):
        return None
    run_dir = (
        _resolve_experiments_path(cfg.training_config.experiments_path)
        / cfg.model.run_name
        / "weights"
        / "last.pt"
    )
    if not run_dir.exists():
        logger.warning(
            "resume=True but last.pt not found at %s — starting fresh with pretrained weights",
            run_dir,
        )
        return None
    return str(run_dir)


def _build_train_kwargs(cfg: DictConfig, dataset_path: Path) -> dict:
    """Map Hydra config to Ultralytics model.train() keyword arguments."""
    tc = cfg.training_config
    aug = tc.get("augmentation", {}) or {}

    kwargs: dict = {
        # Task & data
        "data": str(dataset_path),
        "task": "semantic",
        # Schedule
        "epochs": int(tc.epochs),
        "batch": int(tc.batch_size),
        "imgsz": int(tc.imgsz),
        "nbs": int(tc.get("nbs", 64)),
        # Optimizer
        "optimizer": str(tc.get("optimizer", "SGD")),
        "lr0": float(tc.optimizer_lr_start),
        "lrf": float(tc.get("lrf", 0.01)),
        "momentum": float(tc.get("optimizer_momentum", 0.937)),
        "weight_decay": float(tc.optimizer_weight_decay),
        "warmup_epochs": float(tc.get("warmup_epochs", 3.0)),
        "warmup_momentum": float(tc.get("warmup_momentum", 0.8)),
        "warmup_bias_lr": float(tc.get("warmup_bias_lr", 0.1)),
        "cos_lr": bool(tc.get("cos_lr", True)),
        # Mixed precision & stability
        "amp": bool(tc.get("amp", True)),
        "patience": int(tc.get("patience", 30)),
        # Class imbalance
        "cls_pw": float(tc.get("cls_pw", 0.5)),
        # Output & checkpointing
        "project": str(_resolve_experiments_path(tc.experiments_path)),
        "name": str(cfg.model.run_name),
        "exist_ok": bool(tc.get("exist_ok", False)),
        "resume": bool(tc.get("resume", False)),
        "save": True,
        "save_period": int(tc.get("save_period", 10)),
        # Compute
        "device": cfg.runtime.get("device", 0),
        "workers": int(tc.get("num_workers", 8)),
        "seed": int(cfg.runtime.get("seed", 0)),
        "deterministic": bool(cfg.runtime.get("deterministic", True)),
        # Logging
        "plots": True,
        "verbose": True,
    }

    # Flatten augmentation sub-config into top-level kwargs
    aug_keys = {
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "mosaic",
        "mixup",
        "copy_paste",
        "copy_paste_mode",
        "close_mosaic",
        "multi_scale",
    }
    for key in aug_keys:
        val = aug.get(key)
        if val is not None:
            kwargs[key] = val

    return kwargs


def _build_val_kwargs(cfg: DictConfig, dataset_path: Path, weights: str) -> dict:
    """Map Hydra config to Ultralytics model.val() keyword arguments."""
    vc = cfg.validation_config
    return {
        "data": str(dataset_path),
        "task": "semantic",
        "imgsz": int(cfg.training_config.imgsz),
        "batch": int(vc.get("batch_size", 1)),
        "device": cfg.runtime.get("device", 0),
        "split": str(vc.get("split", "val")),
        "save_json": bool(vc.get("save_json", True)),
        "augment": bool(vc.get("augment", False)),
        "plots": True,
        "verbose": True,
    }


@hydra.main(version_base=None, config_path="../../configs", config_name="train_yolo")
def main(cfg: DictConfig) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is not installed.\n"
            "Install it with:  pip install ultralytics\n"
            "Or add it to the project:  pip install -e '.[yolo]'"
        ) from exc

    console = RichConsoleManager.get_console()
    console.print(OmegaConf.to_yaml(cfg), style="warning")

    mode = str(cfg.get("mode", "train"))
    model_name: str = cfg.model.model_name
    base_name = model_name.replace(".pt", "").replace(".yaml", "")

    if base_name not in SUPPORTED_SEMANTIC_MODELS:
        logger.warning(
            "Model '%s' is not in the known supported-semantic-segmentation list %s. "
            "Proceeding anyway — only the yolo26*-sem family is verified to work "
            "with task='semantic'.",
            base_name,
            SUPPORTED_SEMANTIC_MODELS,
        )

    dataset_path = _resolve_dataset_path(cfg.dataset.config_file)
    console.print(f"Dataset config: {dataset_path}", style="info")

    experiments_path = _resolve_experiments_path(cfg.training_config.experiments_path)

    if mode == "train":
        resume_weights = _resolve_resume_weights(cfg)
        if resume_weights:
            console.print(f"Resuming from: {resume_weights}", style="info")
            model = YOLO(resume_weights)
        else:
            console.print(f"Loading model: {model_name}", style="info")
            model = YOLO(model_name)

        train_kwargs = _build_train_kwargs(cfg, dataset_path)
        console.print("Starting YOLO training...", style="info")
        results = model.train(**train_kwargs)

        if results is not None:
            console.print(
                f"Training complete. Best weights: "
                f"{experiments_path / cfg.model.run_name / 'weights' / 'best.pt'}",
                style="info",
            )

    elif mode == "val":
        weights = cfg.validation_config.get("weights")
        if not weights:
            weights_path = experiments_path / cfg.model.run_name / "weights" / "best.pt"
        else:
            weights_path = Path(weights)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weights not found: {weights_path}\n"
                f"Set validation_config.weights= or train first."
            )

        console.print(f"Validating with weights: {weights_path}", style="info")
        model = YOLO(str(weights_path))
        val_kwargs = _build_val_kwargs(cfg, dataset_path, str(weights_path))
        split = val_kwargs["split"]

        results = model.val(**val_kwargs)
        if results is not None:
            console.print(f"Split: {split}", style="info")
            console.print(f"mIoU: {results.miou:.4f}", style="info")
            console.print(f"Pixel accuracy: {results.pixel_accuracy:.4f}", style="info")

            class_names = _load_dataset_class_names(dataset_path)
            per_class_iou = {
                name: round(float(iou), 4)
                for name, iou in zip(class_names, results.per_class_iou)
            }
            console.print("Per-class IoU:", style="info")
            for name, iou in per_class_iou.items():
                console.print(f"  {name}: {iou:.4f}", style="info")

            metrics_snippet = {
                "eval_split": split,
                "miou": round(float(results.miou), 4),
                "pixel_acc": round(float(results.pixel_accuracy), 4),
                "per_class_iou": per_class_iou,
            }
            console.print(
                "\nPaste into hf_modelcards/model_metrics/<model>/metrics.json:",
                style="info",
            )
            console.print(json.dumps(metrics_snippet, indent=2), style="info")

    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Choose 'train' or 'val'.\n"
            "Override on CLI:  python src/scripts/train_yolo.py mode=val"
        )


if __name__ == "__main__":
    main()
