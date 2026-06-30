#!/usr/bin/env python3
"""Convert UAVid RGB colour-coded label masks to YOLO single-channel format.

UAVid original format (Labels/ directory)
------------------------------------------
Each ground-truth PNG is a 3-channel RGB image where every pixel's colour
encodes a semantic class (see UAVid_info.json). Example:
  [128,  0,  0]  → Building
  [  0,  0,  0]  → Clutter  (ignored in evaluation)

YOLO semantic-segmentation format
----------------------------------
Each mask is a *single-channel* (mode-"L") PNG where:
  pixel value = class index  (0 … N-1)
  pixel value = 255          → ignored during training / evaluation

Mapping used by this script
----------------------------
  Clutter  (ignoreInEval=true)  → 255  (YOLO ignore label)
  Building                      → 0
  Road                          → 1
  Static Car                    → 2
  Tree                          → 3
  Vegetation                    → 4
  Human                         → 5
  Moving Car                    → 6

The script preserves the exact filename stem so images and masks remain
matched when pointed at the same glob pattern in the YOLO dataset YAML.

Usage
------
  python src/scripts/convert_uavid_to_yolo.py \\
      --src  /data/uavid \\
      --dst  /data/uavid_yolo \\
      --info configs/UAVid_info.json \\
      --split both \\
      --workers 4

Directory layout produced
--------------------------
  <dst>/
    images/
      train/   ← symlinks (or copies with --copy-images) to original PNGs
      val/
    masks/
      train/   ← converted single-channel PNGs
      val/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Any, cast

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Palette → class-ID mapping
# ---------------------------------------------------------------------------

IGNORE_LABEL: int = 255


def build_colour_map(labels_info: list[dict]) -> Dict[Tuple[int, int, int], int]:
    """Return mapping  RGB tuple → YOLO class ID.

    Classes with ``ignoreInEval=True`` are mapped to IGNORE_LABEL (255).
    Remaining classes are assigned consecutive IDs (0-based) ordered by
    their original ``trainId`` value.
    """
    # Separate eval and ignore classes
    eval_classes = sorted(
        [c for c in labels_info if not c["ignoreInEval"]], key=lambda c: c["trainId"]
    )
    ignore_classes = [c for c in labels_info if c["ignoreInEval"]]

    colour_map: Dict[Tuple[int, int, int], int] = {}

    for yolo_id, cls in enumerate(eval_classes):
        rgb = tuple(cls["color"])  # type: ignore[arg-type]
        colour_map[rgb] = yolo_id  # type: ignore[assignment]

    for cls in ignore_classes:
        rgb = tuple(cls["color"])  # type: ignore[arg-type]
        colour_map[rgb] = IGNORE_LABEL  # type: ignore[assignment]

    return colour_map  # type: ignore[return-value]


def build_lut(colour_map: Dict[Tuple[int, int, int], int]) -> np.ndarray:
    """Build a (256, 256, 256) uint8 lookup table for fast vectorised mapping.

    Unknown colours default to IGNORE_LABEL.
    """
    lut = np.full((256, 256, 256), IGNORE_LABEL, dtype=np.uint8)
    for (r, g, b), cls_id in colour_map.items():
        lut[r, g, b] = cls_id
    return lut


def convert_mask(
    src_path: Path, dst_path: Path, lut: np.ndarray, dry_run: bool = False
) -> str:
    """Convert a single RGB label PNG to a single-channel YOLO mask.

    Parameters
    ----------
    src_path:
        Path to the original 3-channel colour mask.
    dst_path:
        Output path for the converted single-channel mask.
    lut:
        Pre-built (256, 256, 256) lookup table mapping RGB → class ID.
    dry_run:
        If True, do not write any files.

    Returns
    -------
    str
        A short status string (for progress reporting).
    """
    img = Image.open(src_path).convert("RGB")
    rgb = np.array(img, dtype=np.uint8)  # (H, W, 3)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    mask = lut[r, g, b]  # (H, W), uint8

    if not dry_run:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(dst_path, optimize=False)

    return f"ok:{src_path.name}"


# ---------------------------------------------------------------------------
# Dataset restructuring helpers
# ---------------------------------------------------------------------------


def iter_split(src_root: Path, split: str) -> List[Tuple[Path, str, str]]:
    """Yield (image_path, seq_folder, stem) for all images in a dataset split.

    Expected layout::

        <src_root>/<split>/<seqN>/Images/*.png
        <src_root>/<split>/<seqN>/Labels/*.png   ← colour masks (raw UAVid)
    """
    split_dir = src_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    results: List[Tuple[Path, str, str]] = []
    for seq_dir in sorted(split_dir.iterdir()):
        img_dir = seq_dir / "Images"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.png")):
            results.append((img_path, seq_dir.name, img_path.stem))
    return results


def setup_image_links(
    src_root: Path,
    dst_root: Path,
    split: str,
    copy: bool = False,
) -> None:
    """Create symlinks (or copies) for RGB images into <dst>/images/<split>/."""
    img_out_dir = dst_root / "images" / split
    img_out_dir.mkdir(parents=True, exist_ok=True)

    for img_path, seq, stem in iter_split(src_root, split):
        # Flatten <seq>_<stem> to keep filenames unique across sequences
        out_name = f"{seq}_{stem}.png"
        dst = img_out_dir / out_name
        if dst.exists():
            continue
        if copy:
            shutil.copy2(img_path, dst)
        else:
            dst.symlink_to(img_path.resolve())


def _worker(args: Tuple[Path, Path, np.ndarray, bool]) -> str:
    src, dst, lut, dry_run = args
    return convert_mask(src, dst, lut, dry_run)


def convert_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    lut: np.ndarray,
    workers: int = 1,
    dry_run: bool = False,
    copy_images: bool = False,
) -> int:
    """Convert all colour masks in *split* and set up image references.

    Returns the number of masks converted.
    """
    mask_out_dir = dst_root / "masks" / split
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    # Collect (src_mask_path, dst_mask_path) pairs
    entries = iter_split(src_root, split)
    tasks: List[Tuple[Path, Path, np.ndarray, bool]] = []
    for img_path, seq, stem in entries:
        label_path = img_path.parent.parent / "Labels" / f"{stem}.png"
        if not label_path.exists():
            print(f"[WARN] No Labels/ mask for {img_path.relative_to(src_root)}")
            continue
        dst_mask = mask_out_dir / f"{seq}_{stem}.png"
        tasks.append((label_path, dst_mask, lut, dry_run))

    # Set up image symlinks / copies
    if not dry_run:
        setup_image_links(src_root, dst_root, split, copy=copy_images)

    # Convert masks (parallel if requested)
    n_ok = 0
    if workers <= 1 or len(tasks) <= 1:
        for task in tasks:
            status = _worker(task)
            if status.startswith("ok"):
                n_ok += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker, t): t for t in tasks}
            for future in as_completed(futures):
                try:
                    status = future.result()
                    if status.startswith("ok"):
                        n_ok += 1
                except Exception as exc:
                    print(f"[ERROR] {futures[future][0].name}: {exc}")

    return n_ok


# ---------------------------------------------------------------------------
# Public API (importable by tests)
# ---------------------------------------------------------------------------


def load_labels_info(info_path: str | Path) -> list[dict]:
    """Load and return the UAVid_info.json label list."""
    with open(info_path) as f:
        return cast(List[Dict[str, Any]], json.load(f))


def get_yolo_class_names(labels_info: list[dict]) -> Dict[int, str]:
    """Return {yolo_id: class_name} for non-ignored classes, ordered by trainId."""
    eval_classes = sorted(
        [c for c in labels_info if not c["ignoreInEval"]], key=lambda c: c["trainId"]
    )
    return {yolo_id: cls["name"] for yolo_id, cls in enumerate(eval_classes)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert UAVid Labels/ (RGB) → YOLO single-channel masks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--src",
        required=True,
        type=Path,
        help="Root of the UAVid dataset (contains train/ and val/ sub-dirs)",
    )
    p.add_argument(
        "--dst",
        required=True,
        type=Path,
        help="Output directory for the YOLO-format dataset",
    )
    p.add_argument(
        "--info",
        default="configs/UAVid_info.json",
        type=Path,
        help="Path to UAVid_info.json (label colour palette)",
    )
    p.add_argument(
        "--split",
        choices=["train", "val", "both"],
        default="both",
        help="Which dataset split(s) to convert",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of parallel worker processes",
    )
    p.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy RGB images instead of creating symlinks",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Walk the dataset and report counts without writing any files",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    labels_info = load_labels_info(args.info)
    colour_map = build_colour_map(labels_info)
    lut = build_lut(colour_map)
    class_names = get_yolo_class_names(labels_info)

    print(f"[INFO] YOLO class mapping ({len(class_names)} classes):")
    for yolo_id, name in class_names.items():
        print(f"         {yolo_id:2d} → {name}")
    print(f"         {IGNORE_LABEL} → <ignore> (Clutter + unlabelled)")

    splits = ["train", "val"] if args.split == "both" else [args.split]
    total = 0
    for split in splits:
        n = convert_split(
            src_root=args.src,
            dst_root=args.dst,
            split=split,
            lut=lut,
            workers=args.workers,
            dry_run=args.dry_run,
            copy_images=args.copy_images,
        )
        verb = "would convert" if args.dry_run else "converted"
        print(f"[INFO] {split}: {verb} {n} masks")
        total += n

    print(f"\n[DONE] Total masks {'scanned' if args.dry_run else 'written'}: {total}")
    if not args.dry_run:
        print(f"       Output: {args.dst}")
        print(
            "       Next: update configs/dataset/uavid_yolo.yaml "
            f"with  path: {args.dst}"
        )


if __name__ == "__main__":
    main()
