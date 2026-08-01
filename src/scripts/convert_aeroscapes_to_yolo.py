#!/usr/bin/env python3
"""Convert the AeroScapes dataset into the shared CABiNet/YOLO layout.

AeroScapes original format (as distributed)
--------------------------------------------
    aeroscapes/                 ← *--src* argument
    ├── JPEGImages/*.jpg        ← RGB input images (uniformly 1280x720)
    ├── SegmentationClass/*.png ← single-channel (mode "L") masks, pixel
    │                             value = class ID already (0-11) — NOT
    │                             RGB colour-coded like UAVid's Labels/, so
    │                             no colour lookup table is needed here
    ├── Visualizations/*.png    ← colourized masks (not used for training)
    └── ImageSets/{trn,val}.txt ← one filename stem per line; together the
                                   two files cover every image with zero
                                   overlap. There is no source test split.

Class mapping (standard published AeroScapes 12-class set, see
configs/AeroScapes_info.json)
-----------------------------------------------------------------
   0  Background     6  Animal
   1  Person         7  Obstacle
   2  Bike           8  Construction
   3  Car            9  Vegetation
   4  Drone         10  Road
   5  Boat          11  Sky

Pixel value 255 is reserved for genuinely unrecognized values (there should
be none in a clean copy of the dataset); this script validates that every
mask pixel is in {0..11, 255} and warns on any file that violates this.

Output layout produced
------------------------
  <dst>/
    images/
      train/   ← copies (NOT symlinks — see below) of JPEGImages/*.jpg
      val/
    masks/
      train/   ← copies of SegmentationClass/*.png (already single-channel)
      val/

Unlike convert_uavid_to_yolo.py (which symlinks by default), this script
always COPIES files: the user intends to redistribute the converted dataset
on Hugging Face, and a directory of symlinks is not redistributable.

Usage
------
  python src/scripts/convert_aeroscapes_to_yolo.py \\
      --src /path/to/aeroscapes \\
      --dst /path/to/aeroscapes_yolo \\
      --workers 8

  # Dry-run to check counts before writing
  python src/scripts/convert_aeroscapes_to_yolo.py \\
      --src /path/to/aeroscapes --dst /tmp/out --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

VALID_CLASS_IDS = frozenset(range(12))
IGNORE_LABEL: int = 255


# ---------------------------------------------------------------------------
# Split discovery
# ---------------------------------------------------------------------------


def load_split_stems(imagesets_dir: Path, split_file: str) -> List[str]:
    """Return the sorted list of filename stems listed in an ImageSets file."""
    path = imagesets_dir / split_file
    if not path.exists():
        return []
    with open(path) as f:
        return sorted(line.strip() for line in f if line.strip())


def discover_splits(src_root: Path) -> Dict[str, List[str]]:
    """Return {"train": [...], "val": [...]} stems from ImageSets/{trn,val}.txt.

    There is no source test split — a caller that needs one must carve it
    out of val/train themselves; this script only ever produces train/val.
    """
    imagesets_dir = src_root / "ImageSets"
    return {
        "train": load_split_stems(imagesets_dir, "trn.txt"),
        "val": load_split_stems(imagesets_dir, "val.txt"),
    }


# ---------------------------------------------------------------------------
# Per-file copy + validation
# ---------------------------------------------------------------------------


def validate_mask(mask_path: Path) -> Tuple[bool, str]:
    """Check that every pixel in *mask_path* is a known class ID or IGNORE_LABEL.

    Returns (is_valid, message).
    """
    arr = np.array(Image.open(mask_path))
    unique = set(np.unique(arr).tolist())
    bad = unique - VALID_CLASS_IDS - {IGNORE_LABEL}
    if bad:
        return False, f"unexpected pixel value(s) {sorted(bad)} in {mask_path.name}"
    return True, "ok"


def _copy_one(args: Tuple[Path, Path, Path, Path, bool]) -> str:
    """Copy one (image, mask) pair; validate the mask first. Returns a status string."""
    img_src, img_dst, mask_src, mask_dst, dry_run = args

    valid, msg = validate_mask(mask_src)
    if not valid:
        return f"warn:{msg}"

    if not dry_run:
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        mask_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_src, img_dst)
        shutil.copy2(mask_src, mask_dst)
    return f"ok:{img_src.name}"


# ---------------------------------------------------------------------------
# Split conversion
# ---------------------------------------------------------------------------


def convert_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    stems: List[str],
    workers: int = 1,
    dry_run: bool = False,
) -> int:
    """Copy all (image, mask) pairs for *stems* into <dst>/images|masks/<split>/.

    Returns the number of pairs successfully copied (or counted in dry-run).
    Stems with a missing image or mask counterpart are skipped with a warning.
    """
    img_src_dir = src_root / "JPEGImages"
    mask_src_dir = src_root / "SegmentationClass"
    img_dst_dir = dst_root / "images" / split
    mask_dst_dir = dst_root / "masks" / split

    tasks: List[Tuple[Path, Path, Path, Path, bool]] = []
    for stem in stems:
        img_src = img_src_dir / f"{stem}.jpg"
        mask_src = mask_src_dir / f"{stem}.png"
        if not img_src.exists():
            print(f"[WARN] Missing image for '{stem}' — skipping")
            continue
        if not mask_src.exists():
            print(f"[WARN] Missing mask for '{stem}' — skipping")
            continue
        tasks.append(
            (img_src, img_dst_dir / f"{stem}.jpg", mask_src, mask_dst_dir / f"{stem}.png", dry_run)
        )

    n_ok = 0
    if workers <= 1 or len(tasks) <= 1:
        for task in tasks:
            result = _copy_one(task)
            if result.startswith("ok"):
                n_ok += 1
            else:
                print(f"[WARN] {result}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_copy_one, t): t for t in tasks}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result.startswith("ok"):
                        n_ok += 1
                    else:
                        print(f"[WARN] {result}")
                except Exception as exc:
                    print(f"[ERROR] {futures[future][0].name}: {exc}")

    return n_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert AeroScapes (JPEGImages/ + SegmentationClass/) -> "
        "the shared CABiNet/YOLO images/+masks/ layout",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--src",
        required=True,
        type=Path,
        help="Root directory containing JPEGImages/, SegmentationClass/, "
        "and ImageSets/ (e.g. .../aeroscapes/)",
    )
    p.add_argument(
        "--dst",
        required=True,
        type=Path,
        help="Output directory for the converted dataset",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of parallel worker processes",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Walk the dataset, validate masks, and report counts without writing any files",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.src.exists():
        raise FileNotFoundError(f"--src does not exist: {args.src}")

    splits = discover_splits(args.src)
    for split, stems in splits.items():
        if not stems:
            print(f"[WARN] No stems found for split '{split}' — skipping")

    print(f"[INFO] Source: {args.src}")
    print(f"[INFO] train: {len(splits['train'])} images, val: {len(splits['val'])} images")
    print("[INFO] No source test split exists — only train/val will be produced.")

    total = 0
    for split, stems in splits.items():
        if not stems:
            continue
        n = convert_split(
            src_root=args.src,
            dst_root=args.dst,
            split=split,
            stems=stems,
            workers=args.workers,
            dry_run=args.dry_run,
        )
        verb = "would copy" if args.dry_run else "copied"
        print(f"[INFO] {split}: {verb} {n} pairs")
        total += n

    print(f"\n[DONE] Total pairs {'scanned' if args.dry_run else 'written'}: {total}")
    if not args.dry_run:
        print(f"       Output: {args.dst}")
        print(f"       Next: set AEROSCAPES_YOLO_ROOT={args.dst}")


if __name__ == "__main__":
    main()
