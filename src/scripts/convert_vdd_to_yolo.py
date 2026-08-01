#!/usr/bin/env python3
"""Convert VDD (Varied Drone Dataset) into the shared CABiNet/YOLO layout.

VDD original format (as distributed / downloaded from HF)
------------------------------------------------------------
    VDD/                      ← *--src* argument
    ├── train/
    │   ├── src/*.JPG          ← RGB input images (uniformly 4000x3000)
    │   └── gt/*.png           ← single-channel masks, pixel value = class ID
    ├── val/
    │   ├── src/*.JPG
    │   └── gt/*.png
    ├── test/
    │   ├── src/*.JPG
    │   └── gt/*.png
    └── metadata/{train,val,test}.txt  ← not used by this script (the
                                          train/val/test directory split
                                          already IS the split; test.txt
                                          also has a typo'd .JPG extension
                                          for gt/ paths that doesn't match
                                          the actual .png files on disk)

Masks are already single-channel (mode "L") class-ID PNGs — like AeroScapes
and unlike UAVid's RGB colour-coded masks — so no colour lookup table is
needed here, just a straight symlink.

Class mapping (VDD, see https://github.com/RussRobin/VDD)
-----------------------------------------------------------
   0  Other        4  Vehicle
   1  Wall         5  Roof
   2  Road         6  Water
   3  Vegetation

Output layout produced
------------------------
  <dst>/
    images/
      train/   ← symlinks to *.JPG, renamed to lowercase *.jpg
      val/
      test/
    masks/
      train/   ← symlinks to *.png (already single-channel)
      val/
      test/

Unlike convert_aeroscapes_to_yolo.py, this script symlinks by default (like
convert_uavid_to_yolo.py) — VDD is already published on HF, so there's no
need to produce a redistributable standalone copy.

Usage
------
  python src/scripts/convert_vdd_to_yolo.py \\
      --src /path/to/VDD --dst /path/to/vdd_yolo

  # Dry-run to check counts before writing
  python src/scripts/convert_vdd_to_yolo.py \\
      --src /path/to/VDD --dst /tmp/out --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

VALID_CLASS_IDS = frozenset(range(7))
IGNORE_LABEL: int = 255

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Per-file validation + link
# ---------------------------------------------------------------------------


def validate_mask(mask_path: Path) -> Tuple[bool, str]:
    """Check that every pixel in *mask_path* is a known class ID or IGNORE_LABEL."""
    arr = np.array(Image.open(mask_path))
    unique = set(np.unique(arr).tolist())
    bad = unique - VALID_CLASS_IDS - {IGNORE_LABEL}
    if bad:
        return False, f"unexpected pixel value(s) {sorted(bad)} in {mask_path.name}"
    return True, "ok"


def _link_one(img_src: Path, img_dst: Path, mask_src: Path, mask_dst: Path, dry_run: bool) -> str:
    valid, msg = validate_mask(mask_src)
    if not valid:
        return f"warn:{msg}"

    if not dry_run:
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        mask_dst.parent.mkdir(parents=True, exist_ok=True)
        if not img_dst.exists():
            img_dst.symlink_to(img_src.resolve())
        if not mask_dst.exists():
            mask_dst.symlink_to(mask_src.resolve())
    return f"ok:{img_src.name}"


# ---------------------------------------------------------------------------
# Split conversion
# ---------------------------------------------------------------------------


def discover_stems(split_dir: Path) -> List[str]:
    """Return sorted stems present in both <split_dir>/src/*.JPG and gt/*.png."""
    src_dir = split_dir / "src"
    gt_dir = split_dir / "gt"
    if not src_dir.exists() or not gt_dir.exists():
        return []
    src_stems = {p.stem for p in src_dir.iterdir() if p.suffix.lower() == ".jpg"}
    gt_stems = {p.stem for p in gt_dir.iterdir() if p.suffix.lower() == ".png"}
    return sorted(src_stems & gt_stems)


def convert_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    dry_run: bool = False,
) -> int:
    """Symlink all (image, mask) pairs for *split* into <dst>/images|masks/<split>/.

    Returns the number of pairs successfully linked (or counted in dry-run).
    Stems present on only one side are skipped with a warning.
    """
    split_dir = src_root / split
    src_dir = split_dir / "src"
    gt_dir = split_dir / "gt"
    img_dst_dir = dst_root / "images" / split
    mask_dst_dir = dst_root / "masks" / split

    src_stems = {p.stem: p for p in src_dir.iterdir() if p.suffix.lower() == ".jpg"} if src_dir.exists() else {}
    gt_stems = {p.stem: p for p in gt_dir.iterdir() if p.suffix.lower() == ".png"} if gt_dir.exists() else {}

    all_stems = sorted(set(src_stems) | set(gt_stems))
    n_ok = 0
    for stem in all_stems:
        img_src = src_stems.get(stem)
        mask_src = gt_stems.get(stem)
        if img_src is None:
            print(f"[WARN] Missing image for '{stem}' — skipping")
            continue
        if mask_src is None:
            print(f"[WARN] Missing mask for '{stem}' — skipping")
            continue
        result = _link_one(
            img_src, img_dst_dir / f"{stem}.jpg", mask_src, mask_dst_dir / f"{stem}.png", dry_run
        )
        if result.startswith("ok"):
            n_ok += 1
        else:
            print(f"[WARN] {result}")

    return n_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert VDD (train/val/test/{src,gt}) -> the shared "
        "CABiNet/YOLO images/+masks/ layout",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--src",
        required=True,
        type=Path,
        help="Root directory containing train/, val/, test/ (each with src/ and gt/)",
    )
    p.add_argument(
        "--dst",
        required=True,
        type=Path,
        help="Output directory for the converted dataset",
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

    print(f"[INFO] Source: {args.src}")

    total = 0
    for split in SPLITS:
        stems = discover_stems(args.src / split)
        if not stems:
            print(f"[WARN] No stems found for split '{split}' — skipping")
            continue
        n = convert_split(args.src, args.dst, split, dry_run=args.dry_run)
        verb = "would link" if args.dry_run else "linked"
        print(f"[INFO] {split}: {verb} {n} pairs")
        total += n

    print(f"\n[DONE] Total pairs {'scanned' if args.dry_run else 'written'}: {total}")
    if not args.dry_run:
        print(f"       Output: {args.dst}")
        print(f"       Next: set VDD_YOLO_ROOT={args.dst}")


if __name__ == "__main__":
    main()
