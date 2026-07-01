#!/usr/bin/env python3
"""Convert UAVid RGB colour-coded label masks to YOLO single-channel format.

UAVid original format (Labels/ directory)
------------------------------------------
Each ground-truth PNG is a 3-channel RGB image where every pixel's colour
encodes a semantic class (see UAVid_info.json). Example:
  [128,  0,  0]  → Building
  [  0,  0,  0]  → Clutter  (ignored in evaluation)

UAVid directory layout (as distributed)
-----------------------------------------
Both train and validation sequences live under the same root folder::

    uavid_train/            ← *--src* argument
    ├── seq1/
    │   ├── Images/         ← RGB input images
    │   └── Labels/         ← RGB colour-coded masks  ← what this script reads
    ├── seq2/ … seq15/
    └── seq16/              ← validation sequences (named via --val-seqs)

YOLO semantic-segmentation format
----------------------------------
Each mask is a *single-channel* (mode-"L") PNG where:
  pixel value = class index  (0 … N-1)
  pixel value = 255          → ignored during training / evaluation

Mapping used by this script (YOLO only — NOT the same as CABiNet trainIds)
---------------------------------------------------------------------------
  Clutter  (ignoreInEval=true)  → 255  (YOLO ignore label)
  Building                      → 0
  Road                          → 1
  Static Car                    → 2
  Tree                          → 3
  Vegetation                    → 4
  Human                         → 5
  Moving Car                    → 6

Note: in the CABiNet training pipeline (``uavid.py``) Clutter maps to
trainId=0 and is included in the loss.  The YOLO mapping here drops Clutter
entirely (→ 255) which gives 7 active classes.

Usage
------
  # Convert all sequences; val = seq16, train = everything else
  python src/scripts/convert_uavid_to_yolo.py \\
      --src  /data/uavid_train \\
      --dst  /data/uavid_yolo \\
      --info configs/UAVid_info.json \\
      --val-seqs seq16 \\
      --workers 4

  # Explicit train list
  python src/scripts/convert_uavid_to_yolo.py \\
      --src  /data/uavid_train \\
      --dst  /data/uavid_yolo \\
      --train-seqs seq1 seq2 seq3 seq4 seq5 \\
      --val-seqs   seq16 \\
      --workers 4

  # Dry-run to check counts before writing
  python src/scripts/convert_uavid_to_yolo.py \\
      --src /data/uavid_train --dst /tmp/out \\
      --val-seqs seq16 --dry-run

Directory layout produced
--------------------------
  <dst>/
    images/
      train/   ← symlinks (or copies with --copy-images) to original RGB PNGs
      val/
    masks/
      train/   ← converted single-channel PNGs  (seq_{stem}.png)
      val/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

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


def build_trainid_lut(
    labels_info: list[dict], ignore_lb: int = IGNORE_LABEL
) -> np.ndarray:
    """Build a (256, 256, 256) uint8 LUT mapping RGB colour → CABiNet trainId.

    Unlike :func:`build_lut` (which uses the YOLO 0-based mapping and sends
    Clutter to 255), this function preserves the original ``trainId`` values
    from ``UAVid_info.json`` (Clutter=0, Building=1, …, MovingCar=7).
    Unknown colours default to *ignore_lb* (255).

    This LUT is used by ``uavid.py`` for on-the-fly conversion during
    CABiNet training so that Clutter is included in the loss.
    """
    lut = np.full((256, 256, 256), ignore_lb, dtype=np.uint8)
    for cls in labels_info:
        r, g, b = cls["color"]
        lut[r, g, b] = cls["trainId"]
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
# Dataset structure helpers
# ---------------------------------------------------------------------------


def discover_sequences(src_root: Path) -> List[str]:
    """Return sorted list of sequence folder names found directly under *src_root*."""
    return sorted(
        d.name for d in src_root.iterdir() if d.is_dir() and (d / "Images").exists()
    )


def iter_sequences(src_root: Path, seqs: List[str]) -> List[Tuple[Path, str, str]]:
    """Return (image_path, seq_name, stem) tuples for a list of sequences.

    Expected layout::

        src_root/
        └── <seq>/
            ├── Images/*.png   ← what we iterate
            └── Labels/*.png   ← read by convert_mask

    Parameters
    ----------
    src_root:
        Directory that directly contains sequence sub-folders
        (i.e. ``uavid_train/``).
    seqs:
        Sequence folder names to include (e.g. ``["seq1", "seq2"]``).
    """
    results: List[Tuple[Path, str, str]] = []
    for seq_name in sorted(seqs):
        img_dir = src_root / seq_name / "Images"
        if not img_dir.exists():
            raise FileNotFoundError(
                f"Images/ directory not found for sequence '{seq_name}': {img_dir}"
            )
        for img_path in sorted(img_dir.glob("*.png")):
            results.append((img_path, seq_name, img_path.stem))
    return results


def setup_image_links(
    src_root: Path,
    dst_root: Path,
    split: str,
    seqs: List[str],
    copy: bool = False,
) -> None:
    """Create symlinks (or copies) for RGB images into <dst>/images/<split>/."""
    img_out_dir = dst_root / "images" / split
    img_out_dir.mkdir(parents=True, exist_ok=True)

    for img_path, seq, stem in iter_sequences(src_root, seqs):
        out_name = f"{seq}_{stem}.png"  # unique across sequences
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


def convert_sequences(
    src_root: Path,
    dst_root: Path,
    split: str,
    seqs: List[str],
    lut: np.ndarray,
    workers: int = 1,
    dry_run: bool = False,
    copy_images: bool = False,
) -> int:
    """Convert all colour masks for *seqs* and set up image references.

    Parameters
    ----------
    src_root:
        Directory containing sequence sub-folders (``uavid_train/``).
    dst_root:
        Output root for the YOLO dataset.
    split:
        ``"train"`` or ``"val"`` — determines output sub-directory name.
    seqs:
        Sequence names to convert.
    lut:
        YOLO colour → class-ID lookup table.
    workers:
        Parallel worker processes.
    dry_run:
        If True, count files without writing.
    copy_images:
        Copy images instead of symlinking.

    Returns
    -------
    int
        Number of masks successfully converted (or counted in dry-run).
    """
    mask_out_dir = dst_root / "masks" / split
    if not dry_run:
        mask_out_dir.mkdir(parents=True, exist_ok=True)

    entries = iter_sequences(src_root, seqs)
    tasks: List[Tuple[Path, Path, np.ndarray, bool]] = []
    for img_path, seq, stem in entries:
        label_path = img_path.parent.parent / "Labels" / f"{stem}.png"
        if not label_path.exists():
            print(f"[WARN] No Labels/ mask for {seq}/{stem}.png — skipping")
            continue
        dst_mask = mask_out_dir / f"{seq}_{stem}.png"
        tasks.append((label_path, dst_mask, lut, dry_run))

    if not dry_run:
        setup_image_links(src_root, dst_root, split, seqs, copy=copy_images)

    n_ok = 0
    if workers <= 1 or len(tasks) <= 1:
        for task in tasks:
            if _worker(task).startswith("ok"):
                n_ok += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker, t): t for t in tasks}
            for future in as_completed(futures):
                try:
                    if future.result().startswith("ok"):
                        n_ok += 1
                except Exception as exc:
                    print(f"[ERROR] {futures[future][0].name}: {exc}")

    return n_ok


# ---------------------------------------------------------------------------
# Public API (importable by tests and uavid.py)
# ---------------------------------------------------------------------------


def load_labels_info(info_path: str | Path) -> list[dict]:
    """Load and return the UAVid_info.json label list."""
    with open(info_path) as f:
        return json.load(f)  # type: ignore[no-any-return]


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
        help=(
            "Root directory containing sequence sub-folders "
            "(e.g. /data/uavid_train — contains seq1/, seq2/, …)"
        ),
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
        "--val-seqs",
        nargs="+",
        default=["seq16"],
        metavar="SEQ",
        help=(
            "Sequence folder name(s) to treat as the validation split. "
            "All other sequences in --src are treated as training."
        ),
    )
    p.add_argument(
        "--train-seqs",
        nargs="+",
        default=None,
        metavar="SEQ",
        help=(
            "Explicit list of training sequences. "
            "If omitted, all sequences not in --val-seqs are used."
        ),
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

    if not args.src.exists():
        raise FileNotFoundError(f"--src does not exist: {args.src}")

    labels_info = load_labels_info(args.info)
    colour_map = build_colour_map(labels_info)
    lut = build_lut(colour_map)
    class_names = get_yolo_class_names(labels_info)

    print(f"[INFO] YOLO class mapping ({len(class_names)} classes):")
    for yolo_id, name in class_names.items():
        print(f"         {yolo_id:2d} → {name}")
    print(f"         {IGNORE_LABEL} → <ignore> (Clutter + unlabelled)")

    # Discover all sequences in src
    all_seqs = discover_sequences(args.src)
    val_seqs = args.val_seqs
    train_seqs: List[str] = (
        args.train_seqs
        if args.train_seqs is not None
        else [s for s in all_seqs if s not in set(val_seqs)]
    )

    print(f"\n[INFO] Source     : {args.src}")
    print(f"[INFO] Train seqs : {train_seqs}")
    print(f"[INFO] Val seqs   : {val_seqs}")

    splits = [("train", train_seqs), ("val", val_seqs)]
    total = 0
    for split_name, seqs in splits:
        if not seqs:
            print(f"[WARN] No sequences for split '{split_name}', skipping.")
            continue
        n = convert_sequences(
            src_root=args.src,
            dst_root=args.dst,
            split=split_name,
            seqs=seqs,
            lut=lut,
            workers=args.workers,
            dry_run=args.dry_run,
            copy_images=args.copy_images,
        )
        verb = "would convert" if args.dry_run else "converted"
        print(f"[INFO] {split_name}: {verb} {n} masks")
        total += n

    print(f"\n[DONE] Total masks {'scanned' if args.dry_run else 'written'}: {total}")
    if not args.dry_run:
        print(f"       Output: {args.dst}")
        print(
            f"       Next: set UAVID_YOLO_ROOT={args.dst} "
            "and run: yolo segment train cfg=configs/yolo/uavid_train.yaml"
        )


if __name__ == "__main__":
    main()
