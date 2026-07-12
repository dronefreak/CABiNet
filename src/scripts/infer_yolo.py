#!/usr/bin/env python3
"""Inference + visualization for trained YOLO semantic segmentation models.

Accepts a single image, a folder of images, a single video, or a folder of
videos (any mix in one folder is fine — files are routed by extension). For
each input it runs the model, colorizes the predicted per-pixel class map
using the dataset's RGB palette, and writes:
  - a colored mask (pure prediction, no input image blended in)
  - a simple alpha-blended overlay on top of the original image/frame

The overlay in ``overlay_mask()`` is intentionally a plain alpha blend for
now — a placeholder to iterate on once we decide how we want the "final"
look (edge highlighting, class legend, per-class alpha, etc.).

Usage
-----
  # Single image
  python src/scripts/infer_yolo.py \\
      --weights runs/uavid/yolo26n/weights/best.pt \\
      --source  path/to/image.png \\
      --output  outputs/inference

  # Folder of images
  python src/scripts/infer_yolo.py \\
      --weights runs/uavid/yolo26n/weights/best.pt \\
      --source  path/to/images/ \\
      --output  outputs/inference

  # Single video
  python src/scripts/infer_yolo.py \\
      --weights runs/uavid/yolo26n/weights/best.pt \\
      --source  path/to/clip.mp4 \\
      --output  outputs/inference

  # Folder of videos (or a folder mixing images and videos)
  python src/scripts/infer_yolo.py \\
      --weights runs/uavid/yolo26n/weights/best.pt \\
      --source  path/to/videos/ \\
      --output  outputs/inference

Output layout
--------------
  <output>/
    images/
      <stem>_mask.png
      <stem>_overlay.png
    videos/
      <video_stem>/
        mask.mp4
        overlay.mp4

Showcase mosaic
---------------
  python src/scripts/infer_yolo.py \\
      --weights runs/uavid/yolo26n/weights/best.pt \\
      --showcase-videos clip1.mp4 clip2.mp4 clip3.mp4 clip4.mp4 \\
      --output  outputs/inference

Runs inference on all 4 clips and tiles them into a single 2x2 mosaic
video at ``<output>/showcase_mosaic.mp4``. Each clip is blended from its
raw frame (start) to its full colorized segmentation mask (end) via a
linear temporal ramp, so the reveal progresses independently per quadrant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.utils.logger import RichConsoleManager

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------


def load_palette(info_path: str | Path) -> np.ndarray:
    """Build a (num_classes, 3) uint8 BGR palette from a *_info.json file.

    Classes are ordered by ``trainId`` so palette[class_id] gives the colour
    for that class — this matches the YOLO dataset's class-ID ordering
    (see convert_uavid_to_yolo.py, which uses the same trainId ordering).
    """
    with open(info_path) as f:
        labels_info = json.load(f)
    ordered = sorted(labels_info, key=lambda c: c["trainId"])
    rgb = np.array([c["color"] for c in ordered], dtype=np.uint8)
    return rgb[:, ::-1]  # RGB -> BGR (OpenCV convention used throughout this script)


# ---------------------------------------------------------------------------
# Inference + colorization
# ---------------------------------------------------------------------------


def predict_class_map(model, image_bgr: np.ndarray, imgsz: int, device) -> np.ndarray:
    """Run the semantic segmentation model on a single BGR image/frame.

    Returns
    -------
    np.ndarray
        (H, W) integer class-ID map at the input image's original resolution.
    """
    results = model.predict(
        source=image_bgr, imgsz=imgsz, device=device, task="semantic", verbose=False
    )
    mask = results[0].semantic_mask
    if mask is None:
        raise RuntimeError(
            "Model produced no semantic_mask output — is this a semantic "
            "segmentation ('-sem') checkpoint?"
        )
    # mask is an Ultralytics SemanticMask (BaseTensor subclass): .cpu()/.numpy()
    # each return a new wrapper object, not the raw array, so unwrap via .data.
    return mask.cpu().numpy().data


def colorize_mask(class_map: np.ndarray, palette_bgr: np.ndarray) -> np.ndarray:
    """Map a (H, W) class-ID array to a (H, W, 3) BGR colour image."""
    safe = np.clip(class_map, 0, len(palette_bgr) - 1).astype(np.int64)
    return palette_bgr[safe]


def overlay_mask(
    image_bgr: np.ndarray, colored_mask_bgr: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """Alpha-blend the colored mask on top of the original image.

    Placeholder blending strategy — swap this out once we settle on how the
    final overlay should look (e.g. edge-only highlighting, per-class alpha,
    class legend overlay).
    """
    return cv2.addWeighted(image_bgr, 1.0 - alpha, colored_mask_bgr, alpha, 0)


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------


def discover_sources(
    source: Path, recursive: bool = False
) -> tuple[list[Path], list[Path]]:
    """Return (image_paths, video_paths) found at *source*.

    *source* may be a single image file, a single video file, or a directory
    containing any mix of images and videos.
    """
    if source.is_file():
        ext = source.suffix.lower()
        if ext in IMAGE_EXTS:
            return [source], []
        if ext in VIDEO_EXTS:
            return [], [source]
        raise ValueError(f"Unsupported file extension: {source}")

    if not source.is_dir():
        raise FileNotFoundError(f"Source not found: {source}")

    pattern = "**/*" if recursive else "*"
    images: list[Path] = []
    videos: list[Path] = []
    for p in sorted(source.glob(pattern)):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in IMAGE_EXTS:
            images.append(p)
        elif ext in VIDEO_EXTS:
            videos.append(p)
    return images, videos


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------


def process_image(
    model,
    img_path: Path,
    out_dir: Path,
    palette: np.ndarray,
    imgsz: int,
    device,
    alpha: float,
    save_mask: bool,
    save_overlay: bool,
) -> None:
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"[WARN] Could not read image, skipping: {img_path}")
        return

    class_map = predict_class_map(model, image, imgsz, device)
    colored = colorize_mask(class_map, palette)

    stem = img_path.stem
    if save_mask:
        cv2.imwrite(str(out_dir / f"{stem}_mask.png"), colored)
    if save_overlay:
        overlay = overlay_mask(image, colored, alpha)
        cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), overlay)


def process_video(
    model,
    video_path: Path,
    out_dir: Path,
    palette: np.ndarray,
    imgsz: int,
    device,
    alpha: float,
    save_mask: bool,
    save_overlay: bool,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open video, skipping: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_out_dir = out_dir / video_path.stem
    video_out_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    mask_writer = (
        cv2.VideoWriter(str(video_out_dir / "mask.mp4"), fourcc, fps, (width, height))
        if save_mask
        else None
    )
    overlay_writer = (
        cv2.VideoWriter(
            str(video_out_dir / "overlay.mp4"), fourcc, fps, (width, height)
        )
        if save_overlay
        else None
    )

    pbar = tqdm(
        total=n_frames if n_frames > 0 else None,
        desc=f"Processing {video_path.name}",
    )
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            class_map = predict_class_map(model, frame, imgsz, device)
            colored = colorize_mask(class_map, palette)

            if mask_writer is not None:
                mask_writer.write(colored)
            if overlay_writer is not None:
                overlay_writer.write(overlay_mask(frame, colored, alpha))

            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        if mask_writer is not None:
            mask_writer.release()
        if overlay_writer is not None:
            overlay_writer.release()


def _open_video_or_raise(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    return cap


def build_showcase_mosaic(
    model,
    video_paths: list[Path],
    out_path: Path,
    palette: np.ndarray,
    imgsz: int,
    device,
    mosaic_scale: float,
    fps: float | None,
) -> None:
    """Build a 2x2 showcase mosaic video from 4 clips.

    Each clip is blended from its raw RGB frame (start, alpha=1/beta=0) to
    its full colorized segmentation mask (end, alpha=0/beta=1) via a linear
    per-clip temporal ramp, then all 4 blended clips are tiled into one
    2x2 grid. Each quadrant is resized to ``mosaic_scale`` of the first
    clip's native resolution before tiling — at the default 0.5, the
    assembled mosaic ends up the same total resolution as one native clip
    (4x fewer pixels than tiling all 4 clips at native resolution).

    Clips of different lengths freeze on their last blended frame once
    they run out, rather than ending the mosaic early — the mosaic's
    duration is the longest of the 4 clips.
    """
    if len(video_paths) != 4:
        raise ValueError(
            f"Showcase mosaic requires exactly 4 videos, got {len(video_paths)}"
        )

    caps = [_open_video_or_raise(p) for p in video_paths]
    frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
    reliable_counts = [c for c in frame_counts if c > 0]
    if not reliable_counts:
        raise RuntimeError(
            "None of the 4 showcase videos report a usable frame count; "
            "cannot determine the temporal reveal ramp or mosaic duration."
        )
    max_frames = max(reliable_counts)
    # A clip with an unreliable/missing frame count ramps over the mosaic's
    # overall duration instead of its own (unknown) length.
    frame_counts = [c if c > 0 else max_frames for c in frame_counts]

    first_w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    first_h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Round to even numbers — required by most mp4 codecs.
    quad_w = max(2, int(round(first_w * mosaic_scale / 2)) * 2)
    quad_h = max(2, int(round(first_h * mosaic_scale / 2)) * 2)

    out_fps = fps or caps[0].get(cv2.CAP_PROP_FPS) or 25.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (quad_w * 2, quad_h * 2))

    finished = [False, False, False, False]
    last_quadrant = [np.zeros((quad_h, quad_w, 3), dtype=np.uint8) for _ in range(4)]

    try:
        for frame_idx in tqdm(range(max_frames), desc="Building showcase mosaic"):
            for i, cap in enumerate(caps):
                if finished[i]:
                    continue
                ok, frame = cap.read()
                if not ok:
                    finished[i] = True
                    continue

                class_map = predict_class_map(model, frame, imgsz, device)
                colored = colorize_mask(class_map, palette)
                frame_small = cv2.resize(frame, (quad_w, quad_h))
                colored_small = cv2.resize(
                    colored, (quad_w, quad_h), interpolation=cv2.INTER_NEAREST
                )

                denom = max(frame_counts[i] - 1, 1)
                progress = min(frame_idx / denom, 1.0)
                last_quadrant[i] = overlay_mask(
                    frame_small, colored_small, alpha=progress
                )

            top = np.hstack((last_quadrant[0], last_quadrant[1]))
            bottom = np.hstack((last_quadrant[2], last_quadrant[3]))
            writer.write(np.vstack((top, bottom)))
    finally:
        writer.release()
        for cap in caps:
            cap.release()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run a trained YOLO semantic segmentation model over an image, "
            "an image folder, a video, or a folder of videos, and save "
            "colorized masks + overlays."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--weights",
        required=True,
        type=Path,
        help="Path to a trained checkpoint (e.g. runs/uavid/yolo26n/weights/best.pt)",
    )
    p.add_argument(
        "--source",
        default=None,
        type=Path,
        help=(
            "Image file, video file, or a folder containing images/videos. "
            "Required unless --showcase-videos is given."
        ),
    )
    p.add_argument(
        "--showcase-videos",
        nargs=4,
        type=Path,
        default=None,
        metavar=("VIDEO1", "VIDEO2", "VIDEO3", "VIDEO4"),
        help=(
            "Build a 2x2 showcase mosaic from exactly 4 videos instead of "
            "the normal --source flow. Each clip is blended from raw frame "
            "to full segmentation mask over its own duration."
        ),
    )
    p.add_argument(
        "--mosaic-scale",
        type=float,
        default=0.5,
        help=(
            "Per-quadrant resize factor (relative to the first showcase "
            "video's native resolution) applied before tiling into the "
            "2x2 mosaic."
        ),
    )
    p.add_argument(
        "--mosaic-fps",
        type=float,
        default=None,
        help="Output FPS for the showcase mosaic (default: first video's FPS)",
    )
    p.add_argument(
        "--output",
        default=Path("outputs/inference"),
        type=Path,
        help="Output directory",
    )
    p.add_argument(
        "--info",
        default="configs/UAVid_info.json",
        type=Path,
        help="Path to the *_info.json file defining the class palette",
    )
    p.add_argument("--imgsz", type=int, default=1024, help="Inference image size")
    p.add_argument(
        "--device", default="0", help="Device for inference ('0', 'cpu', '0,1', ...)"
    )
    p.add_argument(
        "--alpha", type=float, default=0.5, help="Overlay blend strength (0-1)"
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into sub-directories when --source is a folder",
    )
    p.add_argument(
        "--no-mask",
        action="store_true",
        help="Skip saving the pure colorized mask output",
    )
    p.add_argument(
        "--no-overlay",
        action="store_true",
        help="Skip saving the blended overlay output",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is not installed.\n"
            "Install it with:  pip install ultralytics\n"
            "Or add it to the project:  pip install -e '.[yolo]'"
        ) from exc

    console = RichConsoleManager.get_console()

    if args.showcase_videos is None and args.source is None:
        raise ValueError("--source is required unless --showcase-videos is given.")

    save_mask = not args.no_mask
    save_overlay = not args.no_overlay
    if not save_mask and not save_overlay:
        raise ValueError("Both --no-mask and --no-overlay were set; nothing to do.")

    console.print(f"Loading model: {args.weights}", style="info")
    model = YOLO(str(args.weights))

    palette = load_palette(args.info)
    console.print(f"Loaded {len(palette)}-class palette from {args.info}", style="info")

    if args.showcase_videos is not None:
        mosaic_out = args.output / "showcase_mosaic.mp4"
        console.print(
            f"Building showcase mosaic from {len(args.showcase_videos)} videos "
            f"-> {mosaic_out}",
            style="info",
        )
        build_showcase_mosaic(
            model,
            args.showcase_videos,
            mosaic_out,
            palette,
            args.imgsz,
            args.device,
            args.mosaic_scale,
            args.mosaic_fps,
        )
        console.print(f"Done. Showcase mosaic saved to: {mosaic_out}", style="info")
        return

    images, videos = discover_sources(args.source, recursive=args.recursive)
    if not images and not videos:
        raise FileNotFoundError(f"No images or videos found at: {args.source}")
    console.print(
        f"Found {len(images)} image(s) and {len(videos)} video(s) in {args.source}",
        style="info",
    )

    if images:
        img_out_dir = args.output / "images"
        img_out_dir.mkdir(parents=True, exist_ok=True)
        for img_path in tqdm(images, desc="Images"):
            process_image(
                model,
                img_path,
                img_out_dir,
                palette,
                args.imgsz,
                args.device,
                args.alpha,
                save_mask,
                save_overlay,
            )

    if videos:
        vid_out_dir = args.output / "videos"
        vid_out_dir.mkdir(parents=True, exist_ok=True)
        for video_path in videos:
            process_video(
                model,
                video_path,
                vid_out_dir,
                palette,
                args.imgsz,
                args.device,
                args.alpha,
                save_mask,
                save_overlay,
            )

    console.print(f"Done. Outputs saved to: {args.output}", style="info")


if __name__ == "__main__":
    main()
