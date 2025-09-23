"""
compute_class_weights_uavid.py
------------------------------
Compute class weights for UAVid trainId segmentation masks.

Dataset structure example:
UAVid-v1/train/seq1/TrainId/000000.png
UAVid-v1/train/seq2/TrainId/xxxxx.png
...

Each PNG is a single-channel trainId image (pixel values = class IDs).
"""

from pathlib import Path

from PIL import Image
import fire
import numpy as np
from tqdm import tqdm


def compute_class_weights(
    root_dir: str,
    num_classes: int = 8,
    method: str = "median",
    ignore_index: int | None = None,
    normalize: bool = True,
) -> list[float]:
    """Compute class weights from UAVid trainId masks.

    Parameters
    ----------
    root_dir : str
        Path to UAVid training root (e.g. 'UAVid-v1/train').
    num_classes : int
        Total number of classes (default: 8).
    method : str
        Weighting method: 'median', 'inverse', or 'log' (default: 'median').
    ignore_index : int | None
        Class index to ignore (e.g. background). Default None.
    normalize : bool
        If True, scale weights so mean(weight)=1.

    Returns
    -------
    list[float]
        Computed class weights (length = num_classes).
    """
    class_pixel_counts = np.zeros(num_classes, dtype=np.int64)
    total_pixels = 0

    # Find all TrainId PNGs recursively
    label_files = list(Path(root_dir).glob("seq*/TrainId/*.png"))
    if not label_files:
        raise RuntimeError(f"No TrainId PNGs found in {root_dir}")

    for lbl_path in tqdm(label_files, desc="Scanning masks"):
        mask = np.array(Image.open(lbl_path))
        for c in range(num_classes):
            if ignore_index is not None and c == ignore_index:
                continue
            class_pixel_counts[c] += np.sum(mask == c)
        total_pixels += mask.size

    # Compute frequencies
    freqs = class_pixel_counts / max(total_pixels, 1)

    if method == "inverse":
        weights = 1.0 / (freqs + 1e-6)
    elif method == "median":
        median_freq = np.median(freqs[freqs > 0])
        weights = median_freq / (freqs + 1e-6)
    elif method == "log":
        k = 1.02
        weights = 1.0 / np.log(k + freqs)
    else:
        raise ValueError("method must be one of: 'median', 'inverse', 'log'")

    if normalize:
        weights = weights / np.mean(weights[class_pixel_counts > 0])

    print("Class pixel counts:", class_pixel_counts)
    print("Class frequencies :", np.round(freqs, 6))
    print("Class weights     :", np.round(weights, 6))

    return weights.tolist()


if __name__ == "__main__":
    fire.Fire({"compute": compute_class_weights})
