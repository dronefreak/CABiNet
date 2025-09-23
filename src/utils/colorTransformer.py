"""
colorTransformer.py
-------------------
Provides the UAVidColorTransformer class for converting UAVid color-coded
segmentation labels to integer train IDs and back.

Usage Example:
--------------
from colorTransformer import UAVidColorTransformer
transformer = UAVidColorTransformer()
train_id_map = transformer.transform(rgb_label)
rgb_label_back = transformer.inverse_transform(train_id_map)
"""

import numpy as np


class UAVidColorTransformer:
    """Convert UAVid dataset color labels to integer IDs and vice versa."""

    def __init__(self) -> None:
        self._color_table = self._create_color_table()
        # Precompute a mapping from color (encoded as single int) to class ID.
        self._id_table = {
            name: self._color_to_int(rgb) for name, rgb in self._color_table.items()
        }

    @staticmethod
    def _create_color_table() -> dict[str, list[int]]:
        """Return a mapping of class names to their RGB colors."""
        return {
            "Clutter": [0, 0, 0],
            "Building": [128, 0, 0],
            "Road": [128, 64, 128],
            "Static_Car": [192, 0, 192],
            "Tree": [0, 128, 0],
            "Vegetation": [128, 128, 0],
            "Human": [64, 64, 0],
            "Moving_Car": [64, 0, 128],
        }

    @staticmethod
    def _color_to_int(rgb: list[int]) -> int:
        """Encode an RGB triplet into a single integer for fast comparison."""
        return rgb[0] + rgb[1] * 255 + rgb[2] * 255 * 255

    @property
    def color_table(self) -> dict[str, list[int]]:
        """Get the color table (class name → RGB)."""
        return self._color_table

    def transform(self, label: np.ndarray, dtype: np.dtype = np.int32) -> np.ndarray:
        """Convert an RGB label image to a 2-D array of class IDs.

        Parameters
        ----------
        label : np.ndarray
            Input H×W×3 uint8 RGB label image.
        dtype : np.dtype, optional
            Output data type (default: np.int32).

        Returns
        -------
        np.ndarray
            H×W array where each pixel is the class ID [0..num_classes-1].
        """
        height, width = label.shape[:2]
        new_label = np.zeros((height, width), dtype=dtype)

        # Encode each pixel color as a single int for fast comparison
        id_label = (
            label[:, :, 0].astype(np.int64)
            + label[:, :, 1].astype(np.int64) * 255
            + label[:, :, 2].astype(np.int64) * 255 * 255
        )

        for class_id, name in enumerate(self._color_table.keys()):
            color_int = self._id_table[name]
            new_label[id_label == color_int] = class_id

        return new_label

    def inverse_transform(self, label: np.ndarray) -> np.ndarray:
        """Convert a class ID map back to an RGB label image.

        Parameters
        ----------
        label : np.ndarray
            H×W array of integer class IDs.

        Returns
        -------
        np.ndarray
            H×W×3 uint8 RGB image.
        """
        h, w = label.shape
        rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
        colors = list(self._color_table.values())

        for class_id, rgb in enumerate(colors):
            rgb_img[label == class_id] = rgb

        return rgb_img
