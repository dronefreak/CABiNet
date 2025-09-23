"""
prepareTrainIdFiles.py
----------------------
Batch-convert UAVid ground-truth RGB label images into trainId (class-ID) images.

This version uses Google Fire for the command-line interface.

Installation:
-------------
pip install fire pillow tqdm numpy

Usage:
------
python prepareTrainIdFiles.py prepare_train_ids /path/to/labels /path/to/trainIds
"""

import os
import os.path as osp
from pathlib import Path

from PIL import Image
from colorTransformer import UAVidColorTransformer
import fire
import numpy as np
from tqdm import tqdm


def prepare_train_ids(source_dir: str, target_dir: str) -> None:
    """Convert all ground-truth RGB labels in `source_dir` to trainId images.

    Parameters
    ----------
    source_dir : str
        Root directory containing sequences (seqXX) with a 'Labels' folder.
    target_dir : str
        Root directory where trainId images will be saved.
    """
    transformer = UAVidColorTransformer()
    seq_dirs = [p for p in os.listdir(source_dir) if p.startswith("seq")]

    for seq in tqdm(seq_dirs, desc="Processing sequences"):
        label_dir = osp.join(source_dir, seq, "Labels")
        save_dir = osp.join(target_dir, seq, "TrainId")
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        for lbl_name in os.listdir(label_dir):
            lbl_path = osp.join(label_dir, lbl_name)
            out_path = osp.join(save_dir, lbl_name)

            rgb = np.array(Image.open(lbl_path))
            train_id = transformer.transform(rgb, dtype=np.uint8)
            Image.fromarray(train_id).save(out_path)


if __name__ == "__main__":
    fire.Fire({"prepare_train_ids": prepare_train_ids})
