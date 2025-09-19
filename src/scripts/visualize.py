#!/usr/bin/python
# -*- encoding: utf-8 -*-

import logging
import os
from pathlib import Path

from PIL import Image
import hydra
import numpy as np
from omegaconf import DictConfig
import torch
from tqdm import tqdm

from src.datasets.cityscapes import CityScapes
from src.models.cabinet import CABiNet
from src.utils import RichConsoleManager

console = RichConsoleManager.get_console()
logger = logging.getLogger(__name__)

# 🎨 Cityscapes Color Palette (19 classes)
CITYSCAPES_COLORS = np.array(
    [
        [128, 64, 128],  # road
        [244, 35, 232],  # sidewalk
        [70, 70, 70],  # building
        [102, 102, 156],  # wall
        [190, 153, 153],  # fence
        [153, 153, 153],  # pole
        [250, 170, 30],  # traffic light
        [220, 220, 0],  # traffic sign
        [107, 142, 35],  # vegetation
        [152, 251, 152],  # terrain
        [70, 130, 180],  # sky
        [220, 20, 60],  # person
        [255, 0, 0],  # rider
        [0, 0, 142],  # car
        [0, 0, 70],  # truck
        [0, 60, 100],  # bus
        [0, 80, 100],  # train
        [0, 0, 230],  # motorcycle
        [119, 11, 32],  # bicycle
    ],
    dtype=np.uint8,
)


def colorize_mask(mask: np.ndarray) -> Image.Image:
    """Convert integer label mask to colorized RGB image.

    Args:
        mask: (H, W), values in [0, 18] or 255 (ignored)
    Returns:
        PIL Image (H, W, 3)
    """
    h, w = mask.shape
    mask_clipped = np.clip(mask, 0, 18).astype(np.int64)
    colored = CITYSCAPES_COLORS[mask_clipped.ravel()].reshape(h, w, 3)
    return Image.fromarray(colored, mode="RGB")


@torch.no_grad()
def infer_image(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    scales: list = [1.0],
    flip: bool = False,
    device: torch.device = "cuda",
):
    """Multi-scale + flip inference."""
    model.eval()
    img_tensor = img_tensor.to(device)

    probs = torch.zeros((1, 19, *img_tensor.shape[2:]), device=device)

    for scale in scales:
        if scale != 1.0:
            h, w = img_tensor.shape[2:]
            scaled_size = (int(h * scale), int(w * scale))
            scaled_img = torch.nn.functional.interpolate(
                img_tensor, size=scaled_size, mode="bilinear", align_corners=False
            )
        else:
            scaled_img = img_tensor

        out, _ = model(scaled_img)
        prob = torch.nn.functional.softmax(out, dim=1)

        if flip:
            flipped_img = torch.flip(scaled_img, dims=(3,))
            flipped_out, _ = model(flipped_img)
            flipped_prob = torch.nn.functional.softmax(
                torch.flip(flipped_out, dims=(3,)), dim=1
            )
            prob = (prob + flipped_prob) / 2

        if scale != 1.0:
            prob = torch.nn.functional.interpolate(
                prob, size=(h, w), mode="bilinear", align_corners=False
            )

        probs += prob

    preds = torch.argmax(probs, dim=1).cpu().numpy()[0]  # (H, W)
    return preds


def visualize_predictions(
    model: torch.nn.Module,
    dataloader,
    output_dir: Path,
    device: torch.device,
    show_gt: bool = True,
    use_gt_for_overlay: bool = False,
):
    """Run visualization over entire dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"💾 Saving visualizations to: {output_dir}")

    for i, data in enumerate(tqdm(dataloader, desc="Visualizing")):
        if len(data) == 2:
            img, lb = data
            has_gt = True
        else:
            img = data[0]
            lb = None
            has_gt = False

        img_pil = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        # orig_size = img_pil.size  # (W, H)

        # Predict
        pred = infer_image(
            model, img.unsqueeze(0), scales=[1.0], flip=False, device=device
        )

        # Colorize
        color_pred = colorize_mask(pred)
        overlay = Image.blend(img_pil, color_pred, alpha=0.6)

        # Save
        name = f"sample_{i:04d}"
        img_pil.save(output_dir / f"{name}_input.png")
        color_pred.save(output_dir / f"{name}_pred.png")
        overlay.save(output_dir / f"{name}_overlay.png")

        if has_gt and show_gt:
            gt_color = colorize_mask(lb.numpy())
            gt_color.save(output_dir / f"{name}_gt.png")

        if i > 50:  # Limit for demo
            break

    console.print("✅ Visualization complete!")


@hydra.main(version_base=None, config_path="../configs", config_name="train_citys")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    n_classes = cfg.dataset_config.num_classes
    model = CABiNet(n_classes=n_classes)
    ckpt_path = (
        Path(cfg.training_config.experiments_path) / cfg.training_config.model_save_name
    )
    console.print(f"[INFO] Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Setup dataset
    ds_val = CityScapes(
        config_file=cfg.dataset_config.dataset_config_file,
        ignore_lb=cfg.dataset_config.ignore_idx,
        rootpth=cfg.dataset_config.dataset_path,
        cropsize=cfg.dataset_config.cropsize,
        mode="val",
    )
    dl_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Output dir
    vis_output = Path(cfg.get("vis_output", "visualizations/results"))
    vis_output.mkdir(parents=True, exist_ok=True)

    # Run
    visualize_predictions(
        model=model,
        dataloader=dl_val,
        output_dir=vis_output,
        device=device,
        show_gt=True,
    )


if __name__ == "__main__":
    main()
