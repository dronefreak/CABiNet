#!/usr/bin/python
# -*- encoding: utf-8 -*-

import logging
from pathlib import Path

from PIL import Image
import hydra
import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.datasets.cityscapes import CityScapes
from src.models.cabinet import CABiNet
from src.utils.logger import RichConsoleManager

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
    """Multi-scale + flip inference on a single image tensor (C, H, W) or (1, C, H, W).

    Returns predicted label map (H, W).
    """
    model.eval()

    # Handle both (C, H, W) and (1, C, H, W)
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)  # → (1, C, H, W)

    img_tensor = img_tensor.to(device)
    B, C, H, W = img_tensor.shape
    assert B == 1, "Only supports one image at a time"

    probs = torch.zeros((1, 19, H, W), device=device)

    for scale in scales:
        if scale == 1.0:
            scaled_input = img_tensor
        else:
            new_size = [int(H * scale), int(W * scale)]
            scaled_input = F.interpolate(
                img_tensor,
                size=new_size,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            )

        # Forward pass
        out, _ = model(scaled_input)  # CABiNet outputs (B, 19, h, w)
        prob = F.softmax(out, dim=1)

        # Flip if enabled
        if flip:
            flipped_input = torch.flip(scaled_input, dims=(3,))
            flipped_out, _ = model(flipped_input)
            flipped_prob = F.softmax(torch.flip(flipped_out, dims=(3,)), dim=1)
            prob = (prob + flipped_prob) / 2

        # Resize back to original resolution
        if scale != 1.0:
            prob = F.interpolate(
                prob,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            )

        probs += prob

    # Average over scales
    probs /= len(scales)

    # Get prediction
    pred = torch.argmax(probs, dim=1).cpu().numpy()[0]  # (H, W)
    return pred


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

        # Extract first image in batch (since batch_size=1, it's safe)
        img_np = img[0].permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
        img_np = (img_np * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        # orig_size = img_pil.size  # (W, H)

        # Predict
        pred = infer_image(model, img, scales=[1.0], flip=False, device=device)

        # Colorize
        color_pred = colorize_mask(pred)
        overlay = Image.blend(img_pil, color_pred, alpha=0.6)

        # Save
        name = f"sample_{i:04d}"
        save_dir = Path(output_dir) / name
        save_dir.mkdir(parents=True, exist_ok=True)
        img_pil.save(save_dir / "input.png")
        color_pred.save(save_dir / "pred.png")
        overlay.save(save_dir / "overlay.png")

        if has_gt and show_gt:
            lb_np = lb[0].cpu().numpy()  # Remove batch dim
            gt_color = colorize_mask(lb_np)
            gt_color.save(save_dir / "gt.png")

        if i > 50:  # Limit for demo
            break

    console.print("✅ Visualization complete!")


@hydra.main(version_base=None, config_path="../../configs", config_name="train_citys")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    n_classes = cfg.dataset_config.num_classes
    mode = cfg.model.mode
    cfgs = cfg.model.cfgs
    model = CABiNet(n_classes=n_classes, backbone_weights=None, mode=mode, cfgs=cfgs)
    ckpt_path = (
        Path(cfg.training_config.experiments_path) / cfg.training_config.model_save_name
    )
    console.print(f"[INFO] Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    console.print("✅ Model loaded and set to eval mode.")

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
    vis_output = Path(cfg.validation_config.results_path) / "visualizations"
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
