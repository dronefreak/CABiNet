import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm  # pip install tqdm for a progress bar

root = Path(
    "/home/neural_debugger/Downloads/datasets/VDD/VDD-YOLO/images/train"
)  # <-- change to your converted VDD train images folder
to_tensor = transforms.ToTensor()  # converts to [0,1] float tensor

mean = torch.zeros(3)
std = torch.zeros(3)
num_pixels = 0

image_files = list(root.glob("*.jpg"))

for img_path in tqdm(image_files):
    img = Image.open(img_path).convert("RGB")
    tensor = to_tensor(img)  # [C,H,W], values in [0,1]
    num_pixels += tensor.numel() / 3  # total pixels per channel
    mean += tensor.sum(dim=[1, 2])
    std += (tensor**2).sum(dim=[1, 2])

mean /= num_pixels
std = (std / num_pixels - mean**2).sqrt()

print("\nVDD Train Statistics")
print("Mean :", mean.tolist())
print("Std  :", std.tolist())
