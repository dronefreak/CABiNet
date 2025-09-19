# test_forward.py
import torch
from src.datasets.cityscapes import CityScapes
from src.models.cabinet import CABiNet

# Config (mimic your YAML)
params = {
    "dataset_config": {
        "dataset_config_file": "/home/neural_debugger/projects/CABiNet/configs/cityscapes_info.json",
        "ignore_idx": 255,
        "dataset_path": "/home/neural_debugger/Downloads/Cityscapes/",
        "cropsize": [512, 1024],
        "num_classes": 19,
    },
    "training_config": {
        "backbone_weights": "src/models/pretrained_backbones/mobilenetv3-small-55df8e1f.pth"
    }
}

# Dataset
ds = CityScapes(
    config_file=params["dataset_config"]["dataset_config_file"],
    ignore_lb=params["dataset_config"]["ignore_idx"],
    rootpth=params["dataset_config"]["dataset_path"],
    cropsize=params["dataset_config"]["cropsize"],
    mode="train"
)

# Model
model = CABiNet(n_classes=19, backbone_weights=params["training_config"]["backbone_weights"])
model.cuda()
model.eval()

# Get one sample
img, lb = ds[0]
print("Input image shape:", img.shape)
print("Label shape:", lb.shape)
print("Label unique values:", torch.unique(lb))

# Forward pass
with torch.no_grad():
    out, out16 = model(img.unsqueeze(0).cuda())

print("Output shapes:", out.shape, out16.shape)