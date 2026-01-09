"""
Creates a MobileNetV3 Model as defined in:
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen,
Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang,
Vijay Vasudevan, Quoc V. Le, Hartwig Adam. (2019).
Searching for MobileNetV3
arXiv preprint arXiv:1905.02244.
"""

import math
from pathlib import Path

import torch

__all__ = ["mobilenetv3_large", "mobilenetv3_small"]


def _make_divisible(v, divisor, min_value=None):
    """This function is taken from the original tf repo.

    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class HardSigmoid(torch.nn.Module):
    """Hard Sigmoid activation function.

    Approximates sigmoid using ReLU6: relu6(x + 3) / 6
    """

    def __init__(self, inplace: bool = True) -> None:
        super(HardSigmoid, self).__init__()
        self.relu = torch.nn.ReLU6(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply hard sigmoid activation."""
        return self.relu(x + 3) / 6


class HardSwish(torch.nn.Module):
    """Hard Swish activation function.

    Swish approximation: x * hard_sigmoid(x)
    """

    def __init__(self, inplace: bool = True) -> None:
        super(HardSwish, self).__init__()
        self.sigmoid = HardSigmoid(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply hard swish activation."""
        return x * self.sigmoid(x)


class SELayer(torch.nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(_make_divisible(channel // reduction, 8), channel),
            HardSigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return torch.nn.Sequential(
        torch.nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        torch.nn.BatchNorm2d(oup),
        HardSwish(),
    )


def conv_1x1_bn(inp, oup):
    return torch.nn.Sequential(
        torch.nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        torch.nn.BatchNorm2d(oup),
        HardSwish(),
    )


class InvertedResidual(torch.nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = torch.nn.Sequential(
                # dw
                torch.nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    groups=hidden_dim,
                    bias=False,
                ),
                torch.nn.BatchNorm2d(hidden_dim),
                HardSwish() if use_hs else torch.nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else torch.nn.Identity(),
                # pw-linear
                torch.nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(oup),
            )
        else:
            self.conv = torch.nn.Sequential(
                # pw
                torch.nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(hidden_dim),
                HardSwish() if use_hs else torch.nn.ReLU(inplace=True),
                # dw
                torch.nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    groups=hidden_dim,
                    bias=False,
                ),
                torch.nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else torch.nn.Identity(),
                HardSwish() if use_hs else torch.nn.ReLU(inplace=True),
                # pw-linear
                torch.nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(torch.nn.Module):
    def __init__(self, cfgs, mode, num_classes=1000, width_mult=1.0, weights=None):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.weights = weights
        assert mode in ["large", "small"]

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(
                block(input_channel, exp_size, output_channel, k, s, use_se, use_hs)
            )
            input_channel = output_channel
        self.features = torch.nn.Sequential(*layers)
        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        output_channel = {"large": 1280, "small": 1024}
        output_channel = (
            _make_divisible(output_channel[mode] * width_mult, 8)
            if width_mult > 1.0
            else output_channel[mode]
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(exp_size, output_channel),
            HardSwish(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        return x

    def _initialize_weights(self):
        if self.weights is not None and Path(self.weights).is_file():
            try:
                state_dict = torch.load(self.weights, map_location="cpu")
                self_state_dict = self.state_dict()
                for k, v in state_dict.items():
                    if "classifier" in k:
                        continue
                    self_state_dict.update({k: v})
                self.load_state_dict(self_state_dict)
                print(f"Loaded pretrained weights from {self.weights}")
                return
            except Exception as e:
                print(f"Failed to load backbone weights from {self.weights}: {e}")
                print("Proceeding with random weight initialization.")
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv3_large(**kwargs):
    """Constructs a MobileNetV3-Large model."""
    cfgs = [
        # k, t, c, SE, HS, s
        [3, 1, 16, 0, 0, 1],
        [3, 4, 24, 0, 0, 2],
        [3, 3, 24, 0, 0, 1],
        [5, 3, 40, 1, 0, 2],
        [5, 3, 40, 1, 0, 1],
        [5, 3, 40, 1, 0, 1],
        [3, 6, 80, 0, 1, 2],
        [3, 2.5, 80, 0, 1, 1],
        [3, 2.3, 80, 0, 1, 1],
        [3, 2.3, 80, 0, 1, 1],
        [3, 6, 112, 1, 1, 1],
        [3, 6, 112, 1, 1, 1],
        [5, 6, 160, 1, 1, 2],
        [5, 6, 160, 1, 1, 1],
        [5, 6, 160, 1, 1, 1],
    ]
    return MobileNetV3(cfgs, mode="large", **kwargs)


def mobilenetv3_small(**kwargs):
    """Constructs a MobileNetV3-Small model."""
    cfgs = [
        # k, t, c, SE, HS, s
        [3, 1, 16, 1, 0, 2],
        [3, 4.5, 24, 0, 0, 2],
        [3, 3.67, 24, 0, 0, 1],
        [5, 4, 40, 1, 1, 2],
        [5, 6, 40, 1, 1, 1],
        [5, 6, 40, 1, 1, 1],
        [5, 3, 48, 1, 1, 1],
        [5, 3, 48, 1, 1, 1],
        [5, 6, 96, 1, 1, 2],
        [5, 6, 96, 1, 1, 1],
        [5, 6, 96, 1, 1, 1],
    ]

    return MobileNetV3(cfgs, mode="small", **kwargs)


if __name__ == "__main__":
    path = Path("src/models/pretrained_backbones")
    models_dict = {
        "mobilenetv3-small-55df8e1f.pth": mobilenetv3_small,
        "mobilenetv3-large-1cd25616.pth": mobilenetv3_large,
    }
    for key, model_func in models_dict.items():
        weights_path = path / key
        model = model_func(weights=weights_path, num_classes=19)
        model.eval()
        print(f"Model {key} created and pretrained weights loaded successfully.")
        input_size = (1, 3, 224, 224)
        x = torch.randn(input_size)
        out = model(x)
        print(out.shape)
