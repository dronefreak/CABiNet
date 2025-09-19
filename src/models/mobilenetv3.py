"""Creates a MobileNetV3 Model as defined in:

Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun
Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam. (2019). Searching for MobileNetV3
arXiv preprint arXiv:1905.02244.
"""

import math
from pathlib import Path

import torch
import torch.nn as nn

__all__ = ["mobilenetv3_large", "mobilenetv3_small"]


def _make_divisible(v, divisor, min_value=None):
    """This function is taken from the original tf repo. It ensures that all layers have a channel
    number that is divisible by 8 It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py.

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


class HardSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HardSigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.sigmoid = HardSigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            HardSigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), HardSwish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), HardSwish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                HardSwish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                HardSwish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                HardSwish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode, width_mult=1.0, pretrained=False, weights=None):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.weights = weights
        self.pretrained = pretrained
        self.width_mult = width_mult
        assert mode in ["large", "small"]

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        self.input_layer = conv_3x3_bn(3, input_channel, 2)
        # print(input_channel)
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
        self.features = nn.Sequential(*layers)
        # print(type(self.features))
        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # output_channel = {'large': 1280, 'small': 1024}
        # output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[mode]
        # self.classifier = nn.Sequential(
        #     nn.Linear(exp_size, output_channel),
        #     HardSwish(),
        #     nn.Dropout(0.2),
        #     nn.Linear(output_channel, num_classes),
        # )
        if self.pretrained:
            if mode == "large":
                self._initialize_weights_large()
            else:
                self._initialize_weights_small()
        else:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = self.conv(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x

    def _initialize_weights_small(self):
        if self.width_mult == 0.75:
            state_dict = torch.load(str(self.weights))
        else:
            state_dict = torch.load(str(self.weights))
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            if "classifier" in k:
                continue
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def _initialize_weights_large(self):
        if self.width_mult == 0.75:
            state_dict = torch.load(str(self.weights))
        else:
            state_dict = torch.load(str(self.weights))
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            if "classifier" in k:
                continue
            self_state_dict.update({k: v})
            # print(k)
        self.load_state_dict(self_state_dict)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm2d)):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


def mobilenetv3_large(pretrained=False, width_mult=1.0, weights=None):
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
    return MobileNetV3(
        cfgs,
        mode="large",
        pretrained=pretrained,
        width_mult=width_mult,
        weights=weights,
    )


def spatial(pretrained=False, width_mult=1.0, weights=None):
    """Constructs a MobileNetV3-Small model."""
    cfgs = [
        # k, t, c, SE, HS, s
        [3, 1, 16, 1, 0, 2],
        [3, 4.5, 24, 0, 0, 2],
        [3, 3.67, 24, 0, 0, 1],
    ]

    return MobileNetV3(
        cfgs,
        mode="small",
        pretrained=pretrained,
        width_mult=width_mult,
        weights=weights,
    )


def mobilenetv3_small(pretrained=False, width_mult=1.0, weights=None):
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

    return MobileNetV3(
        cfgs,
        mode="small",
        pretrained=pretrained,
        width_mult=width_mult,
        weights=weights,
    )


if __name__ == "__main__":
    path = Path("core/models/pretrained_backbones")
    weights_path = path / "mobilenetv3-small-55df8e1f.pth"
    model = mobilenetv3_small(pretrained=True, width_mult=1.0, weights=weights_path)
    model.eval()
    input_size = (1, 3, 2048, 1024)
    # pip install --upgrade git+https://github.com/kuan-wang/pytorch-OpCounter.git
    # from thop import profile
    # flops, params = profile(net, input_size=input_size)
    # # print(flops)
    # # print(params)
    # print('Total params: %.2fM' % (params/1000000.0))
    # print('Total flops: %.2fM' % (flops/1000000.0))
    x = torch.randn(input_size)
    out = model(x)
    print(out.shape)
