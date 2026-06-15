"""Unit tests for model architectures."""

import pytest
import torch

from src.models.cab import ContextAggregationBlock, PSPModule
from src.models.cabinet import AttentionBranch, ConvBNReLU, SpatialBranch
from src.models.constants import MODEL_CONFIG
from src.models.layers import DepthwiseConv, DepthwiseSeparableConv


class TestLayerComponents:
    """Test shared layer components."""

    def test_depthwise_conv_forward(self):
        """Test DepthwiseConv forward pass."""
        layer = DepthwiseConv(in_channels=32, out_channels=32, stride=1)
        x = torch.randn(2, 32, 64, 64)
        out = layer(x)

        assert out.shape == (2, 32, 64, 64)
        assert not torch.isnan(out).any()

    def test_depthwise_separable_conv_forward(self):
        """Test DepthwiseSeparableConv forward pass."""
        layer = DepthwiseSeparableConv(in_channels=32, out_channels=64, stride=2)
        x = torch.randn(2, 32, 64, 64)
        out = layer(x)

        assert out.shape == (2, 64, 32, 32)  # Stride 2 reduces spatial dims
        assert not torch.isnan(out).any()


class TestCABComponents:
    """Test Context Aggregation Block components."""

    def test_psp_module_forward(self):
        """Test PSPModule forward pass."""
        module = PSPModule(sizes=(1, 2, 4), in_channels=256)
        x = torch.randn(2, 256, 32, 32)
        out = module(x)

        assert out.shape == (2, 256, 32, 32)
        assert not torch.isnan(out).any()

    def test_context_aggregation_block_forward(self):
        """Test ContextAggregationBlock forward pass."""
        block = ContextAggregationBlock(512, 128)
        x = torch.randn(2, 512, 32, 64)
        out = block(x)

        assert out.shape == x.shape
        assert not torch.isnan(out).any()


class TestCABiNetComponents:
    """Test CABiNet model components."""

    def test_conv_bn_relu_forward(self):
        """Test ConvBNReLU block."""
        block = ConvBNReLU(in_chan=64, out_chan=128, kernel_size=3, stride=2)
        x = torch.randn(2, 64, 128, 128)
        out = block(x)

        assert out.shape == (2, 128, 64, 64)
        assert not torch.isnan(out).any()

    def test_attention_branch_forward(self):
        """Test AttentionBranch forward pass."""
        branch = AttentionBranch(
            inplanes=960, interplanes=256, outplanes=256, num_classes=19
        )
        x = torch.randn(2, 960, 32, 32)
        low_res, high_res = branch(x)

        assert low_res.shape == (2, 256, 32, 32)
        assert high_res.shape == (2, 19, 32, 32)
        assert not torch.isnan(low_res).any()
        assert not torch.isnan(high_res).any()

    def test_spatial_branch_forward(self):
        """Test SpatialBranch forward pass."""
        branch = SpatialBranch()
        x = torch.randn(2, 3, 512, 512)
        out = branch(x)

        assert out.shape == (2, 128, 64, 64)  # 1/8 resolution
        assert not torch.isnan(out).any()


class TestCABiNetModel:
    """Test full CABiNet model."""

    @pytest.mark.parametrize("mode", ["large", "small"])
    def test_cabinet_forward_shape(
        self, mode, num_classes, mock_small_model, mock_large_model
    ):
        """Test CABiNet forward pass output shapes."""
        model = (
            mock_small_model(num_classes=num_classes)
            if mode == "small"
            else mock_large_model(num_classes=num_classes)
        )
        model.eval()

        x = torch.randn(2, 3, 512, 512)
        with torch.no_grad():
            out, out16 = model(x)

        assert out.shape == (2, num_classes, 512, 512)
        assert out16.shape == (2, num_classes, 512, 512)
        assert not torch.isnan(out).any()
        assert not torch.isnan(out16).any()

    def test_cabinet_model_config(self, mock_small_model, mock_large_model):
        """Test MODEL_CONFIG is correctly used."""
        model_large = mock_large_model(num_classes=19, mode="large")
        model_small = mock_small_model(num_classes=19, mode="small")

        assert model_large.attention_planes == MODEL_CONFIG["large"]["attention_planes"]
        assert model_small.attention_planes == MODEL_CONFIG["small"]["attention_planes"]

    def test_cabinet_get_params(self, mock_small_model):
        """Test get_params returns all 4 parameter groups including decoder."""
        model = mock_small_model(num_classes=19, mode="small")
        result = model.get_params()

        assert (
            len(result) == 4
        ), "get_params() must return 4 groups: wd, nowd, lr_mul_wd, lr_mul_nowd"
        wd_params, nowd_params, lr_mul_wd, lr_mul_nowd = result

        # All groups must be non-empty
        assert len(wd_params) > 0, "wd_params (backbone/sb weights) must be non-empty"
        assert (
            len(nowd_params) > 0
        ), "nowd_params (backbone/sb biases/BN) must be non-empty"
        assert len(lr_mul_wd) > 0, "lr_mul_wd (decoder weights) must be non-empty"
        assert len(lr_mul_nowd) > 0, "lr_mul_nowd (decoder biases/BN) must be non-empty"

        # Decoder params must include the attention branch, FFM, and output conv
        all_decoder_params = set(id(p) for p in lr_mul_wd + lr_mul_nowd)
        for name, child in model.named_children():
            if name in ("ab", "ffm", "conv_out"):
                for p in child.parameters():
                    assert (
                        id(p) in all_decoder_params
                    ), f"Parameter from decoder module '{name}' missing from lr_mul groups"

    def test_cabinet_get_params_no_overlap(self, mock_small_model):
        """Test that no parameter appears in both backbone and decoder groups."""
        model = mock_small_model(num_classes=19, mode="small")
        wd_params, nowd_params, lr_mul_wd, lr_mul_nowd = model.get_params()

        backbone_ids = set(id(p) for p in wd_params + nowd_params)
        decoder_ids = set(id(p) for p in lr_mul_wd + lr_mul_nowd)
        overlap = backbone_ids & decoder_ids
        assert (
            len(overlap) == 0
        ), f"{len(overlap)} parameters appear in both backbone and decoder groups"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cabinet_cuda(self, num_classes, mock_large_model):
        """Test CABiNet on CUDA device."""
        model = mock_large_model(num_classes=num_classes, mode="large").cuda()
        x = torch.randn(1, 3, 512, 512).cuda()

        with torch.no_grad():
            out, out16 = model(x)

        assert out.is_cuda
        assert out16.is_cuda
        assert out.shape == (1, num_classes, 512, 512)
