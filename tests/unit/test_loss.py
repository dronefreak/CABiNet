"""Unit tests for loss functions."""

import pytest
import torch

from src.utils.loss import OhemCELoss, SoftmaxFocalLoss


class TestOhemCELoss:
    """Test OHEM Cross Entropy Loss."""

    def test_ohem_loss_forward(self):
        """Test OHEM loss forward pass."""
        loss_fn = OhemCELoss(thresh=0.7, n_min=100, ignore_lb=255)
        logits = torch.randn(4, 19, 64, 64)
        labels = torch.randint(0, 19, (4, 64, 64))

        loss = loss_fn(logits, labels)

        assert loss.ndim == 0  # Scalar loss
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_ohem_loss_with_ignore_label(self):
        """Test OHEM loss handles ignore label correctly."""
        loss_fn = OhemCELoss(thresh=0.7, n_min=100, ignore_lb=255)
        logits = torch.randn(2, 19, 32, 32)
        labels = torch.randint(0, 19, (2, 32, 32))

        # Set some pixels to ignore label
        labels[0, :10, :10] = 255

        loss = loss_fn(logits, labels)

        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_ohem_loss_empty_valid_pixels(self):
        """Test OHEM loss when all pixels are ignored."""
        loss_fn = OhemCELoss(thresh=0.7, n_min=100, ignore_lb=255)
        logits = torch.randn(1, 19, 32, 32)
        labels = torch.full((1, 32, 32), 255)  # All ignored

        loss = loss_fn(logits, labels)

        assert loss.item() == 0.0
        assert loss.requires_grad

    def test_ohem_loss_with_class_weights(self):
        """Test OHEM loss with class weights."""
        weights = torch.ones(19)
        weights[0] = 2.0  # Give class 0 higher weight

        loss_fn = OhemCELoss(thresh=0.7, n_min=100, ignore_lb=255, weight=weights)
        logits = torch.randn(2, 19, 32, 32)
        labels = torch.randint(0, 19, (2, 32, 32))

        loss = loss_fn(logits, labels)

        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_ohem_loss_backward(self):
        """Test OHEM loss backward pass."""
        loss_fn = OhemCELoss(thresh=0.7, n_min=100, ignore_lb=255)
        logits = torch.randn(2, 19, 32, 32, requires_grad=True)
        labels = torch.randint(0, 19, (2, 32, 32))

        loss = loss_fn(logits, labels)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_ohem_loss_no_criteria_submodule(self):
        """OhemCELoss must not store a separate nn.CrossEntropyLoss sub-module.

        Storing nn.CrossEntropyLoss(weight=w) as self.criteria creates a second
        weight reference that does NOT move when the parent is moved to a new device,
        causing a device mismatch error on CUDA forward passes.
        """
        weights = torch.ones(19)
        loss_fn = OhemCELoss(thresh=0.7, n_min=100, ignore_lb=255, weight=weights)
        # There must be no nested nn.Module child (criteria was the old pattern)
        assert not hasattr(loss_fn, "criteria"), (
            "OhemCELoss must not store an nn.CrossEntropyLoss sub-module; "
            "use F.cross_entropy with self.weight in forward() instead."
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_ohem_loss_cuda_no_device_mismatch(self):
        """Regression: OhemCELoss with class weights must not raise a device-mismatch
        error when the loss module is moved to CUDA after construction with CPU weights.
        """
        weights = torch.ones(19)  # CPU tensor
        weights[0] = 2.0
        loss_fn = OhemCELoss(thresh=0.7, n_min=100, ignore_lb=255, weight=weights)
        loss_fn = loss_fn.cuda()  # Must move weight buffer to CUDA

        logits = torch.randn(2, 19, 32, 32).cuda()
        labels = torch.randint(0, 19, (2, 32, 32)).cuda()

        # Previously raised: RuntimeError: Expected all tensors to be on the same device
        loss = loss_fn(logits, labels)
        assert not torch.isnan(loss)
        assert loss.item() >= 0


class TestSoftmaxFocalLoss:
    """Test Softmax Focal Loss."""

    def test_focal_loss_forward(self):
        """Test Focal loss forward pass."""
        loss_fn = SoftmaxFocalLoss(gamma=2.0, ignore_lb=255)
        logits = torch.randn(4, 19, 64, 64)
        labels = torch.randint(0, 19, (4, 64, 64))

        loss = loss_fn(logits, labels)

        assert loss.ndim == 0  # Scalar loss
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    @pytest.mark.parametrize("gamma", [0.0, 1.0, 2.0, 5.0])
    def test_focal_loss_gamma_values(self, gamma):
        """Test Focal loss with different gamma values."""
        loss_fn = SoftmaxFocalLoss(gamma=gamma, ignore_lb=255)
        logits = torch.randn(2, 19, 32, 32)
        labels = torch.randint(0, 19, (2, 32, 32))

        loss = loss_fn(logits, labels)

        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_focal_loss_with_weights(self):
        """Test Focal loss with class weights."""
        weights = torch.ones(19)
        loss_fn = SoftmaxFocalLoss(gamma=2.0, weight=weights, ignore_lb=255)
        logits = torch.randn(2, 19, 32, 32)
        labels = torch.randint(0, 19, (2, 32, 32))

        loss = loss_fn(logits, labels)

        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_focal_loss_backward(self):
        """Test Focal loss backward pass."""
        loss_fn = SoftmaxFocalLoss(gamma=2.0, ignore_lb=255)
        logits = torch.randn(2, 19, 32, 32, requires_grad=True)
        labels = torch.randint(0, 19, (2, 32, 32))

        loss = loss_fn(logits, labels)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_focal_loss_cuda_no_device_mismatch(self):
        """Test that SoftmaxFocalLoss with class weights works correctly after moving to CUDA.

        Regression test: previously nn.NLLLoss held a CPU weight reference causing a device
        mismatch error when the model was moved to GPU. Now we use F.nll_loss with the
        registered buffer (self.weight) which moves with .to(device).
        """
        weights = torch.ones(19)
        loss_fn = SoftmaxFocalLoss(gamma=2.0, weight=weights, ignore_lb=255).cuda()

        logits = torch.randn(2, 19, 32, 32).cuda()
        labels = torch.randint(0, 19, (2, 32, 32)).cuda()

        # Must not raise RuntimeError: expected scalar type Float but found ...
        loss = loss_fn(logits, labels)
        assert not torch.isnan(loss)
        assert loss.item() >= 0

    def test_focal_loss_no_weight_cuda_compatible(self):
        """Test SoftmaxFocalLoss without weights runs without device issues."""
        loss_fn = SoftmaxFocalLoss(gamma=2.0, ignore_lb=255)
        # weight buffer should be None — no issues expected on any device
        assert loss_fn.weight is None
        logits = torch.randn(2, 19, 16, 16)
        labels = torch.randint(0, 19, (2, 16, 16))
        loss = loss_fn(logits, labels)
        assert not torch.isnan(loss)
