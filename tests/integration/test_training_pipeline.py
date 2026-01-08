"""Integration tests for training pipeline."""

import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.models.cabinet import CABiNet
from src.utils.loss import OhemCELoss
from src.utils.optimizer import Optimizer


class TestTrainingIntegration:
    """Integration tests for training workflow."""

    def test_single_training_step(self, num_classes):
        """Test a single training step completes successfully."""
        # Create model
        model = CABiNet(n_classes=num_classes, mode="large")
        model.train()

        # Create dummy data
        images = torch.randn(2, 3, 256, 256)
        labels = torch.randint(0, num_classes, (2, 256, 256))

        # Create loss and optimizer
        criterion = OhemCELoss(thresh=0.7, n_min=100, ignore_lb=255)
        optimizer = Optimizer(
            model=model,
            lr0=1e-3,
            momentum=0.9,
            wd=5e-4,
            warmup_steps=0,
            max_iter=1000,
        )

        # Forward pass
        out, out16 = model(images)
        loss = criterion(out, labels) + criterion(out16, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify loss is valid
        assert not torch.isnan(loss)
        assert loss.item() > 0

    def test_training_reduces_loss(self, num_classes):
        """Test that training reduces loss over iterations."""
        model = CABiNet(n_classes=num_classes, mode="small")
        model.train()

        # Create fixed dataset
        torch.manual_seed(42)
        images = torch.randn(4, 3, 256, 256)
        labels = torch.randint(0, num_classes, (4, 256, 256))

        criterion = OhemCELoss(thresh=0.7, n_min=100, ignore_lb=255)
        optimizer = Optimizer(
            model=model,
            lr0=1e-2,
            momentum=0.9,
            wd=5e-4,
            warmup_steps=0,
            max_iter=100,
        )

        losses = []
        for _ in range(10):
            out, out16 = model(images)
            loss = criterion(out, labels) + criterion(out16, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Loss should decrease (allow some variance)
        assert losses[-1] < losses[0], "Loss should decrease during training"

    def test_dataloader_integration(self, num_classes):
        """Test training with DataLoader."""
        model = CABiNet(n_classes=num_classes, mode="large")
        model.train()

        # Create dummy dataset
        images = torch.randn(8, 3, 256, 256)
        labels = torch.randint(0, num_classes, (8, 256, 256))
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        criterion = OhemCELoss(thresh=0.7, n_min=100, ignore_lb=255)
        optimizer = Optimizer(
            model=model,
            lr0=1e-3,
            momentum=0.9,
            wd=5e-4,
            warmup_steps=0,
            max_iter=100,
        )

        # Train for one epoch
        for batch_images, batch_labels in dataloader:
            out, out16 = model(batch_images)
            loss = criterion(out, batch_labels) + criterion(out16, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            assert not torch.isnan(loss)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_training_on_cuda(self, num_classes):
        """Test training pipeline on CUDA."""
        model = CABiNet(n_classes=num_classes, mode="large").cuda()
        model.train()

        images = torch.randn(2, 3, 256, 256).cuda()
        labels = torch.randint(0, num_classes, (2, 256, 256)).cuda()

        criterion = OhemCELoss(thresh=0.7, n_min=100, ignore_lb=255)
        optimizer = Optimizer(
            model=model,
            lr0=1e-3,
            momentum=0.9,
            wd=5e-4,
            warmup_steps=0,
            max_iter=100,
        )

        out, out16 = model(images)
        loss = criterion(out, labels) + criterion(out16, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert out.is_cuda
        assert not torch.isnan(loss)


class TestEvaluationIntegration:
    """Integration tests for evaluation workflow."""

    def test_evaluation_mode(self, num_classes):
        """Test model evaluation mode."""
        model = CABiNet(n_classes=num_classes, mode="large")
        model.eval()

        images = torch.randn(2, 3, 512, 512)

        with torch.no_grad():
            out, out16 = model(images)

        assert out.shape == (2, num_classes, 512, 512)
        assert out16.shape == (2, num_classes, 512, 512)
        assert not out.requires_grad
        assert not out16.requires_grad

    def test_prediction_consistency(self, num_classes):
        """Test model produces consistent predictions in eval mode."""
        model = CABiNet(n_classes=num_classes, mode="large")
        model.eval()

        torch.manual_seed(42)
        images = torch.randn(1, 3, 256, 256)

        with torch.no_grad():
            out1, _ = model(images)
            out2, _ = model(images)

        # Predictions should be identical
        assert torch.allclose(out1, out2, atol=1e-6)
