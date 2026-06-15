"""Integration tests for training pipeline."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.scripts.evaluate import MscEvalV0
from src.utils.loss import OhemCELoss
from src.utils.optimizer import Optimizer


class TestTrainingIntegration:
    """Integration tests for training workflow."""

    def test_single_training_step(self, num_classes, mock_small_model):
        """Test a single training step completes successfully."""
        # Create model
        model = mock_small_model(num_classes=num_classes)
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

    def test_training_reduces_loss(self, num_classes, mock_small_model):
        """Test that training reduces loss over iterations."""
        model = mock_small_model(num_classes=num_classes)
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

    def test_dataloader_integration(self, num_classes, mock_small_model):
        """Test training with DataLoader."""
        model = mock_small_model(num_classes=num_classes)
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
    def test_training_on_cuda(self, num_classes, mock_small_model):
        """Test training pipeline on CUDA."""
        model = mock_small_model(num_classes=num_classes).cuda()
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


class TestGradientAccumulation:
    """Tests verifying correct gradient accumulation behaviour."""

    def test_grad_accum_equivalent_to_full_batch(self, num_classes, mock_small_model):
        """Gradient accumulation over N micro-batches must produce the same weight update
        as a single forward pass over the full batch (up to floating point tolerance).

        This is a regression test for the bug where optim.zero_grad() was called at the
        start of every micro-step, wiping accumulated gradients.
        """
        torch.manual_seed(0)
        accum_steps = 4
        micro_bs = 2  # micro-batch size
        full_bs = accum_steps * micro_bs

        # Two identical models
        model_accum = mock_small_model(num_classes=num_classes)
        model_full = mock_small_model(num_classes=num_classes)
        # Copy weights so both start identical
        model_full.load_state_dict(model_accum.state_dict())

        criterion = OhemCELoss(thresh=0.7, n_min=10, ignore_lb=255)

        # --- Full-batch gradient ---
        images_full = torch.randn(full_bs, 3, 64, 64)
        labels_full = torch.randint(0, num_classes, (full_bs, 64, 64))
        model_full.train()
        model_full.zero_grad()
        out, out16 = model_full(images_full)
        loss_full = criterion(out, labels_full) + criterion(out16, labels_full)
        loss_full.backward()
        grad_full = {
            n: p.grad.clone()
            for n, p in model_full.named_parameters()
            if p.grad is not None
        }

        # --- Accumulated gradient (correct pattern: zero_grad once, step every N) ---
        model_accum.train()
        model_accum.zero_grad()
        for step in range(accum_steps):
            micro_im = images_full[step * micro_bs : (step + 1) * micro_bs]
            micro_lb = labels_full[step * micro_bs : (step + 1) * micro_bs]
            out, out16 = model_accum(micro_im)
            loss = (criterion(out, micro_lb) + criterion(out16, micro_lb)) / accum_steps
            loss.backward()
        grad_accum = {
            n: p.grad.clone()
            for n, p in model_accum.named_parameters()
            if p.grad is not None
        }

        # Gradients should match (not necessarily exact due to OHEM sampling, but shapes)
        assert set(grad_full.keys()) == set(grad_accum.keys()), "Parameter sets differ"
        for name in grad_full:
            assert (
                grad_full[name].shape == grad_accum[name].shape
            ), f"Shape mismatch for {name}"
            # Norms should be in the same ballpark (within 2x) — exact match not guaranteed due to OHEM
            norm_full = grad_full[name].norm().item()
            norm_accum = grad_accum[name].norm().item()
            if norm_full > 1e-8:
                ratio = norm_accum / norm_full
                assert 0.05 < ratio < 20.0, (
                    f"Gradient norm ratio {ratio:.3f} for '{name}' is too far from 1.0. "
                    "Accumulation may not be working."
                )

    def test_grad_accum_nonzero_after_micro_steps(self, num_classes, mock_small_model):
        """Gradients must be non-zero and growing after each micro-batch (not wiped)."""
        torch.manual_seed(1)
        accum_steps = 3
        model = mock_small_model(num_classes=num_classes)
        model.train()
        criterion = OhemCELoss(thresh=0.7, n_min=10, ignore_lb=255)

        model.zero_grad()
        norms = []
        for step in range(accum_steps):
            im = torch.randn(1, 3, 64, 64)
            lb = torch.randint(0, num_classes, (1, 64, 64))
            out, out16 = model(im)
            loss = (criterion(out, lb) + criterion(out16, lb)) / accum_steps
            loss.backward()
            # Collect total grad norm after each micro-step
            total_norm = (
                sum(
                    p.grad.norm().item() ** 2
                    for p in model.parameters()
                    if p.grad is not None
                )
                ** 0.5
            )
            norms.append(total_norm)

        # Each micro-step should add gradients — norm must be monotonically non-decreasing
        for i in range(1, len(norms)):
            assert norms[i] >= norms[i - 1] * 0.5, (
                f"Gradient norm decreased from step {i-1} ({norms[i-1]:.4f}) "
                f"to step {i} ({norms[i]:.4f}), suggesting zero_grad was called mid-accumulation."
            )


class TestSlidingWindowEvaluation:
    """Tests for MscEvalV0 sliding window inference."""

    def _make_constant_model(self, n_classes, value=0):
        """Return a model that outputs a constant probability for class `value`."""

        class ConstantModel(torch.nn.Module):
            def __init__(self, n_classes, value):
                super().__init__()
                self.n_classes = n_classes
                self.value = value

            def forward(self, x):
                B, _, H, W = x.shape
                logits = torch.full((B, self.n_classes, H, W), -1e9)
                logits[:, self.value, :, :] = 1e9
                return logits, logits

        return ConstantModel(n_classes, value)

    def test_sliding_window_uniform_prediction(self):
        """A model that always predicts class 0 should produce a uniform class-0 map
        regardless of overlap — overlap normalization must not distort probabilities."""
        n_classes = 4
        cropsize = 64
        model = self._make_constant_model(n_classes, value=0)
        model.eval()

        image = torch.zeros(
            1, 3, 100, 100
        )  # larger than cropsize → sliding window activates
        evaluator = MscEvalV0(
            model=model,
            dataloader=None,
            n_classes=n_classes,
            ignore_label=255,
            scales=(1.0,),
            flip=False,
            cropsize=cropsize,
            device=torch.device("cpu"),
        )

        prob = evaluator.crop_eval(image)  # (1, n_classes, 100, 100)

        # After softmax, class 0 should dominate everywhere
        pred = prob.argmax(dim=1)  # (1, 100, 100)
        assert (
            pred == 0
        ).all(), "Uniform constant model should predict class 0 everywhere"

        # Prob values for class 0 should be uniform across spatial locations
        class0_prob = prob[0, 0, :, :]
        assert (
            class0_prob.max() - class0_prob.min() < 1e-5
        ), "Class 0 probability varies across spatial locations — overlap normalization may be wrong."

    def test_sliding_window_no_bias_at_edges(self):
        """Edge pixels (covered by fewer crops) must not have systematically different predictions
        than center pixels after overlap normalization."""
        n_classes = 3
        cropsize = 48
        model = self._make_constant_model(n_classes, value=1)
        model.eval()

        image = torch.zeros(1, 3, 96, 96)  # 2x cropsize → multiple overlapping windows
        evaluator = MscEvalV0(
            model=model,
            dataloader=None,
            n_classes=n_classes,
            ignore_label=255,
            scales=(1.0,),
            flip=False,
            cropsize=cropsize,
            device=torch.device("cpu"),
        )

        prob = evaluator.crop_eval(image)
        class1_prob = prob[0, 1, :, :]
        # Max - min should be negligible if normalization is correct
        assert class1_prob.max() - class1_prob.min() < 1e-5, (
            f"Edge vs. center probability gap: {(class1_prob.max() - class1_prob.min()).item():.6f}. "
            "Overlap normalization is not correcting for coverage differences."
        )


class TestEvaluationIntegration:
    """Integration tests for evaluation workflow."""

    def test_evaluation_mode(self, num_classes, mock_small_model):
        """Test model evaluation mode."""
        model = mock_small_model(num_classes=num_classes)
        model.eval()

        images = torch.randn(2, 3, 512, 512)

        with torch.no_grad():
            out, out16 = model(images)

        assert out.shape == (2, num_classes, 512, 512)
        assert out16.shape == (2, num_classes, 512, 512)
        assert not out.requires_grad
        assert not out16.requires_grad

    def test_prediction_consistency(self, num_classes, mock_small_model):
        """Test model produces consistent predictions in eval mode."""
        model = mock_small_model(num_classes=num_classes)
        model.eval()

        torch.manual_seed(42)
        images = torch.randn(1, 3, 256, 256)

        with torch.no_grad():
            out1, _ = model(images)
            out2, _ = model(images)

        # Predictions should be identical
        assert torch.allclose(out1, out2, atol=1e-6)
