"""Unit tests for src/utils/ema.py (ModelEMA)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.ema import ModelEMA, _unwrap


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)
        # A non-float buffer, mirroring BatchNorm's num_batches_tracked —
        # must never be touched by the EMA update.
        self.register_buffer("counter", torch.tensor(0, dtype=torch.long))


class TestModelEMAConstruction:
    def test_initial_ema_equals_source_weights(self):
        model = TinyModel()
        ema = ModelEMA(model)
        for p1, p2 in zip(model.parameters(), ema.ema.parameters()):
            assert torch.allclose(p1, p2)

    def test_ema_params_require_no_grad(self):
        model = TinyModel()
        ema = ModelEMA(model)
        assert all(not p.requires_grad for p in ema.ema.parameters())

    def test_ema_is_a_distinct_object_from_source(self):
        """Mutating the source model after construction must not affect the
        EMA shadow copy (it must be a deepcopy, not a reference)."""
        model = TinyModel()
        ema = ModelEMA(model)
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(999.0)
        for p in ema.ema.parameters():
            assert not torch.allclose(p, torch.full_like(p, 999.0))

    def test_device_matches_source_model(self):
        model = TinyModel()
        ema = ModelEMA(model)
        assert next(ema.ema.parameters()).device == next(model.parameters()).device


class TestModelEMADecayRamp:
    def test_decay_near_zero_at_first_update(self):
        model = TinyModel()
        ema = ModelEMA(model, decay=0.9999, tau=2000)
        ema.updates = 1
        assert ema._current_decay() < 0.01

    def test_decay_approaches_target_for_large_updates(self):
        model = TinyModel()
        ema = ModelEMA(model, decay=0.9999, tau=2000)
        ema.updates = 10 * 2000  # 10x the ramp time constant
        assert ema._current_decay() > 0.999 * ema.decay

    def test_decay_is_monotonically_increasing(self):
        model = TinyModel()
        ema = ModelEMA(model, decay=0.9999, tau=2000)
        prev = 0.0
        for updates in (1, 10, 100, 1000, 10000):
            ema.updates = updates
            d = ema._current_decay()
            assert d > prev
            prev = d


class TestModelEMAUpdate:
    def test_update_moves_ema_toward_live_weights(self):
        model = TinyModel()
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
        ema = ModelEMA(model, decay=0.9, tau=1)  # near-max decay after 1 update

        with torch.no_grad():
            for p in model.parameters():
                p.fill_(10.0)
        ema.update(model)

        for p in ema.ema.parameters():
            assert (p > 0).all(), "EMA weights should have moved off zero"
            assert (p < 10.0).all(), "EMA weights should not fully snap to source"

    def test_updates_counter_increments(self):
        model = TinyModel()
        ema = ModelEMA(model)
        assert ema.updates == 0
        ema.update(model)
        ema.update(model)
        assert ema.updates == 2

    def test_non_float_buffer_untouched(self):
        model = TinyModel()
        ema = ModelEMA(model)
        model.counter.fill_(5)
        ema.update(model)
        assert ema.ema.counter.item() == 0, (
            "Non-float buffers must be left untouched by the EMA update"
        )


class TestModelEMAStateDict:
    def test_round_trip_preserves_weights_and_updates(self):
        model = TinyModel()
        ema = ModelEMA(model, decay=0.9, tau=1)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))
        ema.update(model)
        ema.update(model)

        state = ema.state_dict()

        model2 = TinyModel()
        ema2 = ModelEMA(model2)
        ema2.load_state_dict(state)

        assert ema2.updates == ema.updates
        for p1, p2 in zip(ema.ema.parameters(), ema2.ema.parameters()):
            assert torch.allclose(p1, p2)


class TestUnwrap:
    def test_unwrap_plain_module_returns_itself(self):
        model = TinyModel()
        assert _unwrap(model) is model

    def test_unwrap_ddp_like_wrapper_returns_module_attr(self):
        class FakeDDP(nn.Module):
            def __init__(self, module: nn.Module) -> None:
                super().__init__()
                self.module = module

        inner = TinyModel()
        wrapped = FakeDDP(inner)
        assert _unwrap(wrapped) is inner
