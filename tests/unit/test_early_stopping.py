"""Unit tests for src/utils/early_stopping.py (EarlyStopping)."""

from __future__ import annotations

import math

from src.utils.early_stopping import EarlyStopping


class TestEarlyStoppingDisabled:
    def test_patience_zero_never_stops(self):
        stopper = EarlyStopping(patience=0)
        assert stopper.patience == math.inf
        for epoch in range(0, 200, 10):
            assert stopper(epoch, fitness=0.0) is False


class TestEarlyStoppingTriggers:
    def test_stops_exactly_at_patience_epochs_without_improvement(self):
        stopper = EarlyStopping(patience=5)
        assert stopper(0, 0.5) is False  # first fitness always "improves" from 0
        for epoch in range(1, 5):
            assert stopper(epoch, 0.5) is False  # stagnant, but under patience
        assert stopper(5, 0.5) is True  # delta == patience

    def test_improvement_resets_the_clock(self):
        stopper = EarlyStopping(patience=3)
        assert stopper(0, 0.5) is False
        assert stopper(1, 0.6) is False  # improves -> best_epoch resets to 1
        assert stopper(2, 0.6) is False  # delta = 1
        assert stopper(3, 0.6) is False  # delta = 2
        assert stopper(4, 0.6) is True  # delta = 3 >= patience

    def test_worse_fitness_does_not_reset_the_clock(self):
        stopper = EarlyStopping(patience=2)
        assert stopper(0, 0.8) is False
        assert stopper(1, 0.1) is False  # worse than best, ignored
        assert stopper(2, 0.1) is True  # delta from epoch 0 == patience


class TestEarlyStoppingFitnessNone:
    def test_fitness_none_is_a_true_noop(self):
        stopper = EarlyStopping(patience=2)
        stopper(0, 0.5)
        assert stopper(1, None) is False
        assert stopper.best_epoch == 0
        assert stopper.best_fitness == 0.5

    def test_fitness_none_does_not_trigger_a_stop(self):
        """A skipped-eval epoch must never itself be the epoch that trips
        early stopping, regardless of how much patience has elapsed."""
        stopper = EarlyStopping(patience=1)
        stopper(0, 0.5)
        assert stopper(100, None) is False


class TestEarlyStoppingStateDict:
    def test_round_trip_preserves_best_fitness_and_epoch(self):
        stopper = EarlyStopping(patience=5)
        stopper(0, 0.3)
        stopper(2, 0.7)

        state = stopper.state_dict()

        stopper2 = EarlyStopping(patience=5)
        stopper2.load_state_dict(state)

        assert stopper2.best_fitness == stopper.best_fitness
        assert stopper2.best_epoch == stopper.best_epoch
