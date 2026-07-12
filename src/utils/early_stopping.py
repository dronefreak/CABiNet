#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""Patience-based early stopping, monitoring a scalar fitness metric.

Mirrors ``ultralytics.utils.torch_utils.EarlyStopping``'s simple algorithm
(track the epoch fitness last improved; stop once ``epoch - best_epoch``
exceeds ``patience``), reimplemented here so CABiNet's own trainer doesn't
depend on the optional ``ultralytics`` package.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional


class EarlyStopping:
    """Stop training when fitness (here: mIoU) hasn't improved for ``patience``
    epochs. ``patience=0`` (or any falsy value) disables early stopping."""

    def __init__(self, patience: int = 0) -> None:
        self.best_fitness = 0.0
        self.best_epoch = 0
        self.patience = patience or math.inf

    def __call__(self, epoch: int, fitness: Optional[float]) -> bool:
        """Update internal state with this epoch's fitness and return True if
        training should stop. ``fitness=None`` (e.g. an epoch with no mIoU
        eval) is a no-op — it neither counts as an improvement nor advances
        the patience clock's reference point."""
        if fitness is None:
            return False

        if fitness > self.best_fitness or self.best_fitness == 0:
            self.best_epoch = epoch
            self.best_fitness = fitness

        return (epoch - self.best_epoch) >= self.patience

    def state_dict(self) -> Dict[str, Any]:
        return {"best_fitness": self.best_fitness, "best_epoch": self.best_epoch}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.best_fitness = state["best_fitness"]
        self.best_epoch = state["best_epoch"]
