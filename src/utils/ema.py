#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""Exponential Moving Average (EMA) of model weights.

Hand-rolled rather than imported from ``ultralytics`` (which implements the
same decay-ramp formula in ``ultralytics.utils.torch_utils.ModelEMA``) because
``ultralytics`` is an optional ``[yolo]`` extra — core CABiNet training must
not depend on it.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Dict

import torch
import torch.nn as nn


def _unwrap(model: nn.Module) -> nn.Module:
    """Return the underlying module, unwrapping DDP's `.module` if present."""
    inner = getattr(model, "module", model)
    return inner if isinstance(inner, nn.Module) else model


class ModelEMA:
    """Maintains a shadow copy of a model whose weights are an exponential
    moving average of the live model's weights.

    Update rule: ``ema_param = decay * ema_param + (1 - decay) * param``.
    ``decay`` itself ramps up from ~0 towards the target value over the first
    few thousand updates (``decay * (1 - exp(-updates / tau))``) so the
    average isn't dominated by noisy early-training weights.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        tau: int = 2000,
        updates: int = 0,
    ) -> None:
        self.ema = deepcopy(_unwrap(model)).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.tau = tau
        self.updates = updates

    def _current_decay(self) -> float:
        return self.decay * (1 - math.exp(-self.updates / self.tau))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Fold one more set of live weights into the running average."""
        self.updates += 1
        d = self._current_decay()
        msd = _unwrap(model).state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            # Non-float buffers (e.g. BatchNorm's num_batches_tracked) are
            # intentionally left untouched — there's no meaningful "average"
            # of an integer counter; this mirrors ultralytics' ModelEMA.

    def state_dict(self) -> Dict[str, Any]:
        return {"ema_state": self.ema.state_dict(), "updates": self.updates}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.ema.load_state_dict(state["ema_state"])
        self.updates = state["updates"]
