#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""Custom Optimizer with Warmup + Polynomial LR Schedule Supports differential learning
rates (e.g., decoder x10) and integrates safely with AMP."""

import logging
from typing import Any, Dict, List

import torch

logger = logging.getLogger(__name__)


class Optimizer:
    """
    Wrapper around SGD with:
      - Linear warmup
      - Polynomial decay
      - Support for lr-multiplied parameter groups (e.g., decoder)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        lr0: float,
        momentum: float = 0.9,
        wd: float = 1e-4,
        warmup_steps: int = 0,
        warmup_start_lr: float = 1e-5,
        max_iter: int = 100000,
        power: float = 0.9,
        lr_multiplier: float = 10.0,  # Instead of hardcoded x10
    ):
        self.lr0 = lr0
        self.momentum = momentum
        self.wd = wd
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.max_iter = float(max_iter)
        self.power = power
        self.lr_multiplier = lr_multiplier
        self.it = 0

        # Extract parameter groups from model
        try:
            params = model.get_params()
            if len(params) == 2:
                # Encoder-only mode? No decoder-specific groups
                wd_params, nowd_params = params
                lr_mul_wd_params, lr_mul_nowd_params = [], []
                logger.info(
                    "[Optimizer] Model returned 2 param groups (no decoder LR scaling)"
                )
            elif len(params) == 4:
                wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = params
                logger.info(
                    f"[Optimizer] Using differential LR: x{lr_multiplier} for decoder"
                )
            else:
                raise ValueError(f"Expected 2 or 4 param groups, got {len(params)}")
        except AttributeError as e:
            raise RuntimeError(
                "Model must have .get_params() method returning param groups"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Error parsing model parameters: {e}") from e

        # Build parameter list
        param_groups: List[Dict[str, Any]] = []

        if wd_params:
            param_groups.append({"params": wd_params, "weight_decay": wd})
        if nowd_params:
            param_groups.append({"params": nowd_params, "weight_decay": 0.0})
        if lr_mul_wd_params:
            param_groups.append(
                {
                    "params": lr_mul_wd_params,
                    "weight_decay": wd,
                    "lr_scale": lr_multiplier,
                }
            )
        if lr_mul_nowd_params:
            param_groups.append(
                {
                    "params": lr_mul_nowd_params,
                    "weight_decay": 0.0,
                    "lr_scale": lr_multiplier,
                }
            )

        if len(param_groups) == 0:
            raise ValueError("No parameters found in model!")

        # Create base optimizer
        self.optim = torch.optim.SGD(
            param_groups, lr=lr0, momentum=momentum, weight_decay=0.0
        )  # WD handled per-group

        # Warmup schedule
        if warmup_steps > 0:
            self.warmup_factor = (lr0 / warmup_start_lr) ** (1.0 / warmup_steps)
        else:
            self.warmup_factor = 1.0

        logger.info(
            f"[Optimizer] Initialized with LR={lr0}, WD={wd}, "
            f"Warmup={warmup_steps} steps, Poly(power={power}), "
            f"Max Iter={max_iter}"
        )

    def get_lr(self, group_idx: int, group: Dict[str, Any]) -> float:
        """Compute current LR for a given parameter group."""
        base_lr = self.lr0
        lr_scale = group.get("lr_scale", 1.0)

        if self.it < self.warmup_steps:
            # Linear or exponential warmup — use linear for stability
            alpha = self.it / self.warmup_steps
            lr = self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
        else:
            k = (self.it - self.warmup_steps) / (self.max_iter - self.warmup_steps)
            k = max(k, 0.0)  # Avoid negative
            lr = base_lr * ((1 - k) ** self.power)

        return lr * lr_scale

    def step(self):
        """Update learning rates and take optimizer step."""
        # Update LR for each group
        for i, pg in enumerate(self.optim.param_groups):
            pg["lr"] = self.get_lr(i, pg)

        # Take step
        self.optim.step()

        # Logging
        if self.it == self.warmup_steps:
            logger.info(
                f"==> Warmup completed at step {self.it}. "
                f"Switching to poly({self.power}) LR schedule."
            )

        self.it += 1

    def zero_grad(self):
        """Zero gradients."""
        self.optim.zero_grad()

    def state_dict(self):
        """Expose optimizer state."""
        return self.optim.state_dict()

    def load_state_dict(self, state):
        """Load optimizer state."""
        self.optim.load_state_dict(state)

    @property
    def param_groups(self):
        return self.optim.param_groups

    @property
    def defaults(self):
        return self.optim.defaults
