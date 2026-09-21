import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler

"""
scheduler:
    target: ctflow.common.schedulers.CosineAnnealingWithWarmup
    args:
        warmup_steps: 500
        total_steps: 500_000
        eta_min: 1e-5
"""


class CosineAnnealingWithWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=0, last_epoch=-1):
        """
        Custom Learning Rate Scheduler that performs a linear warm-up followed by
        cosine annealing.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_steps (int): Number of steps for the warm-up phase.
            total_steps (int): Total number of training steps.
            eta_min (float, optional): Minimum learning rate. Default: 0.
            last_epoch (int, optional): The index of last epoch. Default: -1.
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super(CosineAnnealingWithWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self, step=None):
        """
        Compute learning rate using linear warm-up and then cosine annealing.

        Returns:
            list: List of learning rates for each parameter group.
        """
        current_step = step or self.last_epoch
        if current_step < self.warmup_steps:
            # Linear warm-up
            warmup_factor = float(current_step) / float(max(1, self.warmup_steps))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        elif current_step <= self.total_steps:
            # Cosine annealing
            progress = float(current_step - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.eta_min + (base_lr - self.eta_min) * cosine_decay
                for base_lr in self.base_lrs
            ]
        else:
            # After total_steps, keep eta_min
            return [self.eta_min for _ in self.base_lrs]


"""
scheduler:
    target: ctflow.common.schedulers.ConstantLRWithWarmup
    args:
        warmup_steps: 500
"""


class ConstantLRWithWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        """
        Custom Learning Rate Scheduler that performs a linear warm-up followed by
        constant learning rate.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_steps (int): Number of steps for the warm-up phase.
            last_epoch (int, optional): The index of last epoch. Default: -1.
        """
        self.warmup_steps = warmup_steps
        super(ConstantLRWithWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self, step=None):
        """
        Compute learning rate using linear warm-up and then constant learning rate.

        Returns:
            list: List of learning rates for each parameter group.
        """
        current_step = step or self.last_epoch
        if current_step < self.warmup_steps:
            # Linear warm-up
            warmup_factor = float(current_step) / float(max(1, self.warmup_steps))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            return self.base_lrs


"""
scheduler:
  target: ctflow.common.schedulers.StepBasedLearningRateScheduleWithWarmup
  args:
    warmup_steps: 500
    ref_steps: 500_000
    eta_min: 1e-5
    decay_rate: 2.0
"""


class StepBasedLearningRateScheduleWithWarmup(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps,
        ref_steps=100_000,
        eta_min=0,
        decay_rate=1.0,  # New parameter
        last_epoch=-1,
    ):
        """
        Custom Learning Rate Scheduler that performs a linear warm-up followed by
        inverse square root decay with an adjustable decay rate.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_steps (int): Number of steps for the warm-up phase.
            ref_steps (int, optional): Reference steps for decay adjustment. Default: 70000.
            eta_min (float, optional): Minimum learning rate. Default: 0.
            decay_rate (float, optional): Decay rate multiplier. Default: 1.0.
            last_epoch (int, optional): The index of last epoch. Default: -1.
        """
        self.warmup_steps = warmup_steps
        self.ref_steps = ref_steps
        self.decay_rate = decay_rate
        self.eta_min = eta_min
        super(StepBasedLearningRateScheduleWithWarmup, self).__init__(
            optimizer, last_epoch
        )
        self.config = {
            "warmup_steps": warmup_steps,
            "ref_steps": ref_steps,
            "eta_min": eta_min,
            "decay_rate": decay_rate,
        }

    def get_lr(self, step=None):
        current_step = step or self.last_epoch
        if current_step < self.warmup_steps:
            # Linear warm-up
            warmup_factor = float(current_step) / float(max(1, self.warmup_steps))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Inverse square root decay with decay_rate
            decay_steps = current_step - self.warmup_steps
            lr = self.ref_lr / math.sqrt(
                1 + (decay_steps / self.ref_steps) * self.decay_rate
            )
            # Ensure learning rate doesn't go below eta_min
            return [max(lr, self.eta_min) for _ in self.base_lrs]

    @property
    def ref_lr(self):
        """
        Reference learning rate, assumed to be the initial learning rate of the first param group.
        Modify if multiple param groups have different initial LRs.
        """
        return self.base_lrs[0]
