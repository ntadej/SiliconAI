"""Learning rate schedulers."""

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class AttentionWarmupScheduler(LRScheduler):
    """Implementation of Eq. 3 from https://arxiv.org/abs/1706.03762.

    This corresponds to increasing the learning
    rate linearly for the first warmup_steps training steps, and decreasing
    it thereafter proportionally to the inverse square root of the step number.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        freeze_step: int | None = None,
        last_epoch: int = -1,
    ) -> None:
        """Initialize the scheduler."""
        self.warmup_steps = warmup_steps
        # TODO: Implement freeze_step
        self.freeze_step = freeze_step

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Compute the initial learning rate."""
        return [
            base_lr
            * min(
                (self.last_epoch + 1) ** (-0.5),
                (self.last_epoch + 1) * self.warmup_steps ** (-1.5),
            )
            for base_lr in self.base_lrs
        ]


class NanoGPTScheduler(LRScheduler):
    """Implementation of learning rate scheduler from nanoGPT."""

    def __init__(
        self,
        optimizer: Optimizer,
        min_lr: float,
        warmup_steps: int,
        freeze_step: int,
        last_epoch: int = -1,
    ) -> None:
        """Initialize the scheduler."""
        self.warmup_steps = warmup_steps
        self.freeze_step = freeze_step
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch)

    def get_lr_from_base_lr(self, base_lr: float, step: int) -> float:
        """Compute the learning rate from the base learning rate."""
        # 1) linear warmup for warmup_steps steps
        if step < self.warmup_steps:
            return base_lr * (step + 1) / (self.warmup_steps + 1)
        # 2) if it > lr_decay_iters, return min learning rate
        if step > self.freeze_step:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (step - self.warmup_steps) / (
            self.freeze_step - self.warmup_steps
        )
        if not (0 <= decay_ratio <= 1):
            error = "Invalid decay ratio"
            raise ValueError(error)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (base_lr - self.min_lr)

    def get_lr(self) -> list[float]:
        """Compute the initial learning rate."""
        return [
            self.get_lr_from_base_lr(base_lr, self.last_epoch)
            for base_lr in self.base_lrs
        ]
