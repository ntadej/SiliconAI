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

    def step(self, epoch: int | None = None) -> None:
        """Step could be called after every batch update."""
        if epoch is None:
            epoch = 0 if self.last_epoch < 0 else self.last_epoch + 1

        self.last_epoch = math.floor(epoch)

        for group, lr in zip(
            self.optimizer.param_groups,
            self.get_lr(),
            strict=False,
        ):
            group["lr"] = lr

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
