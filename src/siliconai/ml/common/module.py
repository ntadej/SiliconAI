"""Common lightning modules for all models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import lightning as L
from torch import optim

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import OptimizerLRScheduler
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

    from siliconai.cli.config import Configuration


class Module(L.LightningModule):
    """Common lightning module."""

    def __init__(self, config: Configuration) -> None:
        """Initialize the module."""
        super().__init__()

        self.config = config

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure optimizers."""
        optimizer = getattr(optim, self.config.training.optimizer)
        optimizer_instance: Optimizer = optimizer(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        config: OptimizerLRScheduler
        if self.config.training.scheduler:
            scheduler = getattr(optim.lr_scheduler, self.config.training.scheduler)
            scheduler_instance: LRScheduler = scheduler(
                optimizer_instance,
                **self.config.training.scheduler_kwargs,
            )
            config = {
                "optimizer": optimizer_instance,
                "lr_scheduler": {
                    "scheduler": scheduler_instance,
                    "monitor": "val_loss",
                    "interval": self.config.training.scheduler_interval,
                },
            }
            return config

        config = {"optimizer": optimizer_instance}
        return config

    def on_train_start(self) -> None:
        """Call when the training begins."""
        if not self.logger:
            return

        self.logger.experiment.log_text(self.logger.run_id, str(self), "model.txt")  # type: ignore
