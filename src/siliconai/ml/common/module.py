"""Common lightning modules for all models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import lightning as L
from torch import optim

if TYPE_CHECKING:
    from torch.optim import Optimizer

    from siliconai.cli.config import Configuration


class Module(L.LightningModule):
    """Common lightning module."""

    def __init__(self, config: Configuration) -> None:
        """Initialize the module."""
        super().__init__()

        self.config = config

    def configure_optimizers(self) -> Optimizer:
        """Configure optimizers."""
        optimizer = getattr(optim, self.config.training.optimizer)
        instance: Optimizer = optimizer(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        return instance

    def on_train_start(self) -> None:
        """Call when the training begins."""
        if not self.logger:
            return

        self.logger.experiment.log_text(self.logger.run_id, str(self), "model.txt")  # type: ignore
