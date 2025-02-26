"""Common lightning modules for all models."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import lightning as L
import torch
from torch import Tensor, optim
from typing_extensions import Unpack

from siliconai.common.enums import ModelType
from siliconai.ml.models.transformer import (
    ChainTransformer,
    DiscreteTransformer,
    HybridTransformer,
    TransformerBase,
    TransformerPredictParams,
)
from siliconai.ml.training import schedulers

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import OptimizerLRScheduler
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

    from siliconai.cli.config import Configuration


class ModuleBase(L.LightningModule):
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
            if hasattr(schedulers, self.config.training.scheduler):
                scheduler = getattr(schedulers, self.config.training.scheduler)
            else:
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


class TransformerModule(ModuleBase):
    """Common lightning module."""

    def __init__(self, config: Configuration) -> None:
        """Initialize the module."""
        super().__init__(config)

        self.model: TransformerBase
        if config.model.type is ModelType.ChainTransformer:
            self.model = ChainTransformer(config)
        elif config.model.type is ModelType.DiscreteTransformer:
            self.model = DiscreteTransformer(config)
        elif config.model.type is ModelType.HybridTransformer:
            self.model = HybridTransformer(config)

        if config.training.compile:
            self.model = cast(
                "TransformerBase",
                torch.compile(self.model, fullgraph=True),
            )

        # save the hyperparameters
        self.save_hyperparameters()

    def forward(self, *args: Tensor) -> Tensor:
        """Forward pass."""
        return self.model.forward_pass(args, self.device, evaluate=True)

    def training_step(self, batch: Tensor, _batch_idx: int) -> Tensor:
        """Run training step."""
        loss, loss_int, loss_float = self.model.process_loss(batch, self.device)

        self.log("train_loss", loss, sync_dist=True)
        if loss_int:
            for label, loss_value in zip(
                self.config.data.columns_integer,
                loss_int,
                strict=True,
            ):
                self.log(f"train_loss_{label}", loss_value, sync_dist=True)
        if loss_float:
            self.log("train_loss_float", loss_float[0], sync_dist=True)
        return loss

    def validation_step(self, batch: Tensor, _batch_idx: int) -> Tensor:
        """Run validation step."""
        loss, loss_int, loss_float = self.model.process_loss(batch, self.device)

        self.log("val_loss", loss, sync_dist=True)
        if loss_int:
            for label, loss_value in zip(
                self.config.data.columns_integer,
                loss_int,
                strict=True,
            ):
                self.log(f"val_loss_{label}", loss_value, sync_dist=True)
        if loss_float:
            self.log("val_loss_float", loss_float[0], sync_dist=True)
        return loss

    def test_step(self, batch: Tensor, _batch_idx: int) -> Tensor:  # noqa: PT019
        """Run test step."""
        loss, loss_int, loss_float = self.model.process_loss(batch, self.device)

        self.log("test_loss", loss, sync_dist=True)
        if loss_int:
            for label, loss_value in zip(
                self.config.data.columns_integer,
                loss_int,
                strict=True,
            ):
                self.log(f"test_loss_{label}", loss_value, sync_dist=True)
        if loss_float:
            self.log("test_loss_float", loss_float[0], sync_dist=True)
        return loss

    @torch.no_grad()
    def predict(
        self,
        batch: tuple[Tensor, ...],
        **kwargs: Unpack[TransformerPredictParams],
    ) -> tuple[Tensor | None, Tensor | None]:
        """Run predictions on the model."""
        return self.model.predict(batch, self.device, **kwargs)
