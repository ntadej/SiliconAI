"""Training utilities."""

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, RichProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import Logger, MLFlowLogger

from siliconai.cli.config import Configuration


def common_setup() -> None:
    """Prepare global training settings."""
    # matmul precision and seed
    torch.set_float32_matmul_precision("high")
    L.seed_everything(42, workers=True)


def setup_callbacks(config: Configuration) -> list[Callback]:
    """Prepare common training callbacks."""
    callbacks = [RichProgressBar(), LearningRateMonitor(logging_interval="step")]

    if config.training.early_stopping > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=config.training.early_stopping,
            ),
        )

    callbacks.append(
        ModelCheckpoint(
            dirpath=f"{config.output_path}/run_{config.run_number()}/checkpoints",
            save_weights_only=True,
            mode="min",
            monitor="val_loss",
            save_top_k=-1,
            every_n_epochs=25,
        ),
    )

    return callbacks


def setup_logging(config: Configuration) -> Logger:
    """Prepare common training logging."""
    return MLFlowLogger(
        experiment_name=config.name,
        run_name=f"Run #{config.run_number()}",
        save_dir=str(config.global_config.output_path / "mlruns"),
        log_model=True,
    )
