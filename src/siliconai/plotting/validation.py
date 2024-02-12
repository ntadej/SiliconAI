"""Validation plotting helpers."""
import math
from pathlib import Path

import lightning as L
import matplotlib
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid  # type: ignore

from siliconai.cli.config import Configuration
from siliconai.cli.logging import Logger
from siliconai.common.enums import DataType, ModelType
from siliconai.ml.training.loaders import load_model_from_latest_checkpoint


def quick_validate(
    logger: Logger,
    config: Configuration,
    model: L.LightningModule,
) -> None:
    """Validate the model after training."""
    if config.data.type is DataType.TRKNtuple:
        logger.info("TRK ntuple validation is not implemented yet.")
        return

    if config.data.type is DataType.MNIST:
        logger.info("Validating MNIST-based model output...")
        file = quick_validate_mnist(config, model)
        logger.info("Validation done and stored in %s.", file)


def quick_validate_mnist(config: Configuration, model: L.LightningModule) -> Path:
    """Validate MNIST-based model output."""
    batch_size = 100
    grid_size = int(math.sqrt(batch_size))

    output_file = (
        config.output_path
        / f"run_{config.run_number(training=False)}"
        / "validation.pdf"
    )

    if config.model.type in [ModelType.ConditioningVAE]:
        x = model.generate(
            batch_size,
            torch.tensor([list(range(10))] * grid_size).clone().view(-1),
        )
    else:
        x = model.generate(batch_size)

    grid = make_grid(x.view(batch_size, 1, *config.data.input_dim), nrow=grid_size)
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap=matplotlib.cm.gray)  # type: ignore
    plt.savefig(output_file)

    return output_file


def validate(logger: Logger, config: Configuration) -> None:
    """Validate the model after training."""
    checkpoint_path = (
        config.global_config.output_path
        / config.output_name
        / f"run_{config.run_number(training=False)}"
        / "checkpoints"
    )
    model = load_model_from_latest_checkpoint(logger, config, checkpoint_path)
    quick_validate(logger, config, model)
