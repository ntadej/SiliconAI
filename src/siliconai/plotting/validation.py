"""Validation plotting helpers."""

import math
from pathlib import Path
from typing import cast

import lightning as L
import matplotlib
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid  # type: ignore

from siliconai.cli.config import Configuration
from siliconai.cli.logging import Logger
from siliconai.common.enums import DataType, ModelType
from siliconai.data.modules import TRKNtupleDataModule
from siliconai.ml.training.loaders import (
    load_data_module,
    load_model_from_latest_checkpoint,
)
from siliconai.plotting.common import plot_hist, setup_style
from siliconai.plotting.utils import PDFDocument


def quick_validate(
    logger: Logger,
    config: Configuration,
    model: L.LightningModule,
) -> None:
    """Validate the model after training."""
    if config.data.type is DataType.TRKNtuple:
        logger.info("Validating TRKNtuple-based model output...")
        file = quick_validate_trkntuple(config, model)
        logger.info("Validation done and stored in %s.", file)

    if config.data.type in [DataType.MNIST, DataType.FashionMNIST]:
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

    if config.model.type in [ModelType.BasicVAE, ModelType.ConvVAE]:
        x = model.generate(
            batch_size,
            torch.tensor([list(range(10))] * grid_size).clone().view(-1),
        )
    else:
        x = model.generate(batch_size)

    if config.model.loss != "logcosh_loss":
        x = torch.sigmoid(x)

    image_size = 3
    if (
        isinstance(config.data.input_dim, list)
        and len(config.data.input_dim) == image_size
    ):
        grid = make_grid(x.view(batch_size, *config.data.input_dim), nrow=grid_size)
    else:
        grid = make_grid(x.view(batch_size, 1, *config.data.input_dim), nrow=grid_size)
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap=matplotlib.cm.gray)  # type: ignore
    plt.savefig(output_file)

    return output_file


def quick_validate_trkntuple(config: Configuration, model: L.LightningModule) -> Path:
    """Validate TRKNtuple-based model output."""
    setup_style()

    batch_size = 1000
    output_file = (
        config.output_path
        / f"run_{config.run_number(training=False)}"
        / "validation.pdf"
    )

    data = cast(TRKNtupleDataModule, load_data_module(None, config))
    data.prepare_data()
    data.setup("test")

    val_data = data.test_data[:batch_size]
    orig = val_data[0].cpu().numpy()
    gen = model.generate(batch_size, val_data[1]).cpu().numpy()

    with PDFDocument(output_file) as pdf:  # type: ignore
        for i, column in enumerate(data.columns):
            fig, ax = plot_hist(
                [gen[:, i], orig[:, i]],
                column,
                labels=["Generated", "Original"],
            )
            if not fig:
                continue
            pdf.save(fig)

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
