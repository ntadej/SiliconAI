"""Validation plotting helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid  # type: ignore

from siliconai.common.enums import DataType, ModelType
from siliconai.data.modules import TestSequenceDataModule, TRKNtupleDataModule
from siliconai.ml.training.loaders import (
    load_data_module_from_latest_checkpoint,
    load_model_from_latest_checkpoint,
)
from siliconai.plotting.common import plot_hist, setup_style
from siliconai.plotting.utils import PDFDocument

if TYPE_CHECKING:
    import lightning as L

    from siliconai.cli.config import Configuration
    from siliconai.cli.logging import Logger
    from siliconai.data.utils import NDArrayType


def quick_validate(
    logger: Logger,
    config: Configuration,
    model: L.LightningModule,
    data: L.LightningDataModule,
) -> None:
    """Validate the model after training."""
    if config.data.type is DataType.TRKNtuple:
        logger.info("Validating TRKNtuple-based model output...")
        file = quick_validate_trkntuple(config, model, data)
        logger.info("Validation done and stored in %s.", file)

    if config.data.type in [DataType.MNIST, DataType.FashionMNIST]:
        logger.info("Validating MNIST-based model output...")
        file = quick_validate_mnist(config, model)
        logger.info("Validation done and stored in %s.", file)

    if config.data.type is DataType.TestSequence:
        logger.info("Validating TestSequence-based model output...")
        file = quick_validate_test_sequence(config, model, data, logger=logger)
        logger.info("Validation done and stored in %s.", file)


def quick_validate_mnist(config: Configuration, model: L.LightningModule) -> Path:
    """Validate MNIST-based model output."""
    batch_size = 100
    grid_size = int(math.sqrt(batch_size))

    output_file = config.output_path / f"run_{config.run_number()}" / "validation.pdf"

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


def quick_validate_trkntuple(
    config: Configuration,
    model: L.LightningModule,
    data: L.LightningDataModule,
) -> Path:
    """Validate TRKNtuple-based model output."""
    setup_style()

    batch_size = 1000
    output_file = config.output_path / f"run_{config.run_number()}" / "validation.pdf"

    data = cast(TRKNtupleDataModule, data)
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


def quick_validate_test_sequence(
    _: Configuration,
    model: L.LightningModule,
    data: L.LightningDataModule,
    logger: Logger | None = None,
) -> Path:
    """Validate TRKNtuple-based model output."""
    _rich_traceback_guard = True
    setup_style()

    model = model.cpu()

    data = cast(TestSequenceDataModule, data)
    data.tokenize_data()  # TODO: should not be needed

    sequence = np.array([[1, 2], [5, 2], [8, 2], [5, 2]])

    sequence_tokenized_tuple: tuple[NDArrayType, NDArrayType] = (
        np.copy(sequence),
        np.copy(sequence),
    )
    for tokenize in data.tokenize:
        sequence_tokenized_tuple = tokenize(sequence_tokenized_tuple)

    sequence_tokenized = sequence_tokenized_tuple[0]

    if logger:
        logger.info("Sequence: %s", sequence)
        logger.info("Tokenized: %s", sequence_tokenized)

    input_tensor = torch.tensor(
        np.array([sequence_tokenized]),
        dtype=torch.long,
        device=model.device,
    )

    result = model.predict(
        input_tensor,
        end_token=data.tokenize[0].dictionary.word2idx[2],
    )

    if logger:
        logger.info("Tokenized result: %s", result)

    result_translated = result.cpu().numpy()[0]
    result_translated_tuple: tuple[NDArrayType, NDArrayType] = (
        np.copy(result_translated),
        np.copy(result_translated),
    )
    for tokenize in data.tokenize:
        result_translated_tuple = tokenize.inverse(result_translated_tuple)

    result_translated = result_translated_tuple[0]

    if logger:
        logger.info("Result: %s", result_translated)

    return Path()


def validate(logger: Logger, config: Configuration) -> None:
    """Validate the model after training."""
    checkpoint_path = (
        config.global_config.output_path
        / config.output_name
        / f"run_{config.run_number()}"
        / "checkpoints"
    )
    data = load_data_module_from_latest_checkpoint(logger, config, checkpoint_path)
    model = load_model_from_latest_checkpoint(logger, config, checkpoint_path)
    quick_validate(logger, config, model, data)
