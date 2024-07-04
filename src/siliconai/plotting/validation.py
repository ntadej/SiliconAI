"""Validation plotting helpers."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.utils import make_grid  # type: ignore

from siliconai.common.enums import DataLoadingType, DataType, ModelType
from siliconai.data.modules import (
    ActsDataModule,
    TestSequenceDataModule,
    TRKNtupleDataModule,
)
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
    data_type: DataLoadingType,
) -> None:
    """Validate the model after training."""
    if config.data.type is DataType.ActsHits:
        logger.info("Validating ActsHits-based model output...")
        file = quick_validate_acts_hits(config, model, data, data_type, logger=logger)
        logger.info("Validation done and stored in %s.", file)

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


def acts_process_data(
    config: Configuration,
    data_int: list[NDArrayType],
    data_float: list[NDArrayType],
) -> tuple[list[NDArrayType], list[NDArrayType], pd.DataFrame]:
    """Process ActsHits data."""
    # non-zero results
    data_nonzero_int = []
    data_nonzero_float = []
    for i in range(len(data_int)):
        nz = np.nonzero(data_int[i][:, 0])
        data_nonzero_int.append(data_int[i][nz[0].min() : nz[0].max() + 1])
        data_nonzero_float.append(data_float[i][nz[0].min() : nz[0].max() + 1])

    # data frame
    data_labels = [
        np.transpose(
            np.array(
                [
                    np.full(
                        shape=len(v),
                        fill_value=i,
                        dtype=int,
                    ),
                    np.arange(0, len(v)),
                ],
            ),
        )
        for i, v in enumerate(data_nonzero_int)
    ]

    data_annotated_int = np.concatenate(
        [np.hstack([a, b]) for a, b in zip(data_labels, data_nonzero_int, strict=True)],
    )
    data_annotated_float = np.concatenate(
        [
            np.hstack([a, b])
            for a, b in zip(data_labels, data_nonzero_float, strict=True)
        ],
    )

    data_df_int = pd.DataFrame(data_annotated_int)
    data_df_float = pd.DataFrame(data_annotated_float)

    columns_labels = {
        0: "event_id",
        1: "index",
    }
    for i, label in enumerate(config.data.columns_integer):
        columns_labels[i + 2] = label
    data_df_int = data_df_int.rename(columns=columns_labels)
    data_df_int = data_df_int.set_index(["event_id", "index"])

    columns_labels = {
        0: "event_id",
        1: "index",
    }
    for i, label in enumerate(config.data.columns_float):
        columns_labels[i + 2] = label
    data_df_float = data_df_float.rename(columns=columns_labels)
    data_df_float = data_df_float.set_index(["event_id", "index"])

    data_df = pd.concat([data_df_int, data_df_float], axis=1)

    if "lxq" in data_df.columns:
        data_df["lxq"] /= 100
    if "lyq" in data_df.columns:
        data_df["lyq"] /= 100

    return data_nonzero_int, data_nonzero_float, data_df


def quick_validate_acts_hits(  # noqa: PLR0912 PLR0915 C901
    config: Configuration,
    model: L.LightningModule,
    data: L.LightningDataModule,
    data_type: DataLoadingType,
    logger: Logger | None = None,
) -> Path:
    """Validate ActsHits-based model output."""
    # _rich_traceback_guard = True
    setup_style()

    output_file = (
        config.output_path
        / f"run_{config.run_number()}"
        / f"validation_{data_type.value}.pdf"
    )

    data = cast(ActsDataModule, data)
    data.setup(data_type.value)
    if logger:
        for tokenize in data.tokenize:
            tokenize.summary(logger)
        for normalize in data.normalize:
            normalize.summary(logger)

    input_full_int = []
    input_full_float = []
    result_full_int = []
    result_full_float = []
    if logger:
        logger.info("Starting inference")
    time_start = time.perf_counter()

    for batch in data.get_dataloader(data_type):
        batch_full_int, batch_full_float = batch[0], batch[1]
        batch_start_int, batch_start_float = (
            batch_full_int[:, :1],
            batch_full_float[:, :1],
        )

        result_int, result_float = model.predict(
            batch_start_int.to(model.device),
            batch_start_float.to(model.device),
            end_token=data.tokenize[0].dictionary.word2idx[10001],
        )

        input_full_int += list(batch_full_int.cpu().numpy())
        input_full_float += list(batch_full_float.cpu().numpy())
        result_full_int += list(result_int.cpu().numpy())
        result_full_float += list(result_float.cpu().numpy())

    input_translated_int = [data.translate_data(i) for i in input_full_int]
    input_translated_float = [data.inverse_data(i) for i in input_full_float]
    result_translated_int = [data.translate_data(i) for i in result_full_int]
    result_translated_float = [data.inverse_data(i) for i in result_full_float]

    time_end = time.perf_counter()

    if logger:
        logger.info(
            "Inference done in %.4f s (%.4f s per 10k particles)",
            time_end - time_start,
            (time_end - time_start) / len(input_full_int) * 10000,
        )

    # non-zero results and DF conversion
    input_nonzero_int, input_nonzero_float, input_df = acts_process_data(
        config,
        input_translated_int,
        input_translated_float,
    )
    result_nonzero_int, result_nonzero_float, result_df = acts_process_data(
        config,
        result_translated_int,
        result_translated_float,
    )

    assert len(input_nonzero_int) == len(result_nonzero_int)

    if logger:
        logger.info("Total events processed: %d", len(result_nonzero_int))

    # store data
    with pd.HDFStore(
        config.output_path
        / f"run_{config.run_number()}"
        / f"data_{data_type.value}.h5",
        mode="w",
    ) as store:
        store["reference_data"] = input_df
        store["generated_data"] = result_df

    # validation plots
    with PDFDocument(output_file) as pdf:  # type: ignore
        labels = ["Original", "Generated"]

        n_hits_input = [len(i) - 2 for i in input_nonzero_int]
        n_hits_result = [len(i) - 2 for i in result_nonzero_int]
        fig, ax = plot_hist(
            [n_hits_input, n_hits_result],
            "Number of hits",
            labels=labels,
        )
        if fig:
            pdf.save(fig)

        if "lx" in config.data.columns_float:
            lx_index = config.data.columns_float.index("lx")
            lx_input = list(
                np.concatenate([i[1:-2, lx_index] for i in input_nonzero_float]),
            )
            lx_result = list(
                np.concatenate([i[1:-2, lx_index] for i in result_nonzero_float]),
            )
            fig, ax = plot_hist(
                [lx_input, lx_result],
                "Local x position",
                labels=labels,
            )
            if fig:
                pdf.save(fig)

        if "ly" in config.data.columns_float:
            ly_index = config.data.columns_float.index("ly")
            ly_input = list(
                np.concatenate([i[1:-2, ly_index] for i in input_nonzero_float]),
            )
            ly_result = list(
                np.concatenate([i[1:-2, ly_index] for i in result_nonzero_float]),
            )
            fig, ax = plot_hist(
                [ly_input, ly_result],
                "Local y position",
                labels=labels,
            )
            if fig:
                pdf.save(fig)

        if "lxq" in config.data.columns_integer:
            lxq_index = config.data.columns_integer.index("lxq")
            lxq_input = list(
                np.concatenate([i[1:-2, lxq_index] for i in input_nonzero_int]),
            )
            lxq_result = list(
                np.concatenate([i[1:-2, lxq_index] for i in result_nonzero_int]),
            )
            fig, ax = plot_hist(
                [lxq_input, lxq_result],
                "Local x position",
                labels=labels,
            )
            if fig:
                pdf.save(fig)

        if "lyq" in config.data.columns_integer:
            lyq_index = config.data.columns_integer.index("lyq")
            lyq_input = list(
                np.concatenate([i[1:-2, lyq_index] for i in input_nonzero_int]),
            )
            lyq_result = list(
                np.concatenate([i[1:-2, lyq_index] for i in result_nonzero_int]),
            )
            fig, ax = plot_hist(
                [lyq_input, lyq_result],
                "Local y position",
                labels=labels,
            )
            if fig:
                pdf.save(fig)

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
        for i, feature in enumerate(data.features):
            fig, ax = plot_hist(
                [gen[:, i], orig[:, i]],
                feature,
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
    """Validate test sequence model output."""
    # _rich_traceback_guard = True
    setup_style()

    data = cast(TestSequenceDataModule, data)
    data.tokenize_data()  # TODO: should not be needed

    sequence = np.array([[1, 2], [5, 2], [8, 2], [5, 2]])

    sequence_tokenized_tuple: tuple[NDArrayType, NDArrayType | None] = (
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
    result_translated_tuple: tuple[NDArrayType, NDArrayType | None] = (
        np.copy(result_translated),
        np.copy(result_translated),
    )
    for tokenize in data.tokenize:
        result_translated_tuple = tokenize.inverse(result_translated_tuple)

    result_translated = result_translated_tuple[0]

    if logger:
        logger.info("Result: %s", result_translated)

    return Path()


def validate(
    logger: Logger,
    config: Configuration,
    data_type: DataLoadingType,
    checkpoint: int,
) -> None:
    """Validate the model after training."""
    checkpoint_path = (
        config.global_config.output_path
        / config.output_name
        / f"run_{config.run_number()}"
        / "checkpoints"
    )
    data = load_data_module_from_latest_checkpoint(
        logger,
        config,
        checkpoint_path,
        checkpoint,
    )
    model = load_model_from_latest_checkpoint(
        logger,
        config,
        checkpoint_path,
        checkpoint,
    )
    quick_validate(logger, config, model, data, data_type)
