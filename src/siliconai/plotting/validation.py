"""Validation plotting helpers."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import torch

from siliconai.common.enums import ColumnType, DataLoadingType, DataType, ModelType
from siliconai.ml.training.loaders import (
    load_data_module_from_latest_checkpoint,
    load_model_from_latest_checkpoint,
)
from siliconai.plotting.common import plot_hist, setup_style
from siliconai.plotting.utils import PDFDocument

if TYPE_CHECKING:
    from pathlib import Path

    import lightning as L

    from siliconai.cli.config import Configuration
    from siliconai.cli.logger import Logger
    from siliconai.data.modules import (
        ActsChainDataModule,
        ActsHitsDataModule,
    )
    from siliconai.data.utils import NDArrayType
    from siliconai.ml.common.module import NanoGPTModule, TransformerModule


def quick_validate(
    logger: Logger,
    config: Configuration,
    model: L.LightningModule,
    data: L.LightningDataModule,
    data_type: DataLoadingType,
    random: bool = False,
    no_random: bool = False,
) -> None:
    """Validate the model after training."""
    if config.data.type is DataType.ActsChain:
        logger.info("Validating ActsChain-based model output...")
        file = quick_validate_acts_chain(
            config,
            cast("NanoGPTModule", model),
            data,
            data_type,
            logger=logger,
        )
        logger.info("Validation done and stored in %s.", file)

    if config.data.type is DataType.ActsHits:
        logger.info("Validating ActsHits-based model output...")
        file = quick_validate_acts_hits(
            config,
            cast("TransformerModule", model),
            data,
            data_type,
            random,
            no_random,
            logger=logger,
        )
        logger.info("Validation done and stored in %s.", file)


def acts_process_data(  # noqa: PLR0912, C901
    config: Configuration,
    data: list[NDArrayType],
) -> tuple[list[NDArrayType], pd.DataFrame]:
    """Process ActsHits data."""
    # non-zero results
    data_nonzero = []

    delta_min_index = (
        config.data.columns.index("tpxq") if "tpxq" in config.data.columns else 0
    )
    delta_max_index = (
        config.data.columns.index("tpzq") if "tpzq" in config.data.columns else 0
    )
    delta_calculation = delta_min_index > 0 and delta_max_index > 0

    for i in range(len(data)):
        nz = np.nonzero(data[i][:, 0] != config.data.padding_token)
        end_token = np.nonzero(data[i][:, 0] == config.data.end_token)
        data_row = data[i][nz[0].min() : nz[0].max() + 1]
        if delta_calculation:
            data_diff = np.diff(
                data_row[:, delta_min_index : delta_max_index + 1],
                axis=0,
            )
            data_diff = np.vstack((np.zeros(3, dtype=np.int64), data_diff))
            data_row = np.hstack((data_row, data_diff))

        if end_token[0].size > 0:
            data_row = data_row[: end_token[0].min() + 1]
        data_row = data_row[1:-1]
        data_nonzero.append(data_row)

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
        for i, v in enumerate(data_nonzero)
    ]

    # restore numerical values
    column_numerical = []
    if config.data.split_numerical:
        i = 2  # indexing offset
        for column_type in config.data.columns_type:
            if column_type is ColumnType.Numerical:
                column_numerical.append((i, i + 1))
                i += 1
            i += 1

    data_annotated = np.concatenate(
        [np.hstack([a, b]) for a, b in zip(data_labels, data_nonzero, strict=True)],
    )

    data_df = pd.DataFrame(data_annotated)

    for i, j in column_numerical:
        data_df[i] = (
            data_df[j] + (np.sign(data_df[j]) + (data_df[j] == 0)) * data_df[i] / 100
        )

    for _, j in column_numerical:
        del data_df[j]

    columns_labels = {
        0: "event_id",
        1: "index",
    }
    if delta_calculation:
        delta_offset = len(config.data.columns) + 2
        columns_labels = columns_labels | {
            delta_offset: "deltapxq",
            delta_offset + 1: "deltapyq",
            delta_offset + 2: "deltapzq",
        }
    k = 2
    for i, label in enumerate(config.data.columns):
        columns_labels[k] = label
        k += 1
        if (
            config.data.split_numerical
            and config.data.columns_type
            and config.data.columns_type[i] == ColumnType.Numerical
        ):
            columns_labels[k] = label
            k += 1
    data_df = data_df.rename(columns=columns_labels)
    data_df = data_df.set_index(["event_id", "index"])

    if not config.data.split_numerical:
        columns_scale = [
            "lxq",
            "lyq",
            "tpxq",
            "tpyq",
            "tpzq",
            "deltapxq",
            "deltapyq",
            "deltapzq",
        ]
        for column in columns_scale:
            if column in data_df.columns:
                data_df[column] /= 100

    return data_nonzero, data_df


def quick_validate_acts_chain(  # noqa: PLR0912, PLR0915, C901
    config: Configuration,
    model: NanoGPTModule,
    data: L.LightningDataModule,
    data_type: DataLoadingType,
    logger: Logger | None = None,
) -> Path:
    """Validate ActsChain-based model output."""
    _rich_traceback_guard = True
    setup_style()

    # make sure we are in eval mode
    model.eval()

    output_file = (
        config.output_path
        / f"run_{config.run_number}"
        / f"validation_{data_type.value}.pdf"
    )

    data = cast("ActsChainDataModule", data)
    data.full_data = True
    data.setup(data_type.value)
    if logger:
        data.tokenizer.summary(logger)

    input_full: list[NDArrayType] = []
    result_full: list[NDArrayType] = []
    if logger:
        logger.info("Starting inference")
    time_start = time.perf_counter()

    ncolumns = len(config.data.columns)
    if config.data.index_with_offset >= 0:
        ncolumns += 1
    if config.data.split_numerical:
        ncolumns += len(
            [c for c in config.data.columns_type if c == ColumnType.Numerical],
        )

    for counter, batch in enumerate(data.get_dataloader(data_type)):
        batch_full = batch[0]
        batch_start = batch_full[:, :ncolumns].to(model.device)

        result = model.predict((batch_start,), tokenizer=data.tokenizer)

        if result is None:
            raise RuntimeError

        input_full += list(batch_full.cpu().numpy())
        result_full += list(result.cpu().numpy())

        if logger:
            logger.info("Processed %d batches", counter + 1)

    input_translated: list[NDArrayType] = [data.translate_data(i) for i in input_full]
    result_translated: list[NDArrayType] = [data.translate_data(i) for i in result_full]

    time_end = time.perf_counter()

    if logger:
        logger.info(
            "Inference done in %.4f s (%.4f s per 10k particles)",
            time_end - time_start,
            (time_end - time_start) / len(input_translated) * 10000,
        )

    # convert from flat to 2D
    input_translated = [
        np.pad(
            i,
            (
                0,
                ncolumns - len(i) + (len(i) // ncolumns) * ncolumns,
            ),
            constant_values=config.data.padding_token,
        ).reshape(-1, ncolumns)
        for i in input_translated
    ]
    result_translated = [
        np.pad(
            i,
            (
                0,
                ncolumns - len(i) + (len(i) // ncolumns) * ncolumns,
            ),
            constant_values=config.data.padding_token,
        ).reshape(-1, ncolumns)
        for i in result_translated
    ]

    # non-zero results and DF conversion
    input_nonzero, input_df = acts_process_data(config, input_translated)
    result_nonzero, result_df = acts_process_data(config, result_translated)

    if len(input_nonzero) != len(result_nonzero):
        error = "Input and result sizes do not match"
        raise ValueError(error)

    if logger:
        logger.info("Total events processed: %d", len(result_nonzero))

    # store data
    with pd.HDFStore(
        config.output_path / f"run_{config.run_number}" / f"data_{data_type.value}.h5",
        mode="w",
    ) as store:
        store["reference_data"] = input_df
        store["generated_data"] = result_df

    # validation plots
    with PDFDocument(output_file) as pdf:
        labels = ["Geant4", "Neural network"]

        n_hits_input = [len(i) for i in input_nonzero]
        n_hits_result = [len(i) for i in result_nonzero]

        fig, ax = plot_hist(
            [n_hits_input, n_hits_result],
            "Number of hits",
            labels=labels,
        )
        if fig:
            pdf.save(fig)

        fig, ax = plot_hist(
            [n_hits_result],
            "Number of hits",
            labels=labels[1:],
        )
        if fig:
            pdf.save(fig)

        columns_list = ["lxq", "lyq", "tpxq", "tpyq", "tpzq"]
        columns_labels = [
            "Local x position",
            "Local y position",
            "Momentum x",
            "Momentum y",
            "Momentum z",
        ]
        for column, column_label in zip(columns_list, columns_labels, strict=True):
            if column in config.data.columns:
                column_input = list(input_df[column].to_numpy())
                column_result = list(result_df[column].to_numpy())
                fig, ax = plot_hist(
                    [column_input, column_result],
                    column_label,
                    labels=labels,
                )
                if fig:
                    pdf.save(fig)

    return output_file


def quick_validate_acts_hits(  # noqa: PLR0912, PLR0915, C901
    config: Configuration,
    model: TransformerModule,
    data: L.LightningDataModule,
    data_type: DataLoadingType,
    random: bool = False,
    no_random: bool = False,
    logger: Logger | None = None,
) -> Path:
    """Validate ActsHits-based model output."""
    # _rich_traceback_guard = True
    setup_style()

    # make sure we are in eval mode
    model.eval()

    suffix = ""
    if random:
        suffix = "_random_test"
    if no_random:
        suffix = "_no_random"

    output_file = (
        config.output_path
        / f"run_{config.run_number}"
        / f"validation_{data_type.value}{suffix}.pdf"
    )

    data = cast("ActsHitsDataModule", data)
    data.setup(data_type.value)
    if logger and data.tokenizer:
        data.tokenizer.summary(logger)

    input_full: list[NDArrayType] = []
    result_full: list[NDArrayType] = []
    if logger:
        logger.info("Starting inference")
    time_start = time.perf_counter()

    for batch in data.get_dataloader(data_type):
        if config.model.type is ModelType.DiscreteTransformer:
            batch_full = batch[0]
            if random:
                batch_full = batch_full[0, :, :].repeat(
                    len(batch_full),
                    1,
                    1,
                )
            batch_start = batch_full[:, :1].to(model.device)

            if config.data.random_int and not no_random:
                batch_start[:, :, -1] = torch.randint(
                    1,
                    config.data.random_int,
                    (len(batch[0]), 1),
                )

            result = model.predict((batch_start,), tokenizer=data.tokenizer)

        input_full += list(
            batch_full.cpu().numpy() if batch_full is not None else [],
        )
        result_full += list(
            result.cpu().numpy() if result is not None else [],
        )

        if random:
            break

    input_translated: list[NDArrayType] = [data.translate_data(i) for i in input_full]
    result_translated: list[NDArrayType] = [data.translate_data(i) for i in result_full]

    time_end = time.perf_counter()

    if logger:
        logger.info(
            "Inference done in %.4f s (%.4f s per 10k particles)",
            time_end - time_start,
            (time_end - time_start) / len(input_full) * 10000,
        )

    # non-zero results and DF conversion
    input_nonzero, input_df = acts_process_data(config, input_translated)
    result_nonzero, result_df = acts_process_data(config, result_translated)

    if len(input_nonzero) != len(result_nonzero):
        error = "Input and result sizes do not match"
        raise ValueError(error)

    if logger:
        logger.info("Total events processed: %d", len(result_nonzero))

    # store data
    with pd.HDFStore(
        config.output_path
        / f"run_{config.run_number}"
        / f"data_{data_type.value}{suffix}.h5",
        mode="w",
    ) as store:
        store["reference_data"] = input_df
        store["generated_data"] = result_df

    # validation plots
    with PDFDocument(output_file) as pdf:
        labels = ["Geant4", "Neural network"]

        n_hits_input = [len(i) - 2 for i in input_nonzero]
        n_hits_result = [len(i) - 2 for i in result_nonzero]
        n_hits_diff = [
            abs(i - j) for i, j in zip(n_hits_input, n_hits_result, strict=True)
        ]

        fig, ax = plot_hist(
            [n_hits_input, n_hits_result],
            "Number of hits",
            labels=labels,
        )
        if fig:
            pdf.save(fig)

        fig, ax = plot_hist(
            [n_hits_result],
            "Number of hits",
            labels=labels[1:],
        )
        if fig:
            pdf.save(fig)

        fig, ax = plot_hist(
            [n_hits_diff],
            "Number of hits difference",
            labels=labels[1:],
        )
        if fig:
            pdf.save(fig)

        columns_list = ["lxq", "lyq", "tpxq", "tpyq", "tpzq"]
        columns_labels = [
            "Local x position",
            "Local y position",
            "Momentum x",
            "Momentum y",
            "Momentum z",
        ]

        for column, column_label in zip(columns_list, columns_labels, strict=True):
            if column in config.data.columns:
                column_index = config.data.columns.index(column)
                column_input = list(
                    np.concatenate([i[:, column_index] for i in input_nonzero]),
                )
                column_result = list(
                    np.concatenate([i[:, column_index] for i in result_nonzero]),
                )
                fig, ax = plot_hist(
                    [column_input, column_result],
                    column_label,
                    labels=labels,
                )
                if fig:
                    pdf.save(fig)

    return output_file


def validate(
    logger: Logger,
    config: Configuration,
    data_type: DataLoadingType,
    checkpoint: int,
    random: bool = False,
    no_random: bool = False,
) -> None:
    """Validate the model after training."""
    checkpoint_path = (
        config.global_config.output_path
        / config.output_name
        / f"run_{config.run_number}"
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
    quick_validate(logger, config, model, data, data_type, random, no_random)
