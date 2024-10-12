"""Validation plotting helpers."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.utils import make_grid  # type: ignore

from siliconai.common.enums import ColumnType, DataLoadingType, DataType, ModelType
from siliconai.data.modules import (
    ActsChainDataModule,
    ActsHitsDataModule,
    TRKNtupleDataModule,
)
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
    from siliconai.cli.logging import Logger
    from siliconai.data.utils import NDArrayType


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
        file = quick_validate_acts_chain(config, model, data, data_type, logger=logger)
        logger.info("Validation done and stored in %s.", file)

    if config.data.type is DataType.ActsHits:
        logger.info("Validating ActsHits-based model output...")
        file = quick_validate_acts_hits(
            config,
            model,
            data,
            data_type,
            random,
            no_random,
            logger=logger,
        )
        logger.info("Validation done and stored in %s.", file)

    if config.data.type is DataType.TRKNtuple:
        logger.info("Validating TRKNtuple-based model output...")
        file = quick_validate_trkntuple(config, model, data)
        logger.info("Validation done and stored in %s.", file)

    if config.data.type in [DataType.MNIST, DataType.FashionMNIST]:
        logger.info("Validating MNIST-based model output...")
        file = quick_validate_mnist(config, model)
        logger.info("Validation done and stored in %s.", file)


def quick_validate_mnist(config: Configuration, model: L.LightningModule) -> Path:
    """Validate MNIST-based model output."""
    batch_size = 100
    grid_size = int(math.sqrt(batch_size))

    output_file = config.output_path / f"run_{config.run_number}" / "validation.pdf"

    x = model.generate(
        batch_size,
        torch.tensor([list(range(10))] * grid_size).clone().view(-1),
    )

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


def acts_process_data(  # noqa: C901 PLR0912 PLR0915
    config: Configuration,
    data_int: list[NDArrayType],
    data_float: list[NDArrayType],
) -> tuple[list[NDArrayType], list[NDArrayType], pd.DataFrame]:
    """Process ActsHits data."""
    # non-zero results
    data_nonzero_int = []
    data_nonzero_float = []

    delta_min_index = (
        config.data.columns_integer.index("tpxq")
        if "tpxq" in config.data.columns_integer
        else 0
    )
    delta_max_index = (
        config.data.columns_integer.index("tpzq")
        if "tpzq" in config.data.columns_integer
        else 0
    )
    delta_calculation = delta_min_index > 0 and delta_max_index > 0

    for i in range(max(len(data_int), len(data_float))):
        nz = np.nonzero(
            (data_int[i][:, 0] if data_int else data_float[i][:, 0])
            != config.data.padding_token,
        )
        if data_int:
            data = data_int[i][nz[0].min() : nz[0].max() + 1]
            if delta_calculation:
                data_diff = np.diff(
                    data[:, delta_min_index : delta_max_index + 1],
                    axis=0,
                )
                data_diff = np.vstack((np.zeros(3, dtype=np.int64), data_diff))
                data = np.hstack((data, data_diff))

            data = data[1:-1]
            data_nonzero_int.append(data)
        if data_float:
            data = data_float[i][nz[0].min() : nz[0].max() + 1]
            data = data[1:-1]
            data_nonzero_float.append(data)

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
        for i, v in enumerate(data_nonzero_int if data_int else data_nonzero_float)
    ]

    # restore numerical values
    column_numerical = []
    i = 2  # indexing offset
    for column_type in config.data.columns_type:
        if column_type is ColumnType.Numerical:
            column_numerical.append((i, i + 1))
            i += 1
        i += 1

    if data_nonzero_int:
        data_annotated_int = np.concatenate(
            [
                np.hstack([a, b])
                for a, b in zip(data_labels, data_nonzero_int, strict=True)
            ],
        )

        data_df_int = pd.DataFrame(data_annotated_int)

        for i, j in column_numerical:
            data_df_int[i] = data_df_int[j] + data_df_int[i] / 100
            del data_df_int[j]

        columns_labels = {
            0: "event_id",
            1: "index",
        }
        if delta_calculation:
            delta_offset = len(config.data.columns_integer) + 2
            columns_labels = columns_labels | {
                delta_offset: "deltapxq",
                delta_offset + 1: "deltapyq",
                delta_offset + 2: "deltapzq",
            }
        k = 2
        for i, label in enumerate(config.data.columns_integer):
            columns_labels[k] = label
            k += 1
            if (
                config.data.columns_type
                and config.data.columns_type[i] == ColumnType.Numerical
            ):
                columns_labels[k] = label
                k += 1
        data_df_int = data_df_int.rename(columns=columns_labels)
        data_df_int = data_df_int.set_index(["event_id", "index"])

        data_df = data_df_int

    if data_nonzero_float:
        data_annotated_float = np.concatenate(
            [
                np.hstack([a, b])
                for a, b in zip(data_labels, data_nonzero_float, strict=True)
            ],
        )

        data_df_float = pd.DataFrame(data_annotated_float)

        columns_labels = {
            0: "event_id",
            1: "index",
        }
        for i, label in enumerate(config.data.columns_float):
            columns_labels[i + 2] = label
        data_df_float = data_df_float.rename(columns=columns_labels)
        data_df_float = data_df_float.set_index(["event_id", "index"])

        data_df = data_df_float

    if data_nonzero_int and data_nonzero_float:
        data_df = pd.concat([data_df_int, data_df_float], axis=1)

    if not config.data.columns_type:
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

    return data_nonzero_int, data_nonzero_float, data_df


def quick_validate_acts_chain(  # noqa: PLR0915, C901
    config: Configuration,
    model: L.LightningModule,
    data: L.LightningDataModule,
    data_type: DataLoadingType,
    logger: Logger | None = None,
) -> Path:
    """Validate ActsChain-based model output."""
    # _rich_traceback_guard = True
    setup_style()

    # make sure we are in eval mode
    model.eval()

    output_file = (
        config.output_path
        / f"run_{config.run_number}"
        / f"validation_{data_type.value}.pdf"
    )

    data = cast(ActsChainDataModule, data)
    data.setup(data_type.value)
    if logger:
        data.tokenizer.summary(logger)

    input_full: list[NDArrayType] = []
    result_full: list[NDArrayType] = []
    if logger:
        logger.info("Starting inference")
    time_start = time.perf_counter()

    ncolumns = len(config.data.columns_integer) + len(
        [c for c in config.data.columns_type if c == ColumnType.Numerical],
    )

    for batch in data.get_dataloader(data_type):
        batch_full = batch[0]
        batch_start = batch_full[:, :ncolumns].to(model.device)

        result, _ = model.predict((batch_start,), tokenizer=data.tokenizer)

        input_full += list(batch_full.cpu().numpy())
        result_full += list(result.cpu().numpy())

    input_translated: list[NDArrayType] = [data.translate_data(i) for i in input_full]
    result_translated: list[NDArrayType] = [data.translate_data(i) for i in result_full]

    time_end = time.perf_counter()

    if logger:
        logger.info(
            "Inference done in %.4f s (%.4f s per 10k particles)",
            time_end - time_start,
            (time_end - time_start) / len(input_full) * 10000,
        )

    # convert from flat to 2D
    input_translated = [
        np.pad(
            i,
            (
                0,
                (len(i) // ncolumns + 1) * ncolumns - len(i),
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
                (len(i) // ncolumns + 1) * ncolumns - len(i),
            ),
            constant_values=config.data.padding_token,
        ).reshape(-1, ncolumns)
        for i in result_translated
    ]

    # non-zero results and DF conversion
    input_nonzero, _, input_df = acts_process_data(config, input_translated, [])
    result_nonzero, _, result_df = acts_process_data(config, result_translated, [])

    assert len(input_nonzero) == len(result_nonzero)

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
    with PDFDocument(output_file) as pdf:  # type: ignore
        labels = ["Geant4", "Neural network"]

        n_hits_input = [len(i) - 2 for i in input_nonzero]
        n_hits_result = [len(i) - 2 for i in result_nonzero]

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
            if column in config.data.columns_integer:
                column_index = config.data.columns_integer.index(column)
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

                if config.data.columns_type:
                    column_input = list(
                        np.concatenate(
                            [i[:, column_index + 1] for i in input_nonzero],
                        ),
                    )
                    column_result = list(
                        np.concatenate(
                            [i[:, column_index + 1] for i in result_nonzero],
                        ),
                    )
                    fig, ax = plot_hist(
                        [column_input, column_result],
                        "Local x position",
                        labels=labels,
                    )
                    if fig:
                        pdf.save(fig)

    return output_file


def quick_validate_acts_hits(  # noqa: PLR0912 PLR0915 C901
    config: Configuration,
    model: L.LightningModule,
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

    data = cast(ActsHitsDataModule, data)
    data.setup(data_type.value)
    if logger:
        data.tokenizer.summary(logger)
        data.transformation.summary(logger)

    input_full_int: list[NDArrayType] = []
    input_full_float: list[NDArrayType] = []
    result_full_int: list[NDArrayType] = []
    result_full_float: list[NDArrayType] = []
    if logger:
        logger.info("Starting inference")
    time_start = time.perf_counter()

    for batch in data.get_dataloader(data_type):
        if config.model.type is ModelType.DiscreteTransformer:
            batch_full_int = batch[0]
            if random:
                batch_full_int = batch_full_int[0, :, :].repeat(
                    len(batch_full_int),
                    1,
                    1,
                )
            batch_start_int = batch_full_int[:, :1].to(model.device)

            if config.data.random_int and not no_random:
                batch_start_int[:, :, -1] = torch.randint(
                    1,
                    config.data.random_int,
                    (len(batch[0]), 1),
                )

            result_int, result_float = model.predict(
                (batch_start_int,),
                tokenizer=data.tokenizer,
            )
        elif config.model.type is ModelType.HybridTransformer:
            batch_full_int = batch[0]
            batch_start_int = batch_full_int[:, :1].to(model.device)
            batch_full_float = batch[1]
            batch_start_float = batch_full_float[:, :1].to(model.device)

            if config.data.random_float:
                batch_start_float[:, :, -1] = torch.normal(0, 1, (len(batch[0]), 1))

            result_int, result_float = model.predict(
                (batch_start_int, batch_start_float),
                tokenizer=data.tokenizer,
            )

        input_full_int += list(
            batch_full_int.cpu().numpy() if batch_full_int is not None else [],
        )
        result_full_int += list(
            result_int.cpu().numpy() if result_int is not None else [],
        )

        input_full_float += list(
            batch_full_float.cpu().numpy() if batch_full_float is not None else [],
        )
        result_full_float += list(
            result_float.cpu().numpy() if result_float is not None else [],
        )

        if random:
            break

    input_translated_int: list[NDArrayType] = [
        data.translate_data(i) for i in input_full_int
    ]
    input_translated_float: list[NDArrayType] = [
        data.inverse_data(i) for i in input_full_float
    ]
    result_translated_int: list[NDArrayType] = [
        data.translate_data(i) for i in result_full_int
    ]
    result_translated_float: list[NDArrayType] = [
        data.inverse_data(i) for i in result_full_float
    ]

    time_end = time.perf_counter()

    if logger:
        logger.info(
            "Inference done in %.4f s (%.4f s per 10k particles)",
            time_end - time_start,
            (time_end - time_start)
            / max(len(input_full_int), len(input_full_float))
            * 10000,
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
        logger.info(
            "Total events processed: %d",
            max(len(result_nonzero_int), len(result_nonzero_float)),
        )

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
    with PDFDocument(output_file) as pdf:  # type: ignore
        labels = ["Geant4", "Neural network"]

        n_hits_input = [
            len(i) - 2
            for i in (input_nonzero_int if input_nonzero_int else input_nonzero_float)
        ]
        n_hits_result = [
            len(i) - 2
            for i in (
                result_nonzero_int if result_nonzero_int else result_nonzero_float
            )
        ]
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
            if column in config.data.columns_integer:
                column_index = config.data.columns_integer.index(column)
                column_input = list(
                    np.concatenate([i[:, column_index] for i in input_nonzero_int]),
                )
                column_result = list(
                    np.concatenate([i[:, column_index] for i in result_nonzero_int]),
                )
                fig, ax = plot_hist(
                    [column_input, column_result],
                    column_label,
                    labels=labels,
                )
                if fig:
                    pdf.save(fig)

        columns_list = ["lx", "ly", "tpx", "tpy", "tpz"]
        columns_labels = [
            "Local x position",
            "Local y position",
            "Momentum x",
            "Momentum y",
            "Momentum z",
        ]

        for column, column_label in zip(columns_list, columns_labels, strict=True):
            if column in config.data.columns_float:
                column_index = config.data.columns_float.index(column)
                column_input = list(
                    np.concatenate([i[:, column_index] for i in input_nonzero_float]),
                )
                column_result = list(
                    np.concatenate([i[:, column_index] for i in result_nonzero_float]),
                )
                fig, ax = plot_hist(
                    [column_input, column_result],
                    column_label,
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
    output_file = config.output_path / f"run_{config.run_number}" / "validation.pdf"

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
