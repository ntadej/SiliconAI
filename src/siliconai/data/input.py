# Copyright (C) 2024 Tadej Novak
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# SPDX-License-Identifier: MPL-2.0

"""Data input loading and preprocessing."""

from __future__ import annotations

import itertools
import pickle
from json import dump
from multiprocessing import Pool
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from siliconai.data.utils import sliding_subarrays

if TYPE_CHECKING:
    from siliconai.cli.config import DataConfiguration
    from siliconai.cli.logger import Logger
    from siliconai.data.utils import NDArrayType


class InputConverter:
    """Input converter class."""

    def __init__(self, logger: Logger, config: DataConfiguration) -> None:
        """Initialize the input converter."""
        self.logger = logger
        self.config = config

    def load(self) -> None:
        """Load the input."""
        if not self.config.conversion_input_path or not self.config.input_suffix:
            error = "Input file and output suffix must be set"
            raise ValueError(error)

        self.logger.info(
            "Loading ACTS hits chain input from %s",
            self.config.conversion_input_path,
        )

        # run the process pool
        with Pool(self.config.workers) as p:
            sizes = p.starmap(
                self.load_acts_chain,
                zip(range(1, self.config.nfiles + 1), strict=True),
            )

        self.logger.info("Processed files with sizes: %s", sizes)
        with (
            self.config.conversion_input_path
            / f"conversion_info_{self.config.input_suffix}.json"
        ).open(
            "w",
        ) as f:
            dump(
                {
                    "nfiles": self.config.nfiles,
                    "sizes": sizes,
                    "starts": list(itertools.accumulate([0, *sizes[:-1]])),
                    "ends": list(itertools.accumulate(sizes)),
                },
                f,
            )

    def process_acts_hits(
        self,
        data_frame: pd.DataFrame,
        column_count: int,
        flatten: bool = False,
    ) -> tuple[list[NDArrayType], pd.DataFrame]:
        """Process loaded ACTS hits data."""
        if not isinstance(data_frame.index, pd.MultiIndex):
            error = "Index must be a MultiIndex"
            raise TypeError(error)

        data_events = len(data_frame.index.levels[0])
        data_hits = len(data_frame.index.levels[1])
        data_hits_exact = data_frame.reset_index().groupby("event_id").count()["index"]

        # handle index as a feature
        if self.config.index_with_offset >= 0:
            column_count += 1
            data_frame = data_frame.reset_index().set_index("event_id")
            data_frame["index_orig"] = data_frame["index"]
            data_frame["index"] += self.config.index_with_offset
            data_frame = data_frame.reset_index().set_index(["event_id", "index_orig"])

        data_frame_final = data_frame

        # do auto-padding
        data_frame = cast(
            "pd.DataFrame",
            data_frame.unstack(fill_value=self.config.padding_token).stack(  # noqa: PD010 PD013
                future_stack=True,
            ),
        )

        output_list = list(
            data_frame.to_numpy().reshape(data_events, data_hits, column_count),
        )

        output_nonzero: list[NDArrayType] = []
        for i in range(len(output_list)):
            if flatten:
                output_nonzero.append(
                    np.trim_zeros(output_list[i][: data_hits_exact[i], :].flatten()),
                )
            else:
                output_nonzero.append(
                    output_list[i][: data_hits_exact[i], :],
                )

        return output_nonzero, data_frame_final

    def load_acts_chain(self, index: int) -> int:
        """Load the ACTS hits as chain."""
        if self.config.conversion_input_path is None:
            error = "Input path not set"
            raise ValueError(error)

        input_file = self.config.conversion_input_path / f"{index}.h5"
        output_file = (
            self.config.conversion_input_path
            / f"{index}_{self.config.input_suffix}.pkl"
        )
        output_metadata_file = (
            self.config.conversion_input_path
            / f"{index}_{self.config.input_suffix}_metadata.json"
        )

        if output_file.exists():
            self.logger.info("File %s already exists, skipping", output_file)
            with output_file.open("rb") as f:
                loaded_output, loaded_output_chunked = pickle.load(f)
                self.logger.info("%s", loaded_output[-3:])
                if loaded_output_chunked is not None:
                    self.logger.info("%s", loaded_output_chunked[-3:])

            return (
                len(loaded_output_chunked)
                if loaded_output_chunked is not None
                else len(loaded_output)
            )

        with pd.HDFStore(input_file, mode="r") as store:
            data_frame: pd.DataFrame | None = store["hits"].set_index(
                ["event_id", "index"],
            )[self.config.columns]

        if data_frame is None:
            error = "No data set to be converted"
            raise ValueError(error)

        ncolumns = len(self.config.columns)

        # transform numerical columns
        numerical_columns = ["lxq", "lyq", "tpxq", "tpyq", "tpzq"]
        for column in numerical_columns:
            if column in data_frame:
                if self.config.split_numerical:
                    data_frame[f"{column}2"], data_frame[f"{column}1"] = np.modf(
                        data_frame[column],
                    )
                    data_frame[f"{column}1"] = data_frame[f"{column}1"]
                    data_frame[f"{column}2"] = abs(
                        round(data_frame[f"{column}2"] * 100),
                    )
                    ncolumns += 1
                    del data_frame[column]
                else:
                    data_frame[column] = data_frame[column] * 100

        # convert to correct types
        data_frame = data_frame.astype("int64")

        self.logger.info("Converting to numpy arrays")

        output_nonzero, data_frame_final = self.process_acts_hits(
            data_frame,
            ncolumns,
            flatten=True,
        )

        output_nonzero_chunked = None
        if self.config.max_blocks > 0:
            chunk_size = (
                self.config.block_size * self.config.max_blocks
                + 1
                + int(self.config.index_with_offset >= 0)
            )
            output_nonzero_chunked = [
                sliding_subarrays(
                    arr,
                    chunk_size,
                    self.config.block_size,
                )
                for arr in output_nonzero
                if len(arr) >= chunk_size
            ]
            output_nonzero_chunked = list(
                itertools.chain.from_iterable(output_nonzero_chunked),
            )

        # pre-cache unique column values
        unique_values = {
            column: data_frame_final[column].unique().tolist()
            for column in data_frame_final.columns
        }

        with output_metadata_file.open("w") as f:
            dump({"unique_values": unique_values}, f)

        # output_file
        self.logger.info("Writing to %s", output_file)

        with output_file.open("wb") as f:
            pickle.dump([output_nonzero, output_nonzero_chunked], f)

        self.logger.info("Testing %s", output_file)

        # test loading
        with output_file.open("rb") as f:
            loaded_output, loaded_output_chunked = pickle.load(f)
            self.logger.info("%s", loaded_output[-3:])
            if loaded_output_chunked is not None:
                self.logger.info("%s", loaded_output_chunked[-3:])

        return (
            len(output_nonzero_chunked)
            if output_nonzero_chunked is not None
            else len(output_nonzero)
        )
