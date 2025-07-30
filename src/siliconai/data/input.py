"""Data input loading and preprocessing."""

from __future__ import annotations

import itertools
import pickle
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from siliconai.common.enums import ColumnType, DataType
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
        if self.config.type is DataType.ActsChain:
            self.load_acts_chain()
            return
        if self.config.type is DataType.ActsHits:
            self.load_acts_hits()
            return
        raise RuntimeError

    @staticmethod
    def process_acts_hits(
        data_frame: pd.DataFrame,
        column_count: int,
        padding_token: int = 0,
        flatten: bool = False,
        random_int: int = 0,
    ) -> list[NDArrayType]:
        """Process loaded ACTS hits data."""
        if not isinstance(data_frame.index, pd.MultiIndex):
            error = "Index must be a MultiIndex"
            raise TypeError(error)

        data_events = len(data_frame.index.levels[0])
        data_hits = len(data_frame.index.levels[1])
        data_hits_exact = data_frame.reset_index().groupby("event_id").count()["index"]

        # do auto-padding
        data_frame = cast(
            "pd.DataFrame",
            data_frame.unstack(fill_value=padding_token).stack(  # noqa: PD010 PD013
                future_stack=True,
            ),
        )

        if random_int:
            event_rnd_int = np.random.randint(1, random_int, data_events)  # noqa: NPY002
            all_rnd_int = np.array(
                [
                    data_hits_exact[i] * [rnd] + (data_hits - data_hits_exact[i]) * [0]
                    for i, rnd in enumerate(event_rnd_int)
                ],
            ).flatten()
            data_frame["random_int"] = all_rnd_int

            column_count += 1

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

        return output_nonzero

    def load_acts_chain(self) -> None:
        """Load the ACTS hits as chain."""
        if not self.config.conversion_input_file or not self.config.input_file:
            error = "Input and output files must be set"
            raise ValueError(error)

        self.logger.info(
            "Loading ACTS hits chain input from %s",
            self.config.conversion_input_file,
        )

        with pd.HDFStore(self.config.conversion_input_file, mode="r") as store:
            data_frame: pd.DataFrame | None = store["hits"][self.config.columns]

        if data_frame is None:
            error = "No data set to be converted"
            raise ValueError(error)

        # transform numerical columns
        numerical_columns = ["lxq", "lyq", "tpxq", "tpyq", "tpzq"]
        for column in numerical_columns:
            if column in data_frame:
                data_frame[f"{column}2"], data_frame[f"{column}1"] = np.modf(
                    data_frame[column],
                )
                data_frame[f"{column}1"] = data_frame[f"{column}1"]
                data_frame[f"{column}2"] = abs(round(data_frame[f"{column}2"] * 100))
                del data_frame[column]

        # convert to correct types
        data_frame = data_frame.astype("int64")

        self.logger.info("Converting to numpy arrays")

        ncolumns = len(self.config.columns) + len(
            [c for c in self.config.columns_type if c == ColumnType.Numerical],
        )

        output_nonzero = self.process_acts_hits(
            data_frame,
            ncolumns,
            padding_token=self.config.padding_token,
            flatten=True,
        )

        output_nonzero_chunked = None
        if self.config.max_blocks > 0:
            output_nonzero_chunked = [
                sliding_subarrays(
                    arr,
                    self.config.block_size * self.config.max_blocks + 1,
                    self.config.block_size,
                )
                for arr in output_nonzero
            ]
            output_nonzero_chunked = list(
                itertools.chain.from_iterable(output_nonzero_chunked),
            )

        self.logger.info("Writing to %s", self.config.input_file)

        with self.config.input_file.open("wb") as f:
            pickle.dump([output_nonzero, output_nonzero_chunked], f)

        self.logger.info("Testing %s", self.config.input_file)

        # test loading
        with self.config.input_file.open("rb") as f:
            loaded_output, loaded_output_chunked = pickle.load(f)
            self.logger.info("%s", loaded_output[-3:])
            if loaded_output_chunked is not None:
                self.logger.info("%s", loaded_output_chunked[-3:])

    def load_acts_hits(self) -> None:
        """Load the ACTS hits input data."""
        if not self.config.conversion_input_file or not self.config.input_file:
            error = "Input and output files must be set"
            raise ValueError(error)

        self.logger.info(
            "Loading ACTS hits input from %s",
            self.config.conversion_input_file,
        )

        columns = self.config.columns[:]
        if self.config.random_int:
            columns.remove("random_int")

        with pd.HDFStore(self.config.conversion_input_file, mode="r") as store:
            data_frame: pd.DataFrame = store["hits"][columns]

        # scale quantised coordinates
        columns_scale = ["lxq", "lyq", "tpxq", "tpyq", "tpzq"]
        for column in columns_scale:
            if column in data_frame:
                data_frame[column] = data_frame[column] * 100

        # convert to correct types
        if data_frame is not None:
            data_frame = data_frame.astype("int64")

        self.logger.info("Converting to numpy arrays")

        output_int_nonzero = self.process_acts_hits(
            data_frame,
            len(columns),
            padding_token=self.config.padding_token,
            random_int=self.config.random_int,
        )

        self.logger.info("Writing to %s", self.config.input_file)

        with self.config.input_file.open("wb") as f:
            pickle.dump(output_int_nonzero, f)

        self.logger.info("Testing %s", self.config.input_file)

        # test loading
        with self.config.input_file.open("rb") as f:
            loaded_output = pickle.load(f)
            self.logger.info("%s", loaded_output[:3])
