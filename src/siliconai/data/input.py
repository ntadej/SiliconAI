"""Data input loading and preprocessing."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from siliconai.common.enums import DataType
from siliconai.plotting.common import plot_feature, setup_style
from siliconai.plotting.utils import PDFDocument

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from siliconai.cli.config import DataConfiguration
    from siliconai.cli.logging import Logger


class InputConverter:
    """Input converter class."""

    def __init__(self, logger: Logger, config: DataConfiguration) -> None:
        """Initialize the input converter."""
        self.logger = logger
        self.config = config
        self.data: ArrayLike | None = None

    def load(self) -> None:
        """Load the input."""
        if self.config.type is DataType.ActsChain:
            self.load_acts_chain()
        elif self.config.type is DataType.ActsHits:
            self.load_acts_hits()
        elif self.config.type is DataType.TRKNtuple:
            self.load_trkntuple()
        else:
            error = f"Unsupported input type: {self.config.type}"
            raise RuntimeError(error)

    def process_acts_hits(
        self,
        data_frame: pd.DataFrame,
        column_count: int,
        flatten: bool = False,
    ) -> list[ArrayLike]:
        """Process loaded ACTS hits data."""
        # do auto-padding
        data_frame = cast(
            pd.DataFrame,
            data_frame.unstack(fill_value=0).stack(  # noqa: PD010 PD013
                future_stack=True,
            ),
        )

        if not isinstance(data_frame.index, pd.MultiIndex):
            error = "Index must be a MultiIndex"
            raise TypeError(error)

        data_events = len(data_frame.index.levels[0])
        data_hits = len(data_frame.index.levels[1])

        output_list = list(
            data_frame.to_numpy().reshape(data_events, data_hits, column_count),
        )

        output_nonzero = []
        for i in range(len(output_list)):
            nz = np.nonzero(output_list[i])
            if flatten:
                output_nonzero.append(
                    output_list[i][
                        nz[0].min() : nz[0].max() + 1,
                        nz[1].min() : nz[1].max() + 1,
                    ].flatten()[: -(column_count - 1)],
                )
            else:
                output_nonzero.append(
                    output_list[i][
                        nz[0].min() : nz[0].max() + 1,
                        nz[1].min() : nz[1].max() + 1,
                    ],
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
            data_frame: pd.DataFrame | None = (
                store["hits"][self.config.columns_integer]
                if self.config.columns_integer
                else None
            )

        if data_frame is None:
            error = "No data set to be converted"
            raise ValueError(error)

        # transform quantised coordinates (TODO: do it more flexibly)
        if "lxq" in data_frame:
            data_frame["lxq"] = data_frame["lxq"] * 100 - 1000000
        if "lyq" in data_frame:
            data_frame["lyq"] = data_frame["lyq"] * 100 - 1000000

        # convert to correct types
        data_frame = data_frame.astype("int64")

        self.logger.info("Converting to numpy arrays")

        output_nonzero = self.process_acts_hits(
            data_frame,
            len(self.config.columns_integer),
            flatten=True,
        )

        self.logger.info("Writing to %s", self.config.input_file)

        with self.config.input_file.open("wb") as f:
            pickle.dump(output_nonzero, f)

        self.logger.info("Testing %s", self.config.input_file)

        # test loading
        with self.config.input_file.open("rb") as f:
            loaded_output = pickle.load(f)
            self.logger.info("%s", loaded_output[:3])

    def load_acts_hits(self) -> None:
        """Load the ACTS hits input data."""
        if not self.config.conversion_input_file or not self.config.input_file:
            error = "Input and output files must be set"
            raise ValueError(error)

        self.logger.info(
            "Loading ACTS hits input from %s",
            self.config.conversion_input_file,
        )

        with pd.HDFStore(self.config.conversion_input_file, mode="r") as store:
            data_frame_int: pd.DataFrame | None = (
                store["hits"][self.config.columns_integer]
                if self.config.columns_integer
                else None
            )
            data_frame_float: pd.DataFrame | None = (
                store["hits"][self.config.columns_float]
                if self.config.columns_float
                else None
            )

        if data_frame_int is None and data_frame_float is None:
            error = "No data set to be converted"
            raise ValueError(error)

        # scale quantised coordinates
        if data_frame_int is not None and "lxq" in data_frame_int:
            data_frame_int["lxq"] = data_frame_int["lxq"] * 100
        if data_frame_int is not None and "lyq" in data_frame_int:
            data_frame_int["lyq"] = data_frame_int["lyq"] * 100

        # convert to correct types
        if data_frame_int is not None:
            data_frame_int = data_frame_int.astype("int64")
        if data_frame_float is not None:
            data_frame_float = data_frame_float.astype("float32")

        self.logger.info("Converting to numpy arrays")

        output_int_nonzero = (
            self.process_acts_hits(
                data_frame_int,
                len(self.config.columns_integer),
            )
            if data_frame_int is not None
            else []
        )

        output_float_nonzero = (
            self.process_acts_hits(
                data_frame_float,
                len(self.config.columns_float),
            )
            if data_frame_float is not None
            else []
        )

        self.logger.info("Writing to %s", self.config.input_file)

        with self.config.input_file.open("wb") as f:
            pickle.dump((output_int_nonzero, output_float_nonzero), f)

        self.logger.info("Testing %s", self.config.input_file)

        # test loading
        with self.config.input_file.open("rb") as f:
            loaded_output_int, loaded_output_float = pickle.load(f)
            self.logger.info("%s", loaded_output_int[:3] if loaded_output_int else None)
            self.logger.info(
                "%s",
                loaded_output_float[:3] if loaded_output_float else None,
            )

    def load_trkntuple(self) -> None:
        """Load the input as TRKNtuple."""
        self.logger.info(
            "Loading TRKNtuple input from %s",
            self.config.conversion_input_file,
        )

        import awkward
        import uproot

        tree = uproot.open(f"{self.config.conversion_input_file}:TRKTree")

        # load tracks and truth particles separately
        data_tracks = tree.arrays(
            [
                "track_pt",
                "track_eta",
                "track_phi",
                "track_charge",
                "track_d0",
                "track_z0",
                "track_phi0",
                "track_theta",
                "track_qOverP",
            ],
        )
        data_truth = tree.arrays(
            [
                "truth_pt",
                "truth_eta",
                "truth_phi",
                "truth_mass",
                "truth_charge",
                "truth_pdgId",
                "truth_StdSel",
                "truth_track_index",
            ],
        )
        cleanup_columns = [
            "truth_StdSel",
            "truth_track_index",
        ]

        # filter out truth particles that pass the standard selection
        # and have a valid track link
        selected_truth = data_truth[
            (data_truth.truth_StdSel == 1) & (data_truth.truth_track_index >= 0)
        ]
        # filter out events that have one particle with one track linked
        filtered_truth = selected_truth[
            awkward.num(selected_truth.truth_track_index) == 1
        ]

        # select same events as for truth particles
        selected_tracks = data_tracks[
            awkward.num(selected_truth.truth_track_index) == 1
        ]
        # filter tracks that are linked to a truth particle
        filtered_tracks = selected_tracks[filtered_truth.truth_track_index]

        # zip and flatten the data into a single awkward array
        data_columns = dict(
            zip(
                awkward.fields(filtered_truth[:, 0]),
                awkward.unzip(filtered_truth[:, 0]),
                strict=True,
            ),
        ) | dict(
            zip(
                awkward.fields(filtered_tracks[:, 0]),
                awkward.unzip(filtered_tracks[:, 0]),
                strict=True,
            ),
        )
        output_data = awkward.zip(data_columns)
        for column in cleanup_columns:
            del output_data[column]

        # cache the result
        self.data = output_data.to_numpy()
        if self.data is not None and self.config.input_file is not None:
            np.save(self.config.input_file, self.data)
            self.logger.info("Saved %s", self.config.input_file)

    def diagnostics(self) -> None:
        """Make diagnostics plots."""
        if not self.data or not hasattr(self.data, "dtype"):
            return

        setup_style()

        with PDFDocument(f"{self.output_file}.diagnostics.pdf") as pdf:  # type: ignore
            for column in self.data.dtype.names:
                self.logger.info("Plotting %s", column)
                fig, ax = plot_feature(self.data, column)
                if not fig:
                    continue
                pdf.save(fig)
