"""Data input loading and preprocessing."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

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
        if self.config.type is DataType.ActsHits:
            self.load_acts_hits()
        elif self.config.type is DataType.TRKNtuple:
            self.load_trkntuple()
        else:
            error = f"Unsupported input type: {self.config.type}"
            raise RuntimeError(error)

    def load_acts_hits(self) -> None:
        """Load the ACTS hits input data."""
        if not self.config.conversion_input_file or not self.config.input_file:
            error = "Input and output files must be set"
            raise ValueError(error)

        self.logger.info(
            "Loading ACTS hits input from %s",
            self.config.conversion_input_file,
        )

        # TODO: make configurable
        column_list = [
            "geometry_id",
            "particle_type",
        ]

        with pd.HDFStore(self.config.conversion_input_file, mode="r") as store:
            data_frame = store["hits"][column_list].astype("int64")

        # do auto-padding
        data_frame = data_frame.unstack(fill_value=0).stack(future_stack=True)  # noqa: PD010 PD013

        if not isinstance(data_frame.index, pd.MultiIndex):
            error = "Index must be a MultiIndex"
            raise TypeError(error)

        self.logger.info("Converting to numpy arrays")

        data_events = len(data_frame.index.levels[0])
        data_hits = len(data_frame.index.levels[1])

        output = list(
            data_frame.to_numpy().reshape(data_events, data_hits, len(column_list)),
        )

        output_nonzero = []
        for i in range(len(output)):
            nz = np.nonzero(output[i])
            output_nonzero.append(
                output[i][nz[0].min() : nz[0].max() + 1, nz[1].min() : nz[1].max() + 1],
            )

        self.logger.info("Writing to %s", self.config.input_file)

        with self.config.input_file.open("wb") as f:
            pickle.dump(output_nonzero, f)

        self.logger.info("Testing %s", self.config.input_file)

        # test loading
        with self.config.input_file.open("rb") as f:
            loaded_output = pickle.load(f)  # noqa: F841

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
