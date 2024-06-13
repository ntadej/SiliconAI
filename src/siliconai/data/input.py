"""Data input loading and preprocessing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

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
        if self.config.type is DataType.TRKNtuple:
            self.load_trkntuple()
        else:
            error = f"Unsupported input type: {self.config.type}"
            raise RuntimeError(error)

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
