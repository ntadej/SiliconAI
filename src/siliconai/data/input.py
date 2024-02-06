"""Data input loading and preprocessing."""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from siliconai.plotting.common import plot_column, setup_style
from siliconai.plotting.utils import PDFDocument

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import ArrayLike

    from siliconai.cli.logging import Logger


class InputType(Enum):
    """Input type."""

    TRKNtuple = "TRKNtuple"


class InputLoader:
    """Input loader class."""

    def __init__(
        self,
        logger: Logger,
        input_type: InputType,
        input_file: Path,
        output_file: Path,
    ) -> None:
        """Initialize the input loader."""
        self.logger = logger
        self.input_type = input_type
        self.input_file = input_file
        self.output_file = output_file
        self.data: ArrayLike | None = None

    def load(self) -> None:
        """Load the input."""
        if self.input_type == InputType.TRKNtuple:
            self.load_trkntuple()

    def load_trkntuple(self) -> None:
        """Load the input as TRKNtuple."""
        self.logger.info("Loading TRKNtuple input from %s", self.input_file)

        import awkward
        import uproot

        tree = uproot.open(f"{self.input_file}:TRKTree")

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
        if self.data is not None:
            np.save(self.output_file, self.data)
            self.logger.info("Saved %s", self.output_file)

    def diagnostics(self) -> None:
        """Make diagnostics plots."""
        if not self.data or not hasattr(self.data, "dtype"):
            return

        setup_style()

        with PDFDocument(f"{self.output_file}.diagnostics.pdf") as pdf:  # type: ignore
            for column in self.data.dtype.names:
                self.logger.info("Plotting %s", column)
                fig, ax = plot_column(self.data, column)
                if not fig:
                    continue
                pdf.save(fig)
