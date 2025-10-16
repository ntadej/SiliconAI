# Copyright (C) 2024 Tadej Novak
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# SPDX-License-Identifier: MPL-2.0

"""Custom datasets."""

from __future__ import annotations

import pickle
from json import load
from typing import TYPE_CHECKING

from torch.utils.data import Dataset

if TYPE_CHECKING:
    from pathlib import Path

    from siliconai.cli.logger import Logger
    from siliconai.data.utils import (
        NDArrayTransformation,
        NDArrayType,
    )


class ActsChainDataset(Dataset):  # type: ignore[type-arg]
    """ActsChain dataset."""

    def __init__(
        self,
        input_path: Path,
        input_suffix: str,
        nfiles: int = 1,
        events_per_file: int = -1,
        transforms: list[NDArrayTransformation] | None = None,
        full_data: bool = False,
        logger: Logger | None = None,
    ) -> None:
        """Load the ActsHits as a dataset."""
        self.logger = logger
        self.input_path = input_path
        self.input_suffix = input_suffix
        self.nfiles = nfiles
        self.events_per_file = events_per_file
        self.transforms = transforms
        self.full_data = full_data

        self.current_index = -1
        self.current_offset = self.current_index * self.events_per_file

        with (self.input_path / f"conversion_info_{input_suffix}.json").open("r") as f:
            self.metadata = load(f)

        if self.events_per_file < 0:
            self.load_data_for_index(0)
            self.events_per_file = len(self.data)

    def load_data_for_index(self, index: int) -> None:
        """Load data from a file with an index."""
        with (self.input_path / f"{index + 1}_{self.input_suffix}.pkl").open("rb") as f:
            data, data_chunked = pickle.load(f)
            self.data = (
                data_chunked
                if not self.full_data and data_chunked is not None
                else data
            )
        self.current_index = index
        self.current_offset = self.metadata["starts"][index]
        if self.logger:
            self.logger.info(
                "Loaded %d sequences from %s",
                len(self.data),
                self.input_path / f"{index + 1}_{self.input_suffix}.pkl",
            )

    def index_for_idx(self, idx: int) -> int:
        """Get the file index for a given dataset index."""
        for i, end in enumerate(self.metadata["ends"]):
            if idx < end:
                return i

        error = "Index out of range"
        raise IndexError(error)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return sum(self.metadata["sizes"])

    def __getitem__(self, idx: int) -> NDArrayType:
        """Return the item at the given index."""
        if self.index_for_idx(idx) != self.current_index:
            self.load_data_for_index(self.index_for_idx(idx))
        sequence: NDArrayType = self.data[idx - self.current_offset]

        if self.transforms:
            for t in self.transforms:
                sequence = t(sequence)

        return sequence
