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

from siliconai.common.enums import DataLoadingType

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
        transforms: list[NDArrayTransformation] | None = None,
        full_data: bool = False,
        logger: Logger | None = None,
    ) -> None:
        """Load the ActsHits as a dataset."""
        self.logger = logger
        self.input_path = input_path
        self.input_suffix = input_suffix
        self.transforms = transforms
        self.full_data = full_data

        self.current_file_index = -1
        self.current_file_offset = 0

        with (self.input_path / f"conversion_info_{self.input_suffix}.json").open(
            "r",
        ) as f:
            self.metadata = load(f)

    def load_data_for_file_index(self, index: int) -> None:
        """Load data from a file with an index."""
        with (self.input_path / f"{index + 1}_{self.input_suffix}.pkl").open("rb") as f:
            data, data_chunked = pickle.load(f)
            self.data = (
                data_chunked
                if not self.full_data and data_chunked is not None
                else data
            )
        self.current_file_index = index
        self.current_file_offset = self.metadata["starts"][index]
        if self.logger:
            self.logger.info(
                "Loaded %d sequences from %s",
                len(self.data),
                self.input_path / f"{index + 1}_{self.input_suffix}.pkl",
            )

    def file_index_for_idx(self, idx: int) -> int:
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
        if self.file_index_for_idx(idx) != self.current_file_index:
            self.load_data_for_file_index(self.file_index_for_idx(idx))

        sequence: NDArrayType = self.data[idx - self.current_file_offset]

        if self.transforms:
            for t in self.transforms:
                sequence = t(sequence)

        return sequence


class ActsChainSubDataset(Dataset):  # type: ignore[type-arg]
    """ActsChain sub-dataset."""

    def __init__(
        self,
        data: NDArrayType,
        start_idx: int,
        end_idx: int,
        transforms: list[NDArrayTransformation] | None = None,
        logger: Logger | None = None,
    ) -> None:
        """Load the ActsHits as a sub-dataset."""
        self.logger = logger
        self.data = data
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.transforms = transforms

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.end_idx - self.start_idx

    def __getitem__(self, idx: int) -> NDArrayType:
        """Return the item at the given index."""
        sequence: NDArrayType = self.data[self.start_idx + idx]

        if self.transforms:
            for t in self.transforms:
                sequence = t(sequence)

        return sequence


class ActsChainSubDatasetFactory:
    """ActsChain sub-dataset factory."""

    def __init__(
        self,
        input_path: Path,
        input_suffix: str,
        loading_type: DataLoadingType,
        epoch_size: int,
        transforms: list[NDArrayTransformation] | None = None,
        full_data: bool = False,
        logger: Logger | None = None,
    ) -> None:
        """Initialise ActsChain sub-dataset factory."""
        if epoch_size <= 0:
            error = "Epoch size must be greater than 0 for sub-dataset factory."
            raise ValueError(error)

        self.logger = logger
        self.input_path = input_path
        self.input_suffix = input_suffix
        self.loading_type = loading_type
        self.epoch_size = epoch_size
        self.transforms = transforms
        self.full_data = full_data

        self.epoch_ranges: list[tuple[int, int, int]] = []
        self.current_epoch = 0
        self.current_epoch_range: tuple[int, int, int] = (0, 0, 0)
        self.current_file_index = -1

        self.build_file_splitting()

    def build_file_splitting(self) -> None:
        """Build file splitting based on epoch size."""
        with (self.input_path / f"conversion_info_{self.input_suffix}.json").open(
            "r",
        ) as f:
            self.metadata = load(f)

        truncated_sizes = [
            (size // self.epoch_size) * self.epoch_size
            for size in self.metadata["sizes"]
        ]

        for i, size in enumerate(truncated_sizes):
            if i == 0 and self.loading_type is DataLoadingType.fit:
                continue
            if i > 0 and self.loading_type is not DataLoadingType.fit:
                break
            n_epochs = size // self.epoch_size
            self.epoch_ranges.extend(
                (
                    i,
                    epoch * self.epoch_size,
                    (epoch + 1) * self.epoch_size,
                )
                for epoch in range(n_epochs)
            )

        self.set_current_epoch(
            1 if self.loading_type is DataLoadingType.validate else 0,
        )

    def set_current_epoch(self, current_epoch: int) -> None:
        """Run to start with correct data for the current epoch."""
        self.current_epoch = current_epoch
        self.current_epoch_range = self.epoch_ranges[
            current_epoch % len(self.epoch_ranges)
        ]
        if self.current_epoch_range[0] != self.current_file_index:
            self.load_data_for_file_index(self.current_epoch_range[0])

    def load_data_for_file_index(self, index: int) -> None:
        """Load data from a file with an index."""
        with (self.input_path / f"{index + 1}_{self.input_suffix}.pkl").open("rb") as f:
            data, data_chunked = pickle.load(f)
            self.data = (
                data_chunked
                if not self.full_data and data_chunked is not None
                else data
            )
        self.current_file_index = index
        if self.logger:
            self.logger.info(
                "Loaded %d sequences from %s",
                len(self.data),
                self.input_path / f"{index + 1}_{self.input_suffix}.pkl",
            )

    def generate_dataset(self) -> ActsChainSubDataset:
        """Generate the sub-dataset for the current epoch."""
        start_idx = self.current_epoch_range[1]
        end_idx = self.current_epoch_range[2]
        return ActsChainSubDataset(
            data=self.data,
            start_idx=start_idx,
            end_idx=end_idx,
            transforms=self.transforms,
            logger=self.logger,
        )
