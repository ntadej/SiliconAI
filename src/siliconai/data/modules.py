# Copyright (C) 2024 Tadej Novak
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# SPDX-License-Identifier: MPL-2.0

"""Data modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import lightning as L
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from siliconai.common.enums import DataLoadingType
from siliconai.data.datasets import ActsChainDataset
from siliconai.data.tokenizers import SequenceTokenizer
from siliconai.data.utils import CollateFnType, NDArrayType, collate_sequence

if TYPE_CHECKING:
    from pathlib import Path

    from siliconai.cli.config import Configuration
    from siliconai.cli.logger import Logger


class BaseDataModule(L.LightningDataModule):
    """Base data module."""

    def __init__(
        self,
        config: Configuration,
        collate_fn: CollateFnType | None = None,
    ) -> None:
        """Initialize the data module."""
        super().__init__()
        self.split_ratio = config.data.split_ratio
        self.batch_size = config.data.batch_size
        self.workers = config.data.workers

        self.collate_fn: CollateFnType | None = collate_fn

        self.train_data: Subset[Any] | Dataset[Any]
        self.val_data: Subset[Any] | Dataset[Any]
        self.test_data: Subset[Any] | Dataset[Any]
        self.predict_data: Subset[Any] | Dataset[Any]

    def get_dataloader(self, data_type: DataLoadingType) -> DataLoader[Any]:
        """Get the DataLoader for the specified data type."""
        if data_type == DataLoadingType.fit:
            return self.train_dataloader()
        if data_type == DataLoadingType.validate:
            return self.val_dataloader()
        if data_type == DataLoadingType.test:
            return self.test_dataloader()
        if data_type == DataLoadingType.predict:
            return self.predict_dataloader()
        raise RuntimeError

    def train_dataloader(self) -> DataLoader[Any]:
        """Return the training DataLoader."""
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return the validation DataLoader."""
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return the test DataLoader."""
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        """Return the predict DataLoader."""
        return DataLoader(
            self.predict_data,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            shuffle=False,
        )


class ActsChainDataModule(BaseDataModule):
    """ACTS-based silicon detector hits chain data module."""

    def __init__(
        self,
        data_config: Configuration,
        logger: Logger | None = None,
    ) -> None:
        """Initialize the data module."""
        super().__init__(data_config, collate_sequence)
        if not data_config.data.input_path:
            error = "Invalid configuration"
            raise RuntimeError(error)

        self.logger = logger

        self.data_path: Path = data_config.data.input_path
        self.data_suffix: str = data_config.data.input_suffix
        self.nfiles: int = data_config.data.nfiles
        self.full_data: bool = False

        self.tokenizer = SequenceTokenizer.load(data_config.data, logger)

        self.save_hyperparameters("data_config")

    def translate_data(self, data: NDArrayType) -> NDArrayType:
        """Translate back from tokens to the original data."""
        return self.tokenizer.inverse(data)

    def setup(self, stage: str) -> None:  # noqa: ARG002
        """Transform and setup the ACTS dataset."""
        self.train_data, self.val_data, self.test_data = random_split(
            ActsChainDataset(
                self.data_path,
                self.data_suffix,
                self.nfiles,
                transforms=[self.tokenizer],
                full_data=self.full_data,
            ),
            self.split_ratio,
        )
        self.predict_data = ActsChainDataset(
            self.data_path,
            self.data_suffix,
            self.nfiles,
            transforms=[self.tokenizer],
            full_data=self.full_data,
        )
