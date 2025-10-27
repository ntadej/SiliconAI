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
from torch.utils.data import DataLoader

from siliconai.common.enums import DataLoadingType
from siliconai.data.datasets import ActsChainSubDatasetFactory
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
        self.epoch_size = config.data.epoch_size
        self.workers = config.data.workers
        self.current_epoch: int = 0

        self.collate_fn: CollateFnType | None = collate_fn

        self.train_data: ActsChainSubDatasetFactory
        self.val_data: ActsChainSubDatasetFactory
        self.test_data: ActsChainSubDatasetFactory
        self.predict_data: ActsChainSubDatasetFactory

        self.first_init = True
        self.from_checkpoint = False

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

    def set_current_epoch(
        self,
        current_epoch: int,
        from_checkpoint: bool = False,
    ) -> None:
        """Run to start with correct data for the current epoch."""
        self.current_epoch = current_epoch
        self.train_data.set_current_epoch(current_epoch)
        if from_checkpoint:
            self.from_checkpoint = True

    def train_dataloader(self) -> DataLoader[Any]:
        """Return the training DataLoader."""
        if self.first_init:
            self.first_init = False
        # it seems we init the data loader before the set_current_epoch callback
        # is called
        elif self.from_checkpoint:
            self.from_checkpoint = False
        else:
            self.set_current_epoch(self.current_epoch + 1)
        return DataLoader(
            self.train_data.generate_dataset(),
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return the validation DataLoader."""
        return DataLoader(
            self.val_data.generate_dataset(),
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return the test DataLoader."""
        return DataLoader(
            self.test_data.generate_dataset(),
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        """Return the predict DataLoader."""
        return DataLoader(
            self.predict_data.generate_dataset(),
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
        self.epoch_size: int = (
            data_config.data.epoch_size if data_config.data.epoch_size else 0
        )
        self.full_data: bool = False

        self.tokenizer = SequenceTokenizer.load(data_config.data, logger)

        self.save_hyperparameters("data_config")

    def translate_data(self, data: NDArrayType) -> NDArrayType:
        """Translate back from tokens to the original data."""
        return self.tokenizer.inverse(data)

    def setup(self, stage: str) -> None:
        """Transform and setup the ACTS dataset."""
        if stage == "fit":
            self.train_data = ActsChainSubDatasetFactory(
                self.data_path,
                self.data_suffix,
                DataLoadingType.fit,
                self.epoch_size,
                transforms=[self.tokenizer],
                full_data=self.full_data,
            )
        if stage in {"fit", "validate"}:
            self.val_data = ActsChainSubDatasetFactory(
                self.data_path,
                self.data_suffix,
                DataLoadingType.validate,
                self.epoch_size,
                transforms=[self.tokenizer],
                full_data=self.full_data,
            )
        if stage == "test":
            self.test_data = ActsChainSubDatasetFactory(
                self.data_path,
                self.data_suffix,
                DataLoadingType.test,
                self.epoch_size,
                transforms=[self.tokenizer],
                full_data=self.full_data,
            )
        if stage == "predict":
            self.predict_data = ActsChainSubDatasetFactory(
                self.data_path,
                self.data_suffix,
                DataLoadingType.predict,
                self.epoch_size,
                transforms=[self.tokenizer],
                full_data=self.full_data,
            )
