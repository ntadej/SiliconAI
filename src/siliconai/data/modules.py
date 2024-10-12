"""Data modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import lightning as L
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import MNIST, FashionMNIST  # type: ignore
from torchvision.transforms import ToTensor  # type: ignore

from siliconai.common.enums import DataLoadingType
from siliconai.data.datasets import (
    ActsChainDataset,
    ActsHitsDataset,
    TRKNtupleDataset,
)
from siliconai.data.tokenizers import ColumnTokenizer, SequenceTokenizer
from siliconai.data.transformations import ScikitLearnTransformation
from siliconai.data.utils import (
    CollateFnType,
    NDArrayToFloatTensor,
    NDArrayType,
    collate_sequence,
    collate_sequence_chain,
)

if TYPE_CHECKING:
    from pathlib import Path

    from siliconai.cli.config import Configuration
    from siliconai.cli.logging import Logger


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

        self.train_data: Subset[Any]
        self.val_data: Subset[Any]
        self.test_data: Subset[Any]
        self.predict_data: Subset[Any]

    def get_dataloader(self, data_type: DataLoadingType) -> DataLoader[Any]:
        """Get the DataLoader for the specified data type."""
        if data_type == DataLoadingType.fit:
            return self.train_dataloader()
        if data_type == DataLoadingType.validate:
            return self.val_dataloader()
        if data_type == DataLoadingType.test:
            return self.test_dataloader()
        if data_type == DataLoadingType.predict:  # noqa: RET503
            return self.predict_dataloader()

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


class MNISTDataModule(BaseDataModule):
    """MNIST data module."""

    def __init__(self, config: Configuration) -> None:
        """Initialize the data module."""
        super().__init__(config)
        self.data_path = config.global_config.data_path / "common_datasets"
        self.fashion = False
        self.save_hyperparameters(ignore=["config"])

    def prepare_data(self) -> None:
        """Download and prepare the MNIST dataset."""
        loader = FashionMNIST if self.fashion else MNIST
        loader(self.data_path, train=True, download=True)
        loader(self.data_path, train=False, download=True)

    def setup(self, stage: str) -> None:  # noqa: ARG002
        """Transform and setup the MNIST dataset."""
        loader = FashionMNIST if self.fashion else MNIST
        self.train_data = loader(self.data_path, train=True, transform=ToTensor())
        self.val_data, self.test_data = random_split(
            loader(self.data_path, train=False, transform=ToTensor()),
            [0.5, 0.5],
        )
        self.predict_data = loader(self.data_path, train=False, transform=ToTensor())


class FashionMNISTDataModule(MNISTDataModule):
    """FashionMNIST data module."""

    def __init__(self, config: Configuration) -> None:
        """Initialize the data module."""
        super().__init__(config)
        self.fashion = True


class ActsChainDataModule(BaseDataModule):
    """ACTS-based silicon detector hits chain data module."""

    def __init__(
        self,
        data_config: Configuration,
        logger: Logger | None = None,
    ) -> None:
        """Initialize the data module."""
        super().__init__(data_config, collate_sequence_chain)
        if not data_config.data.input_file:
            error = "ACTS data path not set."
            raise ValueError(error)

        self.logger = logger

        self.data_path: Path = data_config.data.input_file

        self.tokenizer = SequenceTokenizer.load(data_config.data, logger)

        self.save_hyperparameters("data_config")

    def translate_data(self, data: NDArrayType) -> NDArrayType:
        """Translate back from tokens to the original data."""
        return self.tokenizer.inverse(data)

    def setup(self, stage: str) -> None:  # noqa: ARG002
        """Transform and setup the ACTS dataset."""
        self.train_data, self.val_data, self.test_data = random_split(
            ActsChainDataset(self.data_path, transforms=[self.tokenizer]),
            self.split_ratio,
        )
        self.predict_data = ActsChainDataset(  # type: ignore
            self.data_path,
            transforms=[self.tokenizer],
        )


class ActsHitsDataModule(BaseDataModule):
    """ACTS-based silicon detector hits data module."""

    def __init__(
        self,
        data_config: Configuration,
        logger: Logger | None = None,
    ) -> None:
        """Initialize the data module."""
        super().__init__(data_config, collate_sequence)
        if not data_config.data.input_file:
            error = "ACTS data path not set."
            raise ValueError(error)

        self.logger = logger

        self.data_path: Path = data_config.data.input_file

        self.input_dim_discreet: list[int]
        if isinstance(data_config.data.input_dim, int):
            self.input_dim_discreet = [data_config.data.input_dim]
        else:
            self.input_dim_discreet = data_config.data.input_dim[:]
        self.input_dim_continuous = len(data_config.data.columns_float)

        self.tokenizer = ColumnTokenizer.load(data_config.data, logger)
        self.transformation = ScikitLearnTransformation.load(data_config.data, logger)

        self.save_hyperparameters("data_config")

    def translate_data(self, data: NDArrayType) -> NDArrayType:
        """Translate back from tokens to the original data."""
        return self.tokenizer.inverse(data)

    def inverse_data(self, data: NDArrayType) -> NDArrayType:
        """Inverse normalization on continuous data."""
        return self.transformation.inverse(data)

    def setup(self, stage: str) -> None:  # noqa: ARG002
        """Transform and setup the ACTS dataset."""
        self.train_data, self.val_data, self.test_data = random_split(
            ActsHitsDataset(
                self.data_path,
                transforms_int=[self.tokenizer],
                transforms_float=[self.transformation],
            ),
            self.split_ratio,
        )
        self.predict_data = ActsHitsDataset(  # type: ignore
            self.data_path,
            transforms_int=[self.tokenizer],
            transforms_float=[self.transformation],
        )


class TRKNtupleDataModule(BaseDataModule):
    """TRKNtuple data module."""

    def __init__(self, config: Configuration) -> None:
        """Initialize the data module."""
        super().__init__(config)
        self.data_path = config.data.input_file

        self.save_hyperparameters(ignore=["config"])

    def setup(self, stage: str) -> None:  # noqa: ARG002
        """Transform and setup the TRKNtuple dataset."""
        if self.data_path is None:
            error = "TRKNtuple data path not set."
            raise ValueError(error)

        dataset = TRKNtupleDataset(
            self.data_path,
            tensor_transform=NDArrayToFloatTensor(),
        )
        self.features = dataset.column_list[:]

        self.train_data, self.val_data, self.test_data = random_split(
            dataset,
            self.split_ratio,
        )
