"""Data modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST, FashionMNIST  # type: ignore
from torchvision.transforms import ToTensor  # type: ignore

from siliconai.data.datasets import TestSequenceDataset, TRKNtupleDataset
from siliconai.data.utils import (
    DataDictionary,
    NDArrayToFloatTensor,
    NDArrayToLongTensor,
    Tokenize,
)

if TYPE_CHECKING:
    from siliconai.cli.config import Configuration


class MNISTDataModule(L.LightningDataModule):
    """MNIST data module."""

    def __init__(self, config: Configuration) -> None:
        """Initialize the data module."""
        super().__init__()
        self.data_path = config.global_config.data_path / "common_datasets"
        self.batch_size = config.data.batch_size
        self.workers = 4
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

    def train_dataloader(self) -> DataLoader[Any]:
        """Return the training DataLoader."""
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.workers,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return the validation DataLoader."""
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.workers,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return the test DataLoader."""
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.workers,
        )

    def state_dict(self) -> dict[str, Any]:
        """Track the data module state."""
        return {}

    def load_state_dict(self, _state_dict: dict[str, Any]) -> None:
        """Restore the state based on what is tracked."""


class FashionMNISTDataModule(MNISTDataModule):
    """FashionMNIST data module."""

    def __init__(self, config: Configuration) -> None:
        """Initialize the data module."""
        super().__init__(config)
        self.fashion = True


class TRKNtupleDataModule(L.LightningDataModule):
    """TRKNtuple data module."""

    def __init__(self, config: Configuration) -> None:
        """Initialize the data module."""
        super().__init__()
        self.data_path = config.data.conversion_output_file
        self.batch_size = config.data.batch_size
        self.workers = 4

        self.save_hyperparameters(ignore=["config"])

    def prepare_data(self) -> None:
        """Prepare TRKNtuple dataset."""

    def setup(self, stage: str) -> None:  # noqa: ARG002
        """Transform and setup the TRKNtuple dataset."""
        if self.data_path is None:
            error = "TRKNtuple data path not set."
            raise ValueError(error)

        dataset = TRKNtupleDataset(
            self.data_path,
            tensor_transform=NDArrayToFloatTensor(),
        )
        self.columns = dataset.column_list[:]

        self.train_data, self.val_data, self.test_data = random_split(
            dataset,
            [0.7, 0.15, 0.15],
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return the training DataLoader."""
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.workers,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return the validation DataLoader."""
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.workers,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return the test DataLoader."""
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.workers,
        )

    def state_dict(self) -> dict[str, Any]:
        """Track the data module state."""
        return {}

    def load_state_dict(self, _state_dict: dict[str, Any]) -> None:
        """Restore the state based on what is tracked."""


class TestSequenceDataModule(L.LightningDataModule):
    """Sequence test data module."""

    def __init__(self, data_config: Configuration) -> None:
        """Initialize the data module."""
        super().__init__()
        self.data_path = data_config.global_config.data_path / "test_sequence.npy"
        self.batch_size = data_config.data.batch_size
        self.workers = 4

        self.input_dim_discreet: list[int]
        if isinstance(data_config.data.input_dim, int):
            self.input_dim_discreet = [data_config.data.input_dim]
        else:
            self.input_dim_discreet = data_config.data.input_dim

        self.tokenize = [
            Tokenize(DataDictionary(f"dict{i}"), i)
            for i in range(len(self.input_dim_discreet))
        ]

        self.save_hyperparameters()

    def tokenize_data(self) -> None:
        """Tokenize the sequence test dataset."""
        dataset = TestSequenceDataset(
            self.data_path,
            transforms=[*self.tokenize],
            tensor_transform=NDArrayToLongTensor(),
        )
        for i in range(len(dataset)):
            dataset[i]

        for tokenize in self.tokenize:
            assert len(tokenize.dictionary) > 1

    def setup(self, stage: str) -> None:  # noqa: ARG002
        """Transform and setup the sequence test dataset."""
        dataset = TestSequenceDataset(
            self.data_path,
            transforms=[*self.tokenize],
            tensor_transform=NDArrayToLongTensor(),
        )

        self.train_data, self.val_data, self.test_data = random_split(
            dataset,
            [0.7, 0.15, 0.15],
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return the training DataLoader."""
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.workers,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return the validation DataLoader."""
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.workers,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return the test DataLoader."""
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.workers,
        )

    def state_dict(self) -> dict[str, Any]:
        """Track the data module state."""
        return {}

    def load_state_dict(self, _state_dict: dict[str, Any]) -> None:
        """Restore the state based on what is tracked."""
