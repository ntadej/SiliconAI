"""Data modules."""
from typing import Any

import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST  # type: ignore
from torchvision.transforms import ToTensor  # type: ignore

from siliconai.cli.config import Configuration


class MNISTDataModule(L.LightningDataModule):
    """MNIST data module."""

    def __init__(self, config: Configuration) -> None:
        """Initialize the data module."""
        super().__init__()
        self.data_path = config.global_config.data_path / "common_datasets"
        self.batch_size = config.data.batch_size
        self.workers = 4

    def prepare_data(self) -> None:
        """Download and prepare the MNIST dataset."""
        MNIST(self.data_path, train=True, download=True)
        MNIST(self.data_path, train=False, download=True)

    def setup(self, stage: str) -> None:  # noqa: ARG002
        """Transform and setup the MNIST dataset."""
        self.train_data = MNIST(self.data_path, train=True, transform=ToTensor())
        self.val_data, self.test_data = random_split(
            MNIST(self.data_path, train=False, transform=ToTensor()),
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
