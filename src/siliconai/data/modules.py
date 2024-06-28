"""Data modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import lightning as L
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import MNIST, FashionMNIST  # type: ignore
from torchvision.transforms import ToTensor  # type: ignore

from siliconai.common.enums import DataLoadingType
from siliconai.data.datasets import (
    ActsHitsDataset,
    TestSequenceDataset,
    TRKNtupleDataset,
)
from siliconai.data.utils import (
    CollateFnType,
    DataDictionary,
    NDArrayToFloatTensor,
    NDArrayType,
    Tokenize,
    collate_sequence,
)

if TYPE_CHECKING:
    from pathlib import Path

    from siliconai.cli.config import Configuration


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

    def get_dataloader(self, data_type: DataLoadingType) -> DataLoader[Any]:
        """Get the DataLoader for the specified data type."""
        if data_type == DataLoadingType.fit:
            return self.train_dataloader()
        if data_type == DataLoadingType.validate:
            return self.val_dataloader()
        if data_type == DataLoadingType.test:  # noqa: RET503
            return self.test_dataloader()

    def train_dataloader(self) -> DataLoader[Any]:
        """Return the training DataLoader."""
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return the validation DataLoader."""
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return the test DataLoader."""
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
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


class FashionMNISTDataModule(MNISTDataModule):
    """FashionMNIST data module."""

    def __init__(self, config: Configuration) -> None:
        """Initialize the data module."""
        super().__init__(config)
        self.fashion = True


class ActsDataModule(BaseDataModule):
    """ACTS-based silicon detector hits data module."""

    def __init__(self, data_config: Configuration) -> None:
        """Initialize the data module."""
        super().__init__(data_config, collate_sequence)
        if not data_config.data.input_file:
            error = "ACTS data path not set."
            raise ValueError(error)

        self.data_path: Path = data_config.data.input_file

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
        """Tokenize the ACTS dataset manually."""
        dataset = ActsHitsDataset(
            self.data_path,
            transforms=self.tokenize,  # type: ignore
        )
        for i in range(len(dataset)):
            dataset[i]

        for i, tokenize in enumerate(self.tokenize):
            assert len(tokenize.dictionary) <= self.input_dim_discreet[i]
            # TODO: add summary printing

    def translate_data(self, data: NDArrayType) -> NDArrayType:
        """Translate back from tokens to the original data."""
        for tokenize in self.tokenize:
            data, _ = tokenize.inverse((data, None))
        return data

    def prepare_data(self) -> None:
        """Prepare and tokenise the ACTS dataset."""
        self.tokenize_data()

    def setup(self, stage: str) -> None:  # noqa: ARG002
        """Transform and setup the ACTS dataset."""
        dataset = ActsHitsDataset(
            self.data_path,
            transforms=self.tokenize,  # type: ignore
        )

        self.train_data, self.val_data, self.test_data = random_split(
            dataset,
            self.split_ratio,
        )

    def state_dict(self) -> dict[str, Any]:
        """Track the data module state."""
        word2idx = {i: t.dictionary.word2idx for i, t in enumerate(self.tokenize)}
        idx2word = {i: t.dictionary.idx2word for i, t in enumerate(self.tokenize)}
        return {"word2idx": word2idx, "idx2word": idx2word}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the state based on what is tracked."""
        for i, tokenize in enumerate(self.tokenize):
            tokenize.dictionary.word2idx = state_dict["word2idx"][i]
            tokenize.dictionary.idx2word = state_dict["idx2word"][i]


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


class TestSequenceDataModule(BaseDataModule):
    """Sequence test data module."""

    def __init__(self, data_config: Configuration) -> None:
        """Initialize the data module."""
        super().__init__(data_config)
        self.data_path = data_config.global_config.data_path / "test_sequence.npy"

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
        )

        self.train_data, self.val_data, self.test_data = random_split(
            dataset,
            self.split_ratio,
        )
