"""Custom datasets."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

    from siliconai.data.utils import (
        NDArrayTransformation,
        NDArrayType,
        TensorTransformation,
    )


class ActsHitsDataset(Dataset):  # type: ignore
    """ActsHits dataset."""

    def __init__(
        self,
        input_file: Path,
        transforms_int: list[NDArrayTransformation] | None = None,
        transforms_float: list[NDArrayTransformation] | None = None,
    ) -> None:
        """Load the ActsHits as a dataset."""
        self.transforms_int = transforms_int
        self.transforms_float = transforms_float

        with input_file.open("rb") as f:
            self.data_int, self.data_float = pickle.load(f)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data_int) if self.data_int else len(self.data_float)

    def __getitem__(self, idx: int) -> tuple[NDArrayType, NDArrayType]:
        """Return the item at the given index."""
        sequence_int: NDArrayType = self.data_int[idx] if self.data_int else None
        sequence_float: NDArrayType = self.data_float[idx] if self.data_float else None

        if self.transforms_int:
            for t in self.transforms_int:
                sequence_int, _ = t((sequence_int, None))
        if self.transforms_float:
            for t in self.transforms_float:
                sequence_float, _ = t((sequence_float, None))

        return sequence_int, sequence_float


class TRKNtupleDataset(Dataset):  # type: ignore
    """TRKNtuple dataset."""

    def __init__(
        self,
        input_file: Path,
        tensor_transform: TensorTransformation,
        transforms: list[NDArrayTransformation] | None = None,
    ) -> None:
        """Load the processed TRKNtuple as a dataset."""
        self.column_list = [
            "track_d0",
            "track_z0",
            "track_phi",
            "track_theta",
            "track_qOverP",
        ]
        self.label_list = ["truth_pt", "truth_eta", "truth_phi", "truth_charge"]

        self.data = np.load(input_file)
        self.features = np.array(self.data[self.column_list].tolist())
        self.labels = np.array(self.data[self.label_list].tolist())

        self.transforms = transforms
        self.tensor_transform = tensor_transform

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor | None]:
        """Return the item at the given index."""
        features: NDArrayType = self.features[idx]
        labels: NDArrayType | None = self.labels[idx]

        if self.transforms:
            for t in self.transforms:
                features, labels = t((features, labels))

        return self.tensor_transform((features, labels))


class TestSequenceDataset(Dataset):  # type: ignore
    """Sequence test dataset."""

    def __init__(
        self,
        input_file: Path,
        transforms: list[NDArrayTransformation] | None = None,
    ) -> None:
        """Load the test sequence as a dataset."""
        self.data = np.load(input_file)
        self.transforms = transforms

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[NDArrayType, NDArrayType | None]:
        """Return the item at the given index."""
        y = self.data[idx]
        y_input: NDArrayType = y[0]
        y_expected: NDArrayType | None = y[1]

        if self.transforms:
            for t in self.transforms:
                y_input, y_expected = t((y_input, y_expected))

        return (y_input, y_expected)
