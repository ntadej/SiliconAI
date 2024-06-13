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
        transforms: list[NDArrayTransformation] | None = None,
    ) -> None:
        """Load the ActsHits as a dataset."""
        self.transforms = transforms

        with input_file.open("rb") as f:
            self.data = pickle.load(f)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> NDArrayType:
        """Return the item at the given index."""
        sequence: NDArrayType = self.data[idx]

        if self.transforms:
            for t in self.transforms:
                sequence, _ = t((sequence, None))

        return sequence


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
