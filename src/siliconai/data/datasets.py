"""Custom datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from pathlib import Path

    from siliconai.data.utils import (
        NDArrayTransformation,
        NDArrayType,
        TensorTransformation,
    )


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
        self.columns = np.array(self.data[self.column_list].tolist())
        self.labels = np.array(self.data[self.label_list].tolist())

        self.transforms = transforms
        self.tensor_transform = tensor_transform

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(
        self,
        idx: Tensor | list[int],
    ) -> tuple[Tensor, Tensor]:
        """Return the item at the given index."""
        if torch.is_tensor(idx):  # type: ignore
            idx = idx.tolist()  # type: ignore

        columns: NDArrayType = self.columns[idx]
        labels: NDArrayType = self.labels[idx]

        if self.transforms:
            for t in self.transforms:
                columns, labels = t((columns, labels))

        return self.tensor_transform((columns, labels))


class TestSequenceDataset(Dataset):  # type: ignore
    """Sequence test dataset."""

    def __init__(
        self,
        input_file: Path,
        tensor_transform: TensorTransformation,
        transforms: list[NDArrayTransformation] | None = None,
    ) -> None:
        """Load the test sequence as a dataset."""
        self.data = np.load(input_file)
        self.transforms = transforms
        self.tensor_transform = tensor_transform

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(
        self,
        idx: Tensor | int,
    ) -> tuple[Tensor, Tensor]:
        """Return the item at the given index."""
        if torch.is_tensor(idx):  # type: ignore
            idx = idx.tolist()  # type: ignore

        y = self.data[idx]
        y_input: NDArrayType = y[0]
        y_expected: NDArrayType = y[1]

        if self.transforms:
            for t in self.transforms:
                y_input, y_expected = t((y_input, y_expected))

        return self.tensor_transform((y_input, y_expected))
