"""Custom datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
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


class ActsHitsDataset(Dataset):  # type: ignore
    """ActsHits dataset."""

    def __init__(
        self,
        input_file: Path,
        transforms: list[NDArrayTransformation] | None = None,
    ) -> None:
        """Load the ActsHits as a dataset."""
        # TODO: make configurable
        self.column_list = [
            "geometry_id",
            "particle_type",
        ]

        self.transforms = transforms
        with pd.HDFStore(input_file, mode="r") as store:
            self.data_frame = store["hits"][self.column_list].astype("int64")

    def __len__(self) -> int:
        """Return the length of the dataset."""
        if not isinstance(self.data_frame.index, pd.MultiIndex):
            error = "Index must be a MultiIndex"
            raise TypeError(error)
        return int(self.data_frame.index.levshape[0])

    def __getitem__(
        self,
        idx: int | list[int] | Tensor,
    ) -> NDArrayType:
        """Return the item at the given index."""
        if torch.is_tensor(idx):  # type: ignore
            idx = idx.tolist()  # type: ignore
        elif isinstance(idx, int):
            idx = [idx]

        id_list: list[int] = idx  # type: ignore

        sequence = self.data_frame.loc[id_list].to_numpy()

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

    def __getitem__(
        self,
        idx: Tensor | list[int],
    ) -> tuple[Tensor, Tensor | None]:
        """Return the item at the given index."""
        if torch.is_tensor(idx):  # type: ignore
            idx = idx.tolist()  # type: ignore

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

    def __getitem__(
        self,
        idx: Tensor | int,
    ) -> tuple[NDArrayType, NDArrayType | None]:
        """Return the item at the given index."""
        if torch.is_tensor(idx):  # type: ignore
            idx = idx.tolist()  # type: ignore

        y = self.data[idx]
        y_input: NDArrayType = y[0]
        y_expected: NDArrayType | None = y[1]

        if self.transforms:
            for t in self.transforms:
                y_input, y_expected = t((y_input, y_expected))

        return (y_input, y_expected)
