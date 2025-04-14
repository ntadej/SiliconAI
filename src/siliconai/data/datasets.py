"""Custom datasets."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from pathlib import Path

    from siliconai.data.utils import (
        NDArrayTransformation,
        NDArrayType,
    )


class ActsChainDataset(Dataset):  # type: ignore[type-arg]
    """ActsChain dataset."""

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
                sequence = t(sequence)

        return sequence


class ActsHitsDataset(Dataset):  # type: ignore[type-arg]
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

        self.data_int: list[NDArrayType]
        self.data_float: list[NDArrayType]

        with input_file.open("rb") as f:
            self.data_int, self.data_float = pickle.load(f)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data_int) if self.data_int else len(self.data_float)

    def __getitem__(self, idx: int) -> tuple[NDArrayType, NDArrayType]:
        """Return the item at the given index."""
        sequence_int: NDArrayType = (
            self.data_int[idx] if self.data_int else np.empty(0, dtype=np.float32)
        )
        sequence_float: NDArrayType = (
            self.data_float[idx] if self.data_float else np.empty(0, dtype=np.float32)
        )

        if self.transforms_int:
            for t in self.transforms_int:
                sequence_int = t(sequence_int)
        if self.transforms_float:
            for t in self.transforms_float:
                sequence_float = t(sequence_float)

        return sequence_int, sequence_float
