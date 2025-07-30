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
        full_data: bool = False,
    ) -> None:
        """Load the ActsHits as a dataset."""
        self.transforms = transforms

        with input_file.open("rb") as f:
            data, data_chunked = pickle.load(f)
            self.data = (
                data_chunked if not full_data and data_chunked is not None else data
            )

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
        transforms: list[NDArrayTransformation] | None = None,
    ) -> None:
        """Load the ActsHits as a dataset."""
        self.transforms = transforms

        self.data: list[NDArrayType]

        with input_file.open("rb") as f:
            self.data = pickle.load(f)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> NDArrayType:
        """Return the item at the given index."""
        sequence: NDArrayType = (
            self.data[idx] if self.data else np.empty(0, dtype=np.float32)
        )

        if self.transforms:
            for t in self.transforms:
                sequence = t(sequence)

        return sequence
