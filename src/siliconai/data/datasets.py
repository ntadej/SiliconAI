"""Custom datasets."""
from pathlib import Path

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor
from torch.utils.data import Dataset


class TRKNtupleToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(
        self,
        sample: tuple[ArrayLike, ArrayLike],
    ) -> tuple[Tensor, Tensor]:
        """Transform the sample to tensors."""
        columns, labels = sample
        return (torch.from_numpy(columns).float(), torch.from_numpy(labels).float())


class TRKNtupleDataset(Dataset):  # type: ignore
    """TRKNtuple dataset."""

    def __init__(self, input_file: Path, transforms: list[TRKNtupleToTensor]) -> None:
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

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(
        self,
        idx: Tensor | list[int],
    ) -> tuple[ArrayLike, ArrayLike] | tuple[Tensor, Tensor]:
        """Return the item at the given index."""
        if torch.is_tensor(idx):  # type: ignore
            idx = idx.tolist()  # type: ignore

        columns: ArrayLike = self.columns[idx]
        labels: ArrayLike = self.labels[idx]

        if self.transforms:
            for t in self.transforms:
                columns, labels = t((columns, labels))

        return (columns, labels)
