"""Data utility classes and helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numpy.typing import NDArray

if TYPE_CHECKING:
    from torch import Tensor


NDArrayType = NDArray[np.float32 | np.uint64]
CollateFnType = Callable[[list[Any]], Any]


def sliding_subarrays(arr: NDArrayType, n: int, m: int) -> np.ndarray:
    """Return all subarrays of size n, shifted by m."""
    shape = ((len(arr) - n) // m + 1, n)
    strides = (arr.strides[0] * m, arr.strides[0])
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def collate_sequence_chain(batch: list[Any]) -> list[Any]:
    """Collate the ACTS chain dataset."""
    # get max length of the sequences
    max_len = max([len(item) for item in batch])

    output: list[Any] = []
    for item in batch:
        # append zeros to the end of the sequence if needed
        # Note: here zeros are OK as this is the assummed padding token representation
        out_item = (
            np.pad(item, (0, max_len - len(item))) if len(item) < max_len else item
        )[:-1]
        # shift the prediction for one to the left and append zeros to the end
        out_item_shifted = np.pad(item[1:], (0, max_len - len(item) + 1))[:-1]
        # append to the output
        output.append((out_item, out_item_shifted))

    # now with proper padding run the default collate function
    return torch.utils.data.default_collate(output)  # type: ignore[no-any-return]


def collate_sequence(batch: list[Any]) -> list[Any]:
    """Collate the ACTS dataset."""
    # get sequence feature length
    feature_len = len(batch[0][0]) if batch[0].size > 0 else 0
    # get max length of the sequences
    max_len = max([len(item[0] if item[0] is not None else item[1]) for item in batch])

    output: list[Any] = []
    for item in batch:
        # append zeros to the end of the sequence if needed
        out_item = (
            (
                np.vstack(
                    [
                        item,
                        np.zeros(
                            (max_len - len(item), feature_len),
                            dtype=np.int64,
                        ),
                    ],
                )
                if len(item) < max_len
                else item
            )
            if item is not None
            else None
        )
        # shift the prediction for one to the left and append zeros to the end
        out_item_shifted = (
            np.vstack(
                [
                    item[1:],
                    np.zeros(
                        (max_len - len(item) + 1, feature_len),
                        dtype=np.int64,
                    ),
                ],
            )
            if item is not None
            else None
        )
        # append to the output
        output.append((out_item, out_item_shifted))

    # now with proper padding run the default collate function
    return torch.utils.data.default_collate(output)  # type: ignore[no-any-return]


class NDArrayTransformation(ABC):
    """numpy ndarray transformation base class."""

    @abstractmethod
    def __call__(self, sample: NDArrayType) -> NDArrayType:
        """Transform the sample."""
        raise NotImplementedError

    @abstractmethod
    def inverse(self, sample: NDArrayType) -> NDArrayType:
        """Inverse the transformation."""
        raise NotImplementedError


class TensorTransformation(ABC):
    """numpy to tensor transformation base class."""

    @abstractmethod
    def __call__(
        self,
        sample: tuple[NDArrayType, NDArrayType | None],
    ) -> tuple[Tensor, Tensor | None]:
        """Transform the sample."""
        raise NotImplementedError


class NDArrayToFloatTensor(TensorTransformation):
    """Convert ndarrays in sample to float Tensors."""

    def __call__(
        self,
        sample: tuple[NDArrayType, NDArrayType | None],
    ) -> tuple[Tensor, Tensor | None]:
        """Transform the sample to tensors."""
        features, labels = sample
        return (
            torch.from_numpy(features).float(),
            torch.from_numpy(labels).float() if labels is not None else None,
        )


class NDArrayToLongTensor(TensorTransformation):
    """Convert ndarrays in sample to long Tensors."""

    def __call__(
        self,
        sample: tuple[NDArrayType, NDArrayType | None],
    ) -> tuple[Tensor, Tensor | None]:
        """Transform the sample to tensors."""
        features, labels = sample
        return (
            torch.from_numpy(features).long(),
            torch.from_numpy(labels).long() if labels is not None else None,
        )
