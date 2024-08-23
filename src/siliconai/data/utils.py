"""Data utility classes and helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numpy.typing import NDArray

if TYPE_CHECKING:
    from sklearn import BaseEstimator  # type: ignore
    from torch import Tensor

    from siliconai.cli.logging import Logger


NDArrayType = NDArray[np.float32 | np.uint64]
CollateFnType = Callable[[list[Any]], Any]


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
        )
        # shift the prediction for one to the left and append zeros to the end
        out_item_shifted = np.pad(item[1:], (0, max_len - len(item) + 1))
        # append to the output
        output.append((out_item, out_item_shifted))

    # now with proper padding run the default collate function
    return torch.utils.data.default_collate(output)  # type: ignore


def collate_sequence(batch: list[Any]) -> list[Any]:
    """Collate the ACTS dataset."""
    # get sequence feature length
    feature_len_int = len(batch[0][0][0]) if batch[0][0] is not None else 0
    feature_len_float = len(batch[0][1][0]) if batch[0][1] is not None else 0
    # get max length of the sequences
    max_len = max([len(item[0] if item[0] is not None else item[1]) for item in batch])

    output: list[Any] = []
    for item in batch:
        item_int, item_float = item
        # append zeros to the end of the sequence if needed
        out_item_int = (
            (
                np.vstack(
                    [
                        item_int,
                        np.zeros(
                            (max_len - len(item_int), feature_len_int),
                            dtype=np.int64,
                        ),
                    ],
                )
                if len(item_int) < max_len
                else item_int
            )
            if item_int is not None
            else None
        )
        out_item_float = (
            (
                np.vstack(
                    [
                        item_float,
                        np.zeros(
                            (max_len - len(item_float), feature_len_float),
                            dtype=np.float32,
                        ),
                    ],
                )
                if len(item_float) < max_len
                else item_float
            )
            if item_float is not None
            else None
        )
        # shift the prediction for one to the left and append zeros to the end
        out_item_int_shifted = (
            np.vstack(
                [
                    item_int[1:],
                    np.zeros(
                        (max_len - len(item_int) + 1, feature_len_int),
                        dtype=np.int64,
                    ),
                ],
            )
            if item_int is not None
            else None
        )
        out_item_float_shifted = (
            np.vstack(
                [
                    item_float[1:],
                    np.zeros(
                        (max_len - len(item_float) + 1, feature_len_float),
                        dtype=np.float32,
                    ),
                ],
            )
            if item_float is not None
            else None
        )
        # append to the output
        if out_item_int is not None and out_item_float is not None:
            output.append(
                (
                    out_item_int,
                    out_item_float,
                    out_item_int_shifted,
                    out_item_float_shifted,
                ),
            )
        elif out_item_int is not None:
            output.append((out_item_int, out_item_int_shifted))

    # now with proper padding run the default collate function
    return torch.utils.data.default_collate(output)  # type: ignore


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


class ScikitLearnTransformation(NDArrayTransformation):
    """Normalize the input data to a gaussian function."""

    def __init__(self, name: str, index: int, transformation: BaseEstimator) -> None:
        """Initialize the tokenizer."""
        self.name = name
        self.transformation = transformation()
        self.index = index

    def summary(self, logger: Logger) -> None:
        """Log summary of the transformation."""
        logger.info(
            'Scale for "%s": %f',
            self.name,
            self.transformation.scale_[0],
        )

    def fit(self, data: tuple[NDArrayType, NDArrayType | None]) -> None:
        """Fit the transformation."""
        self.transformation.fit(data[0][:, self.index].reshape(-1, 1))

    def __call__(self, sample: NDArrayType) -> NDArrayType:
        """Transform the sample to tensors."""
        sample[:, self.index] = self.transformation.transform(
            sample[:, self.index].reshape(-1, 1),
        ).reshape(1, -1)
        return sample

    def inverse(self, sample: NDArrayType) -> NDArrayType:
        """Inverse the tokenization."""
        sample[:, self.index] = self.transformation.inverse_transform(
            sample[:, self.index].reshape(-1, 1),
        ).reshape(1, -1)
        return sample
