"""Data utility classes and helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numpy.typing import NDArray

if TYPE_CHECKING:
    from torch import Tensor

Word = Any

NDArrayType = NDArray[np.float32 | np.uint64]
CollateFnType = Callable[[list[Any]], Any]


def collate_sequence(batch: list[Any]) -> list[Any]:
    """Collate the ACTS dataset."""
    # get sequence feature length
    feature_len = len(batch[0][0])
    # get max length of the sequences
    max_len = max([len(item) for item in batch])

    output: list[Any] = []
    for item in batch:
        # append zeros to the end of the sequence if needed
        out_item = (
            np.vstack(
                [
                    item,
                    np.zeros((max_len - len(item), feature_len), dtype=np.int64),
                ],
            )
            if len(item) < max_len
            else item
        )
        # shift the prediction for one to the left and append zeros to the end
        out_item_shifted = np.vstack(
            [
                item[1:],
                np.zeros(
                    (max_len - len(item) + 1, feature_len),
                    dtype=np.int64,
                ),
            ],
        )
        # append to the output
        output.append((out_item, out_item_shifted))

    # now with proper padding run the default collate function
    return torch.utils.data.default_collate(output)  # type: ignore


class NDArrayTransformation(ABC):
    """numpy ndarray transformation base class."""

    @abstractmethod
    def __call__(
        self,
        sample: tuple[NDArrayType, NDArrayType | None],
    ) -> tuple[NDArrayType, NDArrayType | None]:
        """Transform the sample."""
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


class Tokenize(NDArrayTransformation):
    """Tokenize the input data."""

    def __init__(self, dictionary: DataDictionary, index: int) -> None:
        """Initialize the tokenizer."""
        self.dictionary = dictionary
        self.index = index

    def __call__(
        self,
        sample: tuple[NDArrayType, NDArrayType | None],
    ) -> tuple[NDArrayType, NDArrayType | None]:
        """Transform the sample to tensors."""
        features, labels = sample
        helper = np.vectorize(self.dictionary.add_word)
        features[:, self.index] = helper(features[:, self.index])
        if labels is not None:
            labels[:, self.index] = helper(labels[:, self.index])
        return (features, labels)

    def inverse(
        self,
        sample: tuple[NDArrayType, NDArrayType | None],
    ) -> tuple[NDArrayType, NDArrayType | None]:
        """Inverse the tokenization."""
        features, labels = sample
        helper = np.vectorize(self.dictionary.get_word)
        features[:, self.index] = helper(features[:, self.index])
        if labels is not None:
            labels[:, self.index] = helper(labels[:, self.index])
        return (features, labels)


class DataDictionary:
    """Tokenized data dictionary."""

    def __init__(self, name: str, data_type: Type | None = None) -> None:
        """Initialize tokenized data dictionary."""
        self.name: str = name
        self.data_type: DataDictionary.Type = (
            data_type if data_type else DataDictionary.Type.Number
        )

        self.word2idx: dict[Word, int] = {}
        self.idx2word: list[Word] = []

        # add padding token
        if self.data_type == DataDictionary.Type.Text:
            self.add_word("<PAD>")
        else:
            self.add_word(0)

    def add_word(self, word: Word) -> int:
        """Add a word to the dictionary."""
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def get_word(self, index: int) -> Word:
        """Get the word from the index."""
        return self.idx2word[index]

    def __len__(self) -> int:
        """Return the size of the dictionary."""
        return len(self.idx2word)

    class Type(Enum):
        """Data type enumeration."""

        Number = "number"
        Text = "text"
