"""Data utility classes and helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from torch import Tensor

Word = Any


class Transformation(ABC):
    """Transformation base class."""

    @abstractmethod
    def __call__(
        self,
        sample: tuple[ArrayLike, ArrayLike],
    ) -> tuple[ArrayLike | Tensor, ArrayLike | Tensor]:
        """Transform the sample."""
        raise NotImplementedError


class NdarrayToFloatTensor(Transformation):
    """Convert ndarrays in sample to float Tensors."""

    def __call__(
        self,
        sample: tuple[ArrayLike, ArrayLike],
    ) -> tuple[Tensor, Tensor]:
        """Transform the sample to tensors."""
        columns, labels = sample
        return (torch.from_numpy(columns).float(), torch.from_numpy(labels).float())


class NdarrayToLongTensor(Transformation):
    """Convert ndarrays in sample to long Tensors."""

    def __call__(
        self,
        sample: tuple[ArrayLike, ArrayLike],
    ) -> tuple[Tensor, Tensor]:
        """Transform the sample to tensors."""
        columns, labels = sample
        return (torch.from_numpy(columns).long(), torch.from_numpy(labels).long())


class Tokenize(Transformation):
    """Tokenize the input data."""

    def __init__(self, dictionary: DataDictionary) -> None:
        """Initialize the tokenizer."""
        self.dictionary = dictionary

    def __call__(
        self,
        sample: tuple[ArrayLike, ArrayLike],
    ) -> tuple[Tensor, Tensor]:
        """Transform the sample to tensors."""
        columns, labels = sample
        helper = np.vectorize(self.dictionary.add_word)
        return (helper(columns), helper(labels))


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

    def __len__(self) -> int:
        """Return the size of the dictionary."""
        return len(self.idx2word)

    class Type(Enum):
        """Data type enumeration."""

        Number = "number"
        Text = "text"
