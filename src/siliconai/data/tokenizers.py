# Copyright (C) 2024 Tadej Novak
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# SPDX-License-Identifier: MPL-2.0

"""Data tokenizers and helpers."""

from __future__ import annotations

from enum import Enum
from json import dump, load
from typing import TYPE_CHECKING, Any

import numpy as np

from siliconai.cli.logger import progress_bar
from siliconai.common.enums import ColumnType
from siliconai.data.datasets import ActsChainDataset
from siliconai.data.utils import NDArrayTransformation, NDArrayType

if TYPE_CHECKING:
    from siliconai.cli.config import DataConfiguration
    from siliconai.cli.logger import Logger

Word = Any


class DataDictionary:
    """Tokenized data dictionary."""

    def __init__(
        self,
        name: str,
        padding_token: int | str | None = None,
        data_type: Type | None = None,
    ) -> None:
        """Initialize tokenized data dictionary."""
        self.name: str = name
        self.data_type: DataDictionary.Type = (
            data_type if data_type else DataDictionary.Type.Number
        )

        self.word2idx: dict[Word, int] = {}
        self.idx2word: list[Word] = []

        # add padding token
        if padding_token is not None:
            self.add_word(padding_token)

        # add vectorized versions of methods
        self.add_word_v = np.vectorize(self.add_word)
        self.get_word_v = np.vectorize(self.get_word)

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

    @staticmethod
    def json_decode(dct: dict[str, Any]) -> dict[str, Any]:
        """Decode data represented as a JSON."""
        if "idx2word" in dct:
            dct["idx2word"] = [np.int64(i) for i in dct["idx2word"]]
        if "word2idx" in dct:
            dct["word2idx"] = {np.int64(i): v for i, v in dct["word2idx"].items()}
        return dct

    def to_dict(self) -> dict[str, dict[Word, int] | list[Word]]:
        """Return a JSON/dictionary representation."""
        return {
            "word2idx": {int(k): v for k, v in self.word2idx.items()},
            "idx2word": [int(i) for i in self.idx2word],
        }

    def from_dict(self, dct: dict[str, Any]) -> None:
        """Load data from a JSON/dictionary representation."""
        self.word2idx = dct["word2idx"]
        self.idx2word = dct["idx2word"]

    class Type(Enum):
        """Data type enumeration."""

        Number = "number"
        Text = "text"


class SequenceTokenizer(NDArrayTransformation):
    """Tokenize the sequence input data."""

    def __init__(self) -> None:
        """Initialize the tokenizer."""
        self.dictionary = DataDictionary("dictionary")
        self.summary_dict: dict[str, dict[str, int]] = {}
        self.masks: dict[int, list[bool]] = {}

    def __call__(self, sample: NDArrayType) -> NDArrayType:
        """Transform the sample to tensors."""
        output: NDArrayType = self.dictionary.add_word_v(sample)
        return output

    def inverse(self, sample: NDArrayType) -> NDArrayType:
        """Inverse the tokenization."""
        output: NDArrayType = self.dictionary.get_word_v(sample)
        return output

    def summary(self, logger: Logger) -> None:
        """Print summary."""
        logger.info("Total dictionary size: %d words", len(self.dictionary))

        for column, summary in self.summary_dict.items():
            logger.info(
                "Category %s: %d tokens (%d-%d)",
                column,
                summary["tokens"],
                summary["start"],
                summary["end"],
            )

    @staticmethod
    def json_decode(dct: dict[str, Any]) -> dict[str, Any]:
        """Decode the tokenizer represented as a JSON."""
        if "masks" in dct:
            dct["masks"] = {int(k): v for k, v in dct["masks"].items()}
        return DataDictionary.json_decode(dct)

    @staticmethod
    def load(
        config: DataConfiguration,
        logger: Logger | None = None,
    ) -> SequenceTokenizer:
        """Load the tokenizer from JSON."""
        if not config.input_path:
            error = "Invalid configuration"
            raise RuntimeError(error)

        tokenizer_file = config.input_path / f"tokenizer_{config.input_suffix}.json"
        if logger:
            logger.info('Loading the tokenizer from "%s"', tokenizer_file)
        with tokenizer_file.open("r") as f:
            data = load(f, object_hook=SequenceTokenizer.json_decode)

        dictionary = DataDictionary("dictionary", config.padding_token)
        dictionary.from_dict(data["dictionary"])
        tokenizer = SequenceTokenizer()
        tokenizer.dictionary = dictionary
        tokenizer.summary_dict = data["summary"]
        tokenizer.masks = data["masks"]

        return tokenizer

    @staticmethod
    def train(config: DataConfiguration, logger: Logger) -> None:  # noqa: C901, PLR0915
        """Train the tokenizer."""
        if not config.input_path:
            return

        tokenizer = SequenceTokenizer()
        tokenizer.dictionary = DataDictionary("dictionary", config.padding_token)

        ncolumns = len(config.columns)
        if config.index_with_offset >= 0:
            ncolumns += 1
        if config.split_numerical:
            ncolumns += len(
                [c for c in config.columns_type if c == ColumnType.Numerical],
            )

        logger.info(
            "Tokenizing the input file with %d columns and %d parts",
            len(config.columns),
            ncolumns,
        )

        total = 1

        summary_dict = {}

        dataset = ActsChainDataset(
            config.input_path,
            config.input_suffix,
            logger=logger,
        )
        s = 0
        token_sets: dict[int, set[int]] = {c: set() for c in range(ncolumns)}
        for c, (column, column_type) in enumerate(
            zip(
                (["index"] if config.index_with_offset >= 0 else []) + config.columns,
                config.columns_type,
                strict=True,
            ),
        ):
            logger.info("Tokenizing column %d: %s", c, column)
            with progress_bar(transient=True) as progress:
                task = progress.add_task("Processing", total=len(dataset))
                for i in range(len(dataset)):
                    row = dataset[i]
                    token_sets[s] |= set(tokenizer(row[s::ncolumns]))
                    if config.split_numerical and column_type == ColumnType.Numerical:
                        token_sets[s + 1] |= set(tokenizer(row[s + 1 :: ncolumns]))
                    progress.update(task, advance=1)
                s += (
                    2
                    if config.split_numerical and column_type == ColumnType.Numerical
                    else 1
                )

            if column_type is ColumnType.Categorical:
                column_tokens = len(tokenizer.dictionary) - total
                logger.info("Category %s: %d", column, column_tokens)
                summary_dict[column] = {
                    "tokens": column_tokens,
                    "start": total,
                    "end": total + column_tokens - 1,
                }
                total += column_tokens

        column_tokens = len(tokenizer.dictionary) - total
        logger.info("Numerical tokens: %d", column_tokens)
        summary_dict["numerical"] = {
            "tokens": column_tokens,
            "start": total,
            "end": total + column_tokens - 1,
        }
        tokenizer.summary_dict = summary_dict

        dim = sum(config.input_dim) + 1  # with padding tokens

        logger.info("Expected dictionary size: %d words", dim)
        tokenizer.summary(logger)

        if len(tokenizer.dictionary) != dim:
            error = (
                f"Dictionary sizes do not match ({len(tokenizer.dictionary)} vs {dim})"
            )
            raise ValueError(error)

        logger.info("Building column token masks")
        masks: dict[int, NDArrayType] = {
            c: np.zeros(dim, dtype=bool) for c in range(ncolumns)
        }
        for c in range(ncolumns):
            masks[c][0] = True  # padding token
            masks[c][list(token_sets[c])] = True
        for c in range(ncolumns):
            logger.info("Mask size for column %d: %d", c, masks[c].sum())
        tokenizer.masks = {c: masks[c].tolist() for c in range(ncolumns)}

        # build JSON representation
        tokenizer_dict = {
            "dictionary": tokenizer.dictionary.to_dict(),
            "summary": tokenizer.summary_dict,
            "masks": tokenizer.masks,
        }

        tokenizer_file = config.input_path / f"tokenizer_{config.input_suffix}.json"
        logger.info('Writing the tokenizer to "%s"', tokenizer_file)
        with tokenizer_file.open("w") as f:
            dump(tokenizer_dict, f)

        logger.info("Validating file representation")
        with tokenizer_file.open("r") as f:
            data = load(f, object_hook=SequenceTokenizer.json_decode)

            if data != tokenizer_dict:
                error = "Loaded data is not the same as the original dictionary"
                raise ValueError(error)

        tokenizer_loaded = SequenceTokenizer.load(config, logger)
        if (
            tokenizer_loaded.dictionary.word2idx != tokenizer.dictionary.word2idx
            or tokenizer_loaded.dictionary.idx2word != tokenizer.dictionary.idx2word
            or tokenizer_loaded.summary_dict != tokenizer.summary_dict
        ):
            error = "Sequence tokenizer data does not match"
            raise ValueError(error)
