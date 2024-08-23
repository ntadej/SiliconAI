"""Data tokenizers and helpers."""

from __future__ import annotations

from enum import Enum
from json import dump, load
from typing import TYPE_CHECKING, Any

import numpy as np

from siliconai.common.enums import ColumnType
from siliconai.data.datasets import ActsChainDataset, ActsHitsDataset
from siliconai.data.utils import NDArrayTransformation, NDArrayType

if TYPE_CHECKING:
    from siliconai.cli.config import DataConfiguration
    from siliconai.cli.logging import Logger

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


class ColumnTokenizer(NDArrayTransformation):
    """Tokenize the input data."""

    def __init__(self, size: int) -> None:
        """Initialize the tokenizer."""
        self.size = size
        self.dictionaries = [DataDictionary(f"dictionary_{i}") for i in range(size)]

    def summary(self, logger: Logger) -> None:
        """Log summary of the dictionary."""
        for dictionary in self.dictionaries:
            logger.info(
                'Dictionary for "%s": %d words',
                dictionary.name,
                len(dictionary),
            )

    def __call__(self, sample: NDArrayType) -> NDArrayType:
        """Transform the sample to tensors."""
        for i in range(self.size):
            sample[:, i] = self.dictionaries[i].add_word_v(sample[:, i])
        return sample

    def inverse(self, sample: NDArrayType) -> NDArrayType:
        """Inverse the tokenization."""
        for i in range(self.size):
            sample[:, i] = self.dictionaries[i].get_word_v(sample[:, i])
        return sample

    @staticmethod
    def load(
        config: DataConfiguration,
        logger: Logger | None = None,
    ) -> ColumnTokenizer:
        """Load the tokenizer from JSON."""
        if not config.input_file:
            error = "Invalid configuration"
            raise RuntimeError(error)

        tokenizer_file = config.input_file.with_suffix(".tokenizer.json")
        if logger:
            logger.info('Loading the tokenizer from "%s"', tokenizer_file)
        with tokenizer_file.open("r") as f:
            data = load(f, object_hook=DataDictionary.json_decode)

        tokenizer = ColumnTokenizer(len(data["dictionaries"]))
        keys_list = list(data["dictionaries"].keys())
        for i in range(tokenizer.size):
            tokenizer.dictionaries[i].name = keys_list[i]
            tokenizer.dictionaries[i].from_dict(data["dictionaries"][keys_list[i]])

        return tokenizer

    @staticmethod
    def train(config: DataConfiguration, logger: Logger) -> None:
        """Train the tokenizer."""
        if not config.input_file:
            return

        ncolumns = len(config.columns_integer)
        if ncolumns == 0:
            return

        if isinstance(config.input_dim, list):
            dimensions = config.input_dim
        else:
            dimensions = [config.input_dim]

        tokenizer = ColumnTokenizer(ncolumns)
        tokenizer.dictionaries = [
            DataDictionary(f"dictionary_{column}", config.padding_token)
            for column in config.columns_integer
        ]

        logger.info("Tokenizing the input file with %d columns", ncolumns)

        dataset = ActsHitsDataset(config.input_file)
        for i in range(len(dataset)):
            row = dataset[i][0]
            tokenizer(row)

        for i, (column, dim) in enumerate(
            zip(config.columns_integer, dimensions, strict=True),
        ):
            logger.info("Expected dictionary size for %s: %d words", column, dim)
            logger.info(
                "Actual dictionary size for %s: %d words",
                column,
                len(tokenizer.dictionaries[i]),
            )
            # tokenizer.summary(logger)

            assert len(tokenizer.dictionaries[i]) == dim

        # build JSON representation
        tokenizer_dict = {
            "dictionaries": {
                dictionary.name: dictionary.to_dict()
                for dictionary in tokenizer.dictionaries
            },
        }

        tokenizer_file = config.input_file.with_suffix(".tokenizer.json")
        logger.info('Writing the tokenizer to "%s"', tokenizer_file)
        with tokenizer_file.open("w") as f:
            dump(tokenizer_dict, f)

        logger.info("Validating file representation")
        with tokenizer_file.open("r") as f:
            data = load(f, object_hook=DataDictionary.json_decode)

            assert data == tokenizer_dict

        tokenizer_loaded = ColumnTokenizer.load(config, logger)
        for i in range(ncolumns):
            assert (
                tokenizer_loaded.dictionaries[i].name == tokenizer.dictionaries[i].name
            )
            assert (
                tokenizer_loaded.dictionaries[i].word2idx
                == tokenizer.dictionaries[i].word2idx
            )
            assert (
                tokenizer_loaded.dictionaries[i].idx2word
                == tokenizer.dictionaries[i].idx2word
            )


class SequenceTokenizer(NDArrayTransformation):
    """Tokenize the sequence input data."""

    def __init__(self) -> None:
        """Initialize the tokenizer."""
        self.dictionary = DataDictionary("dictionary")
        self.summary_dict: dict[str, dict[str, int]] = {}

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
    def load(
        config: DataConfiguration,
        logger: Logger | None = None,
    ) -> SequenceTokenizer:
        """Load the tokenizer from JSON."""
        if not config.input_file:
            error = "Invalid configuration"
            raise RuntimeError(error)

        tokenizer_file = config.input_file.with_suffix(".tokenizer.json")
        if logger:
            logger.info('Loading the tokenizer from "%s"', tokenizer_file)
        with tokenizer_file.open("r") as f:
            data = load(f, object_hook=DataDictionary.json_decode)

        dictionary = DataDictionary("dictionary", config.padding_token)
        dictionary.from_dict(data["dictionary"])
        tokenizer = SequenceTokenizer()
        tokenizer.dictionary = dictionary
        tokenizer.summary_dict = data["summary"]

        return tokenizer

    @staticmethod
    def train(config: DataConfiguration, logger: Logger) -> None:
        """Train the tokenizer."""
        if not config.input_file:
            return

        tokenizer = SequenceTokenizer()
        tokenizer.dictionary = DataDictionary("dictionary", config.padding_token)

        ncolumns = len(config.columns_integer) + len(
            [c for c in config.columns_type if c == ColumnType.Numerical],
        )

        logger.info(
            "Tokenizing the input file with %d columns and %d parts",
            len(config.columns_integer),
            ncolumns,
        )

        total = 1

        summary_dict = {}

        dataset = ActsChainDataset(config.input_file)
        s = 0
        for c, (column, column_type) in enumerate(
            zip(
                config.columns_integer,
                config.columns_type,
                strict=True,
            ),
        ):
            logger.info("Tokenizing column %d: %s", c, column)
            for i in range(len(dataset)):
                row = dataset[i]
                tokenizer(row[s::ncolumns])
                if column_type == ColumnType.Numerical:
                    tokenizer(row[s + 1 :: ncolumns])
            s += 2 if column_type == ColumnType.Numerical else 1

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

        if isinstance(config.input_dim, list):
            dim = sum(config.input_dim)
        else:
            dim = config.input_dim
        dim += 1  # add padding token

        logger.info("Expected dictionary size: %d words", dim)
        tokenizer.summary(logger)

        assert len(tokenizer.dictionary) == dim

        # build JSON representation
        tokenizer_dict = {
            "dictionary": tokenizer.dictionary.to_dict(),
            "summary": tokenizer.summary_dict,
        }

        tokenizer_file = config.input_file.with_suffix(".tokenizer.json")
        logger.info('Writing the tokenizer to "%s"', tokenizer_file)
        with tokenizer_file.open("w") as f:
            dump(tokenizer_dict, f)

        logger.info("Validating file representation")
        with tokenizer_file.open("r") as f:
            data = load(f, object_hook=DataDictionary.json_decode)

            assert data == tokenizer_dict

        tokenizer_loaded = SequenceTokenizer.load(config, logger)
        assert tokenizer_loaded.dictionary.word2idx == tokenizer.dictionary.word2idx
        assert tokenizer_loaded.dictionary.idx2word == tokenizer.dictionary.idx2word
        assert tokenizer_loaded.summary_dict == tokenizer.summary_dict
