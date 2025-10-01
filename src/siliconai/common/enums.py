"""Common enumerations."""

from enum import Enum


class DataLoadingType(str, Enum):
    """Data loading type."""

    fit = "fit"
    validate = "validate"
    test = "test"
    predict = "predict"


class ColumnType(Enum):
    """Column type."""

    Categorical = "categorical"
    Numerical = "numerical"


class ModelType(Enum):
    """Model type."""

    NanoGPT = "NanoGPT"
