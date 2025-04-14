"""Common enumerations."""

from enum import Enum


class DataType(Enum):
    """Input type."""

    ActsChain = "ActsChain"
    ActsHits = "ActsHits"


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

    ChainTransformer = "ChainTransformer"
    DiscreteTransformer = "DiscreteTransformer"
    HybridTransformer = "HybridTransformer"

    def is_transformer(self) -> bool:
        """Return true if the model is a transformer."""
        return self in [
            ModelType.ChainTransformer,
            ModelType.DiscreteTransformer,
            ModelType.HybridTransformer,
        ]
