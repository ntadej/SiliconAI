"""Common enumerations."""
from enum import Enum


class DataType(Enum):
    """Input type."""

    # actual types
    TRKNtuple = "TRKNtuple"

    # data types
    MNIST = "MNIST"


class ModelType(Enum):
    """Model type."""

    BasicVAE = "BasicVAE"
    ConditioningVAE = "ConditioningVAE"
