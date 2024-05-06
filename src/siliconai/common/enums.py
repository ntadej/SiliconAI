"""Common enumerations."""

from enum import Enum


class DataType(Enum):
    """Input type."""

    # actual types
    TRKNtuple = "TRKNtuple"

    # data types
    MNIST = "MNIST"
    FashionMNIST = "FashionMNIST"

    # test types
    TestSequence = "TestSequence"


class ModelType(Enum):
    """Model type."""

    BasicVAE = "BasicVAE"
    ConvVAE = "ConvVAE"
    Transformer = "Transformer"
