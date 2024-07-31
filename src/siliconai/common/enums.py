"""Common enumerations."""

from enum import Enum


class DataType(Enum):
    """Input type."""

    # actual types
    ActsChain = "ActsChain"
    ActsHits = "ActsHits"
    TRKNtuple = "TRKNtuple"

    # data types
    MNIST = "MNIST"
    FashionMNIST = "FashionMNIST"

    # test types
    TestSequence = "TestSequence"


class DataLoadingType(str, Enum):
    """Data loading type."""

    fit = "fit"
    validate = "validate"
    test = "test"


class ModelType(Enum):
    """Model type."""

    BasicVAE = "BasicVAE"
    ConvVAE = "ConvVAE"
    ChainTransformer = "ChainTransformer"
    DiscreteTransformer = "DiscreteTransformer"
