"""Continuous data transformations and helpers."""

from __future__ import annotations

from json import dump, load
from typing import TYPE_CHECKING

import numpy as np
from sklearn.preprocessing import (  # type: ignore[import-untyped]
    MinMaxScaler,
    StandardScaler,
)

from siliconai.data.datasets import ActsHitsDataset
from siliconai.data.utils import NDArrayTransformation, NDArrayType

if TYPE_CHECKING:
    from sklearn import BaseEstimator  # type: ignore[import-untyped]

    from siliconai.cli.config import DataConfiguration
    from siliconai.cli.logger import Logger

MIN_MAX = False


class ScikitLearnTransformation(NDArrayTransformation):
    """Normalize the input data to a function."""

    def __init__(self, size: int) -> None:
        """Initialize the transformation."""
        self.size = size
        self.transformations: list[BaseEstimator] = []

    def summary(self, logger: Logger) -> None:
        """Log summary of the transformation."""
        # logger.info(
        #     'Min/max for "%s": %f/%f',
        #     "name",  # self.name,
        #     self.transformation.scale_[0],
        # )

    def __call__(self, sample: NDArrayType) -> NDArrayType:
        """Transform the sample to tensors."""
        for i in range(self.size):
            sample[:, i] = (
                self.transformations[i]
                .transform(sample[:, i].reshape(-1, 1))
                .reshape(1, -1)
            )
        return sample

    def inverse(self, sample: NDArrayType) -> NDArrayType:
        """Inverse the tokenization."""
        for i in range(self.size):
            sample[:, i] = (
                self.transformations[i]
                .inverse_transform(sample[:, i].reshape(-1, 1))
                .reshape(1, -1)
            )
        return sample

    @staticmethod
    def load(
        config: DataConfiguration,
        logger: Logger | None = None,
    ) -> ScikitLearnTransformation:
        """Load the transformation from JSON."""
        if not config.input_file:
            error = "Invalid configuration"
            raise RuntimeError(error)

        transformation_file = config.input_file.with_suffix(".transformation.json")
        if logger:
            logger.info('Loading the transformation from "%s"', transformation_file)
        with transformation_file.open("r") as f:
            data = load(f)

        transformation = ScikitLearnTransformation(len(data["transformations"]))
        transformation.transformations = [
            MinMaxScaler() if MIN_MAX else StandardScaler()
            for i in range(transformation.size)
        ]
        for i in range(transformation.size):
            if MIN_MAX:
                transformation.transformations[i].min_ = np.array(
                    [data["transformations"][i]["min"]],
                )
            else:
                transformation.transformations[i].var_ = np.array(
                    [data["transformations"][i]["var"]],
                )
                transformation.transformations[i].mean_ = np.array(
                    [data["transformations"][i]["mean"]],
                )
            transformation.transformations[i].scale_ = np.array(
                [data["transformations"][i]["scale"]],
            )

        return transformation

    @staticmethod
    def train(config: DataConfiguration, logger: Logger) -> None:
        """Train the transformation."""
        if not config.input_file:
            return

        ncolumns = len(config.columns_float)
        if ncolumns == 0:
            return

        transformation = ScikitLearnTransformation(ncolumns)
        transformation.transformations = [
            MinMaxScaler() if MIN_MAX else StandardScaler() for i in range(ncolumns)
        ]

        logger.info("Tokenizing the input file with %d continuous columns", ncolumns)

        dataset = ActsHitsDataset(config.input_file)
        data_float_list = [dataset[i][1] for i in range(len(dataset))]
        data_float = np.vstack(data_float_list)

        for c, column in enumerate(config.columns_float):
            logger.info("Fitting column %d: %s", c, column)
            transformation.transformations[c].fit(data_float[:, c].reshape(-1, 1))

        # build JSON representation
        transformation_dict = {
            "transformations": [
                {
                    "name": column,
                    "min": transformation.min_.tolist()[0],
                    "scale": transformation.scale_.tolist()[0],
                }
                if MIN_MAX
                else {
                    "name": column,
                    "var": transformation.var_.tolist()[0],
                    "mean": transformation.mean_.tolist()[0],
                    "scale": transformation.scale_.tolist()[0],
                }
                for column, transformation in zip(
                    config.columns_float,
                    transformation.transformations,
                    strict=True,
                )
            ],
        }

        transformation_file = config.input_file.with_suffix(".transformation.json")
        logger.info('Writing the transformation to "%s"', transformation_file)
        with transformation_file.open("w") as f:
            dump(transformation_dict, f)

        logger.info("Validating file representation")
        with transformation_file.open("r") as f:
            data = load(f)

            if data != transformation_dict:
                error = "Loaded data is not the same as the original dictionary"
                raise ValueError(error)

        transformation_loaded = ScikitLearnTransformation.load(config, logger)
        for i in range(ncolumns):
            valid = (
                transformation_loaded.transformations[i].scale_
                == transformation.transformations[i].scale_
            )
            if MIN_MAX:
                valid |= (
                    transformation_loaded.transformations[i].min_
                    == transformation.transformations[i].min_
                )
            else:
                valid |= (
                    transformation_loaded.transformations[i].mean_
                    == transformation.transformations[i].mean_
                )
                valid |= (
                    transformation_loaded.transformations[i].var_
                    == transformation.transformations[i].var_
                )

            if not valid:
                error = (
                    "ScikitLearnTransformation saved representation"
                    " does not match the original one"
                )
                raise ValueError(error)
