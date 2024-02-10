"""Module loaders."""
from pathlib import Path

import lightning as L

from siliconai.cli.config import Configuration
from siliconai.cli.logging import Logger
from siliconai.common.enums import DataType, ModelType
from siliconai.data.modules import MNISTDataModule
from siliconai.ml.models.vae import BasicVAE, EmbeddingVAE


def load_data_module(logger: Logger, config: Configuration) -> L.LightningDataModule:
    """Load the data module based on the configuration."""
    logger.info("Loading data type: %s", config.data.type.value)

    if config.data.type is DataType.MNIST:
        return MNISTDataModule(config)

    error = f"Data type {config.data.type} not supported."
    raise ValueError(error)


def load_model(logger: Logger, config: Configuration) -> L.LightningModule:
    """Load the model based on the configuration."""
    logger.info("Loading model type: %s", config.model.type.value)

    if config.model.type is ModelType.BasicVAE:
        return BasicVAE(config)
    if config.model.type is ModelType.EmbeddingVAE:
        return EmbeddingVAE(config)

    error = f"Model type {config.model.type} not supported."  # type: ignore
    raise ValueError(error)


def load_model_from_checkpoint(
    logger: Logger,
    config: Configuration,
    checkpoint: Path,
) -> L.LightningModule:
    """Load the model from checkpoint."""
    logger.info("Loading model type: %s", config.model.type.value)

    model: L.LightningModule
    if config.model.type is ModelType.BasicVAE:
        model = BasicVAE.load_from_checkpoint(checkpoint)
        return model  # noqa: RET504
    if config.model.type is ModelType.EmbeddingVAE:
        model = EmbeddingVAE.load_from_checkpoint(checkpoint)
        return model  # noqa: RET504

    error = f"Model type {config.model.type} not supported."  # type: ignore
    raise ValueError(error)


def load_model_from_latest_checkpoint(
    logger: Logger,
    config: Configuration,
    path: Path,
) -> L.LightningModule:
    """Load the model from the latest checkpoint."""
    logger.info("Loading model from the latest checkpoint in %s.", path)

    files = path.glob("*")
    file = max(files, key=lambda x: x.stat().st_ctime)

    return load_model_from_checkpoint(logger, config, file)
