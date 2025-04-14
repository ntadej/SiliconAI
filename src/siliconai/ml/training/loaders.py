"""Module loaders."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from siliconai.common.enums import DataType
from siliconai.data.modules import (
    ActsChainDataModule,
    ActsHitsDataModule,
)
from siliconai.ml.common.module import TransformerModule

if TYPE_CHECKING:
    import lightning as L

    from siliconai.cli.config import Configuration
    from siliconai.cli.logger import Logger


def load_data_module(
    logger: Logger | None,
    config: Configuration,
) -> L.LightningDataModule:
    """Load the data module based on the configuration."""
    if logger:
        logger.info("Loading data type: %s", config.data.type.value)

    if config.data.type is DataType.ActsChain:
        return ActsChainDataModule(config, logger)
    if config.data.type is DataType.ActsHits:
        return ActsHitsDataModule(config, logger)
    raise RuntimeError


def load_data_module_from_checkpoint(
    logger: Logger,
    config: Configuration,
    checkpoint: Path,
) -> L.LightningDataModule:
    """Load the data module from checkpoint."""
    logger.info("Loading data module type: %s", config.model.type.value)

    data_module: L.LightningDataModule
    if config.data.type is DataType.ActsChain:
        data_module = ActsChainDataModule.load_from_checkpoint(checkpoint)
        return data_module
    if config.data.type is DataType.ActsHits:
        data_module = ActsHitsDataModule.load_from_checkpoint(checkpoint)
        return data_module
    raise RuntimeError


def load_data_module_from_latest_checkpoint(
    logger: Logger,
    config: Configuration,
    path: Path,
    checkpoint: int = -1,
) -> L.LightningDataModule:
    """Load the model from the latest checkpoint."""
    logger.info("Loading data module from the latest checkpoint in %s.", path)

    files = path.glob("*")
    file = Path()
    if checkpoint > 0:
        for f in files:
            if f.name.startswith(f"epoch={checkpoint}"):
                file = f
                break
        if not file:
            error = f"Checkpoint {checkpoint} not found."
            raise ValueError(error)
    else:
        file = max(files, key=lambda x: x.stat().st_ctime)

    return load_data_module_from_checkpoint(logger, config, file)


def load_model(logger: Logger, config: Configuration) -> L.LightningModule:
    """Load the model based on the configuration."""
    logger.info("Loading model type: %s", config.model.type.value)
    return TransformerModule(config)


def load_model_from_checkpoint(
    logger: Logger,
    config: Configuration,
    checkpoint: Path,
) -> L.LightningModule:
    """Load the model from checkpoint."""
    logger.info("Loading model type: %s", config.model.type.value)

    model: L.LightningModule
    if config.model.type.is_transformer():
        model = TransformerModule.load_from_checkpoint(checkpoint)
        return model

    error = f"Model type {config.model.type} not supported."
    raise ValueError(error)


def load_model_from_latest_checkpoint(
    logger: Logger,
    config: Configuration,
    path: Path,
    checkpoint: int = -1,
) -> L.LightningModule:
    """Load the model from the latest checkpoint."""
    logger.info("Loading model from the latest checkpoint in %s.", path)

    files = path.glob("*")
    file = Path()
    if checkpoint > 0:
        for f in files:
            if f.name.startswith(f"epoch={checkpoint}"):
                file = f
                break
        if not file:
            error = f"Checkpoint {checkpoint} not found."
            raise ValueError(error)
    else:
        file = max(files, key=lambda x: x.stat().st_ctime)

    return load_model_from_checkpoint(logger, config, file)
