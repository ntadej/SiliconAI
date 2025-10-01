"""Module loaders."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from siliconai.data.modules import ActsChainDataModule
from siliconai.ml.common.module import NanoGPTModule

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
        logger.info("Loading data")

    return ActsChainDataModule(config, logger)


def load_data_module_from_checkpoint(
    logger: Logger,
    config: Configuration,
    checkpoint: Path,
) -> L.LightningDataModule:
    """Load the data module from checkpoint."""
    logger.info("Loading data module type: %s", config.model.type.value)

    data_module: L.LightningDataModule = ActsChainDataModule.load_from_checkpoint(
        checkpoint,
    )
    return data_module


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
        if file == Path():
            error = f"Checkpoint {checkpoint} not found."
            raise ValueError(error)
    else:
        file = max(files, key=lambda x: x.stat().st_ctime)

    return load_data_module_from_checkpoint(logger, config, file)


def load_model(logger: Logger, config: Configuration) -> L.LightningModule:
    """Load the model based on the configuration."""
    logger.info("Loading model type: %s", config.model.type.value)
    return NanoGPTModule(config)

    # error = f"Model type {config.model.type} not supported."
    # raise ValueError(error)


def load_model_from_checkpoint(
    logger: Logger,
    config: Configuration,
    checkpoint: Path,
) -> L.LightningModule:
    """Load the model from checkpoint."""
    logger.info("Loading model type: %s", config.model.type.value)

    device = torch.device(config.inference.device) if config.inference.device else None

    return NanoGPTModule.load_from_checkpoint(checkpoint, map_location=device)

    # error = f"Model type {config.model.type} not supported."
    # raise ValueError(error)


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
        if file == Path():
            error = f"Checkpoint {checkpoint} not found."
            raise ValueError(error)
    else:
        file = max(files, key=lambda x: x.stat().st_ctime)

    return load_model_from_checkpoint(logger, config, file)
