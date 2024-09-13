"""Model training helpers."""

import lightning as L

from siliconai.cli.config import Configuration
from siliconai.cli.logging import Logger
from siliconai.common.enums import DataLoadingType
from siliconai.ml.training.loaders import load_data_module, load_model
from siliconai.ml.training.utils import common_setup, setup_callbacks, setup_logging
from siliconai.plotting.validation import quick_validate


def train(
    logger: Logger,
    config: Configuration,
    diagnostics: bool = False,
    batch: bool = False,
    n_gpu: int = 1,
    n_node: int = 1,
) -> None:
    """Train the model."""
    common_setup()

    # setup run number
    logger.info("Run number %d", config.run_number)

    # load data
    data = load_data_module(logger, config)

    # load model
    model = load_model(logger, config)
    logger.info(model)

    # define callbacks
    callbacks = setup_callbacks(config)

    # setup logging
    ml_logger = setup_logging(config)

    # setup training
    trainer = L.Trainer(
        accelerator="gpu" if batch else "auto",
        devices=n_gpu,
        num_nodes=n_node,
        strategy="ddp_find_unused_parameters_true"
        if n_gpu > 1 or n_node > 1
        else "auto",
        max_epochs=config.training.epochs,
        logger=ml_logger,
        callbacks=callbacks,
        default_root_dir="run/training",
        gradient_clip_val=config.training.gradient_clipping
        if config.training.gradient_clipping > 0
        else None,
    )

    # train the model
    trainer.fit(model, datamodule=data, ckpt_path="last")

    # test the model
    trainer.test(model, datamodule=data)

    # diagnostics
    if diagnostics:
        quick_validate(logger, config, model, data, DataLoadingType.test)
