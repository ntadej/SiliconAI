"""Model training helpers."""

import lightning as L

from siliconai.cli.config import Configuration
from siliconai.cli.logging import Logger
from siliconai.common.enums import DataLoadingType, ModelType
from siliconai.ml.training.loaders import load_data_module, load_model
from siliconai.ml.training.utils import common_setup, setup_callbacks, setup_logging
from siliconai.plotting.validation import quick_validate


def train(logger: Logger, config: Configuration, diagnostics: bool) -> None:
    """Train the model."""
    common_setup()

    # setup run number
    logger.info("Run number %d", config.run_number(training=True))

    # load data
    data = load_data_module(logger, config)

    # load model
    model = load_model(logger, config)
    logger.info(model)
    if config.model.type is ModelType.ConvVAE:
        logger.info(
            "Convolution image sizes: %s, %s, %s",
            model.encoder.sizes,
            model.decoder.sizes,
            model.decoder.paddings,
        )

    # define callbacks
    callbacks = setup_callbacks(config)

    # setup logging
    ml_logger = setup_logging(config)

    # setup training
    trainer = L.Trainer(
        max_epochs=config.training.epochs,
        logger=ml_logger,
        callbacks=callbacks,
        default_root_dir="run/training",
    )

    # train the model
    trainer.fit(model, datamodule=data)

    # test the model
    trainer.test(model, datamodule=data)

    # diagnostics
    if diagnostics:
        quick_validate(logger, config, model, data, DataLoadingType.test)
