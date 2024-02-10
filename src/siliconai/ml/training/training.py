"""Model training helpers."""
import lightning as L
import matplotlib
import matplotlib.pyplot as plt
import torch

from siliconai.cli.config import Configuration
from siliconai.cli.logging import Logger
from siliconai.common.enums import ModelType
from siliconai.ml.training.loaders import load_data_module, load_model
from siliconai.ml.training.utils import common_setup, setup_callbacks, setup_logging


def train(logger: Logger, config: Configuration, diagnostics: bool) -> None:
    """Train the model."""
    common_setup()

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
        with torch.no_grad():
            if config.model.type is ModelType.BasicVAE:
                x = model.generate(50)
            else:
                x = model.generate_class(
                    torch.tensor([list(range(10))] * 5).clone().view(-1),
                )

        for i in range(50):
            plt.subplot(5, 10, i + 1)
            plt.axis("off")
            plt.imshow(x[i].squeeze(0).cpu().numpy(), cmap=matplotlib.cm.gray)  # type: ignore
            plt.savefig("test.pdf")
