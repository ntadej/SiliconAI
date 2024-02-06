"""Model training helpers."""
import lightning as L
import matplotlib
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.loggers import MLFlowLogger
from torch import utils
from torchvision.datasets import MNIST  # type: ignore
from torchvision.transforms import ToTensor  # type: ignore

from siliconai.cli.config import TaskConfiguration
from siliconai.cli.logging import Logger
from siliconai.ml.models.vae import Decoder, Encoder, VAEModule


def train(logger: Logger, task_config: TaskConfiguration, diagnostics: bool) -> None:
    """Train the VAE model - temporary test function."""
    dataset_path = "~/datasets"

    logger.info(task_config.output_file)

    encoder = Encoder(784 + 16, 400, 200)
    decoder = Decoder(200 + 16, 400, 784)

    autoencoder = VAEModule(encoder, decoder)

    # setup data
    dataset = MNIST(dataset_path, train=True, download=True, transform=ToTensor())
    train_loader = utils.data.DataLoader(dataset, batch_size=100, shuffle=True)

    # setup trainint
    ml_logger = MLFlowLogger(
        experiment_name="siliconai",
        tracking_uri="http://127.0.0.1:8080",
    )
    trainer = L.Trainer(max_epochs=30, logger=ml_logger)
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)

    if diagnostics:
        with torch.no_grad():
            x = autoencoder.generate_class(
                torch.tensor([list(range(10))] * 5).clone().view(-1),
            )

        for i in range(50):
            plt.subplot(5, 10, i + 1)
            plt.axis("off")
            plt.imshow(x[i].squeeze(0).cpu().numpy(), cmap=matplotlib.cm.gray)  # type: ignore
            plt.savefig("test.pdf")
