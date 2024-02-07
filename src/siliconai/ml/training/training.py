"""Model training helpers."""
import lightning as L
import matplotlib
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from torch import utils
from torchvision.datasets import MNIST  # type: ignore
from torchvision.transforms import ToTensor  # type: ignore

from siliconai.cli.config import TaskConfiguration
from siliconai.cli.logging import Logger
from siliconai.ml.models.vae import Decoder, Encoder, VAEModule


def train(logger: Logger, task_config: TaskConfiguration, diagnostics: bool) -> None:
    """Train the VAE model - temporary test function."""
    # matmul precision and seed
    torch.set_float32_matmul_precision("high")
    L.seed_everything(42, workers=True)

    dataset_path = "~/datasets"

    logger.info(task_config.output_file)

    encoder = Encoder(784 + 16, 400, 200)
    decoder = Decoder(200 + 16, 400, 784)

    autoencoder = VAEModule(encoder, decoder)

    # setup data
    train_set = MNIST(dataset_path, train=True, download=True, transform=ToTensor())
    test_set = MNIST(dataset_path, train=False, download=True, transform=ToTensor())

    # split the test set into two
    seed = torch.Generator().manual_seed(42)
    val_set, test_set = utils.data.random_split(
        test_set,
        [0.5, 0.5],
        generator=seed,
    )

    train_loader = utils.data.DataLoader(
        train_set,
        batch_size=100,
        shuffle=True,
        num_workers=4,
    )
    val_loader = utils.data.DataLoader(
        val_set,
        batch_size=100,
        shuffle=False,
        num_workers=4,
    )
    test_loader = utils.data.DataLoader(
        test_set,
        batch_size=100,
        shuffle=False,
        num_workers=4,
    )

    # define callbacks
    callbacks = [
        RichProgressBar(),
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(monitor="val_loss", mode="min", patience=3),
        ModelCheckpoint(
            dirpath="run/checkpoints",
            save_weights_only=True,
            mode="min",
            monitor="val_loss",
            save_top_k=1,
        ),
    ]

    # setup logging
    ml_logger = MLFlowLogger(
        experiment_name="siliconai",
        tracking_uri="http://127.0.0.1:8080",
    )

    # setup training
    trainer = L.Trainer(
        max_epochs=50,
        logger=ml_logger,
        callbacks=callbacks,
        default_root_dir="run/training",
    )
    trainer.fit(
        model=autoencoder,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # test the model
    trainer.test(model=autoencoder, dataloaders=test_loader)

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
