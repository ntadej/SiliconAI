"""Variational autoencoder models."""
from __future__ import annotations

from typing import TYPE_CHECKING

import lightning as L
import torch
from torch import Tensor, nn, optim

from siliconai.ml.models.common import SequentialMLP

if TYPE_CHECKING:
    from siliconai.cli.config import DataConfiguration, ModelConfiguration


class Encoder(nn.Module):
    """Encoder model module."""

    def __init__(self, config: ModelConfiguration, input_dim: int) -> None:
        """Initialize the module."""
        super().__init__()

        if config.embedding:
            input_dim += config.embedding[1]

        self.model_main = SequentialMLP(
            [input_dim, *config.encoder_layers],
            config.activation,
            config.activation_parameters,
            config.batch_norm,
            config.dropout,
        )
        self.model_mu = nn.Linear(config.encoder_layers[-1], config.latent_dim)
        self.model_log_var = nn.Linear(config.encoder_layers[-1], config.latent_dim)

    def forward(
        self,
        x: Tensor,
        y: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass."""
        i = torch.cat((x, y), dim=1) if y is not None else x
        h = self.model_main(i)
        mu = self.model_mu(h)
        log_var = self.model_log_var(h)
        return mu, log_var

    @staticmethod
    def reparameterization(mu: Tensor, log_var: Tensor) -> Tensor:
        """Reparameterize the results for Gaussians."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps


class Decoder(nn.Module):
    """Decoder model module."""

    def __init__(self, config: ModelConfiguration, input_dim: int) -> None:
        """Initialize the module."""
        super().__init__()

        latent_dim = config.latent_dim
        if config.embedding:
            latent_dim += config.embedding[1]

        self.model = SequentialMLP(
            [latent_dim, *config.decoder_layers, input_dim],
            config.activation,
            config.activation_parameters,
            config.batch_norm,
            config.dropout,
            output_batch_norm=False,
        )

    def forward(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """Forward pass."""
        i = torch.cat((x, y), dim=1) if y is not None else x

        return torch.sigmoid(self.model(i))


class BasicVAE(L.LightningModule):
    """Basic VAE model."""

    def __init__(
        self,
        config: ModelConfiguration,
        data_config: DataConfiguration,
    ) -> None:
        """Initialize the module."""
        super().__init__()
        self.config = config
        self.input_dim = data_config.input_dim
        self.flat_input_dim = data_config.flat_input_dim

        self.encoder = Encoder(config, data_config.flat_input_dim)
        self.decoder = Decoder(config, data_config.flat_input_dim)

        self.example_input_array = (
            torch.Tensor(data_config.batch_size, 1, *data_config.input_dim)
            if isinstance(data_config.input_dim, list)
            else torch.Tensor(data_config.batch_size, 1, data_config.input_dim)
        )

    @staticmethod
    def loss_function(x: Tensor, x_hat: Tensor, mu: Tensor, log_var: Tensor) -> Tensor:
        """VAE loss function."""
        reproduction_loss = nn.functional.binary_cross_entropy(
            x_hat,
            x,
            reduction="mean",
        )
        kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return reproduction_loss + kld

    def process_loss(self, batch: Tensor) -> Tensor:
        """Process the loss of a batch."""
        x, _ = batch
        x = x.view(x.size(0), self.flat_input_dim)
        mu, log_var = self.encoder(x)
        z = self.encoder.reparameterization(mu, log_var)
        x_hat: Tensor = self.decoder(z)
        return self.loss_function(x, x_hat, mu, log_var)

    def forward(self, *args: Tensor) -> Tensor:
        """Forward pass."""
        x = args[0]
        x = x.view(x.size(0), self.flat_input_dim)
        mu, log_var = self.encoder(x)
        z = self.encoder.reparameterization(mu, log_var)
        x_hat: Tensor = self.decoder(z)
        return x_hat

    def training_step(self, batch: Tensor, _batch_idx: int) -> Tensor:
        """Run training step."""
        loss = self.process_loss(batch)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tensor, _batch_idx: int) -> Tensor:
        """Run validation step."""
        loss = self.process_loss(batch)

        self.log("val_loss", loss)
        return loss

    def test_step(self, batch: Tensor, _batch_idx: int) -> Tensor:  # noqa: PT019
        """Run test step."""
        loss = self.process_loss(batch)

        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        """Configure optimizers."""
        return optim.Adam(self.parameters(), lr=1e-3)

    def generate(self, batch_size: int) -> Tensor:
        """Generate class."""
        z = torch.randn((batch_size, self.config.latent_dim)).to(self.device)
        res: Tensor = self.decoder(z)
        if isinstance(self.input_dim, list):
            res = res.view(batch_size, *self.input_dim)
        return res


class EmbeddingVAE(BasicVAE):
    """VAE model model."""

    def __init__(
        self,
        config: ModelConfiguration,
        data_config: DataConfiguration,
    ) -> None:
        """Initialize the module."""
        super().__init__(config, data_config)
        if config.embedding is None:
            error = "EmbeddingVAE requires an embedding layer"
            raise ValueError(error)

        self.embedding = nn.Embedding(*config.embedding)

        self.example_input_array = (
            self.example_input_array,
            torch.randint(0, config.embedding[0], (data_config.batch_size,)),
        )

    def process_loss(self, batch: Tensor) -> Tensor:
        """Process the loss of a batch."""
        x, y = batch
        y = self.embedding(y)
        x = x.view(x.size(0), self.flat_input_dim)
        mu, log_var = self.encoder(x, y)
        z = self.encoder.reparameterization(mu, log_var)
        x_hat: Tensor = self.decoder(z, y)
        return self.loss_function(x, x_hat, mu, log_var)

    def forward(self, *args: Tensor) -> Tensor:
        """Forward pass."""
        x, y = args[0], args[1]
        x = x.view(x.size(0), self.flat_input_dim)
        y = self.embedding(y)
        mu, log_var = self.encoder(x, y)
        z = self.encoder.reparameterization(mu, log_var)
        x_hat: Tensor = self.decoder(z, y)
        return x_hat

    def generate_class(self, class_idx: Tensor) -> Tensor:
        """Generate class."""
        if self.config.embedding is None:
            error = "EmbeddingVAE requires an embedding layer"
            raise RuntimeError(error)

        class_idx = class_idx.to(self.device)
        batch_size = class_idx.shape[0]
        z = torch.randn((batch_size, self.config.latent_dim)).to(self.device)
        y = self.embedding(class_idx)
        res: Tensor = self.decoder(z, y)
        if isinstance(self.input_dim, list):
            res = res.view(batch_size, *self.input_dim)
        return res
