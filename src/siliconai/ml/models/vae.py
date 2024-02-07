"""Variational autoencoder models."""
from __future__ import annotations

import lightning as L
import torch
from torch import Tensor, nn, optim


class Encoder(nn.Module):
    """Encoder model module."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
    ) -> None:
        """Initialize the module."""
        super().__init__()

        self.model_main = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.model_mu = nn.Linear(hidden_dim, latent_dim)
        self.model_log_var = nn.Linear(hidden_dim, latent_dim)

        self.training = True

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


class Decoder(nn.Module):
    """Decoder model module."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        """Initialize the module."""
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """Forward pass."""
        i = torch.cat((x, y), dim=1) if y is not None else x

        return torch.sigmoid(self.model(i))


class VAEModule(L.LightningModule):
    """VAE model model."""

    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        """Initialize the module."""
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = nn.Embedding(10, 16)

        self.example_input_array = (
            torch.Tensor(100, 1, 28, 28),
            torch.randint(0, 10, (100,)),
        )

    @staticmethod
    def reparameterization(mu: Tensor, log_var: Tensor) -> Tensor:
        """Reparameterize the results for Gaussians."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

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
        x, y = batch
        y = self.embedding(y)
        x = x.view(x.size(0), 784)
        mu, log_var = self.encoder(x, y)
        z = self.reparameterization(mu, log_var)
        x_hat: Tensor = self.decoder(z, y)
        return self.loss_function(x, x_hat, mu, log_var)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Forward pass."""
        x = x.view(x.size(0), 784)
        y = self.embedding(y)
        mu, log_var = self.encoder(x, y)
        z = self.reparameterization(mu, log_var)
        x_hat: Tensor = self.decoder(z, y)
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

    def generate_class(self, class_idx: int | Tensor) -> Tensor:
        """Generate class."""
        if isinstance(class_idx, int):
            class_idx = torch.tensor(class_idx)
        class_idx = class_idx.to(self.device)
        if len(class_idx.shape) == 0:
            batch_size = 1
            class_idx = class_idx.unsqueeze(0)
            z = torch.randn((1, 200)).to(self.device)
        else:
            batch_size = class_idx.shape[0]
            z = torch.randn((batch_size, 200)).to(self.device)
        y = self.embedding(class_idx)
        res: Tensor = self.decoder(z, y)
        res = res.view(batch_size, 28, 28)
        if not batch_size:
            res = res.squeeze(0)
        return res
