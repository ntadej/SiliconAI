"""Variational autoencoder models."""
from __future__ import annotations

import lightning as L
import torch
from torch import Tensor, nn, optim


class Encoder(nn.Module):
    """Encoder model module."""

    def __init__(
        self: Encoder,
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
        self.model_result = nn.Linear(hidden_dim, latent_dim)

        self.training = True

    def forward(
        self: Encoder,
        x: Tensor,
        y: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass."""
        i = torch.cat((x, y), dim=1) if y is not None else x
        h = self.model_main(i)
        mu = self.model_result(h)
        log_var = self.model_result(h)
        return mu, log_var


class Decoder(nn.Module):
    """Decoder model module."""

    def __init__(
        self: Decoder,
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

    def forward(self: Decoder, x: Tensor, y: Tensor | None = None) -> Tensor:
        """Forward pass."""
        i = torch.cat((x, y), dim=1) if y is not None else x

        return torch.sigmoid(self.model(i))


class VAEModule(L.LightningModule):
    """VAE model model."""

    def __init__(self: VAEModule, encoder: Encoder, decoder: Decoder) -> None:
        """Initialize the module."""
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = nn.Embedding(10, 16)

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
            reduction="sum",
        )
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return reproduction_loss + kld

    def training_step(self: VAEModule, batch: Tensor, _: int) -> Tensor:
        """Training step."""
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        y = self.embedding(y)
        x = x.view(x.size(0), 784)
        mu, log_var = self.encoder(x, y)
        z = self.reparameterization(mu, log_var)
        x_hat = self.decoder(z, y)
        loss = self.loss_function(x, x_hat, mu, log_var)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self: VAEModule) -> optim.Optimizer:
        """Configure optimizers."""
        return optim.Adam(self.parameters(), lr=1e-3)

    def generate_class(self: VAEModule, class_idx: int | Tensor) -> Tensor:
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
