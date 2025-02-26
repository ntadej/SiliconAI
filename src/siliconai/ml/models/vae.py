"""Variational autoencoder models."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from siliconai.ml.common.module import ModuleBase
from siliconai.ml.models.common import SequentialConv2d, SequentialMLP
from siliconai.ml.models.utils import conv2d_sizes, conv2d_transpose_sizes

if TYPE_CHECKING:
    from siliconai.cli.config import Configuration


class Encoder(nn.Module):
    """Encoder model module."""

    def __init__(self, config: Configuration) -> None:
        """Initialize the module."""
        super().__init__()

        if (
            not config.model.encoder_layers
            or isinstance(config.model.encoder_layers, int)
            or any(
                isinstance(layer_spec, tuple)
                for layer_spec in config.model.encoder_layers
            )
        ):
            error = "Encoder layers must be a list of integers"
            raise TypeError(error)

        input_dim = config.data.flat_input_dim
        encoder_layers = cast("list[int]", config.model.encoder_layers)

        self.model_main = SequentialMLP(
            [input_dim, *config.model.encoder_layers],
            config.model.activation,
            config.model.activation_parameters,
            config.model.batch_norm,
            config.model.dropout,
        )
        self.model_mu = nn.Linear(
            encoder_layers[-1],
            config.model.model_dim,
        )
        self.model_log_var = nn.Linear(
            encoder_layers[-1],
            config.model.model_dim,
        )

    def forward(
        self,
        x: Tensor,
        _y: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass."""
        h = self.model_main(x)
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

    def __init__(self, config: Configuration) -> None:
        """Initialize the module."""
        super().__init__()

        if (
            not config.model.decoder_layers
            or isinstance(config.model.decoder_layers, int)
            or any(
                isinstance(layer_spec, tuple)
                for layer_spec in config.model.decoder_layers
            )
        ):
            error = "Decoder layers must be a list of integers"
            raise TypeError(error)

        model_dim = config.model.model_dim
        if config.model.conditioning:
            model_dim += config.model.conditioning[1]
        if config.model.embedding:
            model_dim += config.model.embedding[1]

        self.model = SequentialMLP(
            [model_dim, *config.model.decoder_layers, config.data.flat_input_dim],
            config.model.activation,
            config.model.activation_parameters,
            config.model.batch_norm,
            config.model.dropout,
            output_activation=False,
            output_batch_norm=False,
        )

    def forward(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """Forward pass."""
        i = torch.cat((x, y), dim=1) if y is not None else x
        z: Tensor = self.model(i)
        return z


class EncoderConv2d(nn.Module):
    """Encoder model module."""

    def __init__(self, config: Configuration) -> None:
        """Initialize the module."""
        super().__init__()

        if isinstance(config.data.input_dim, int):
            error = "ConvVAE requires a list input_dim"
            raise TypeError(error)

        if (
            not config.model.encoder_layers
            or isinstance(config.model.encoder_layers, int)
            or any(
                isinstance(layer_spec, int)
                for layer_spec in config.model.encoder_layers
            )
        ):
            error = "Encoder layers must be a list of tuples"
            raise TypeError(error)

        encoder_layers = cast(
            "list[tuple[int, int, int, int]]",
            config.model.encoder_layers,
        )

        self.sizes = conv2d_sizes(config.data.input_dim[1], encoder_layers)

        self.model_main = SequentialConv2d(
            [(config.data.input_dim[0], 0, 1, 1), *encoder_layers],
            config.model.activation,
            config.model.activation_parameters,
            config.model.batch_norm,
            config.model.dropout,
        )
        self.model_flatten = nn.Flatten(start_dim=1)
        self.model_mu = nn.Linear(
            encoder_layers[-1][0] * self.sizes[-1] * self.sizes[-1],
            config.model.model_dim,
        )
        self.model_log_var = nn.Linear(
            encoder_layers[-1][0] * self.sizes[-1] * self.sizes[-1],
            config.model.model_dim,
        )

    def forward(
        self,
        x: Tensor,
        _y: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass."""
        h = self.model_main(x)
        f = self.model_flatten(h)
        mu = self.model_mu(f)
        log_var = self.model_log_var(f)
        return mu, log_var

    @staticmethod
    def reparameterization(mu: Tensor, log_var: Tensor) -> Tensor:
        """Reparameterize the results for Gaussians."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps


class DecoderConv2d(nn.Module):
    """Decoder model module."""

    def __init__(self, config: Configuration) -> None:
        """Initialize the module."""
        super().__init__()

        if isinstance(config.data.input_dim, int):
            error = "ConvVAE requires a list input_dim"
            raise TypeError(error)

        if (
            not config.model.decoder_layers
            or isinstance(config.model.decoder_layers, int)
            or any(
                isinstance(layer_spec, int)
                for layer_spec in config.model.decoder_layers
            )
        ):
            error = "Decoder layers must be a list of tuples"
            raise TypeError(error)

        encoder_layers = cast(
            "list[tuple[int, int, int, int]]",
            config.model.encoder_layers,
        )
        decoder_layers = cast(
            "list[tuple[int, int, int, int]]",
            config.model.decoder_layers[:],
        )

        encoder_sizes = conv2d_sizes(config.data.input_dim[1], encoder_layers)
        self.sizes, self.paddings = conv2d_transpose_sizes(
            encoder_sizes[-1],
            decoder_layers,
            encoder_sizes,
        )

        decoder_layers.insert(0, (decoder_layers[0][0], 0, 1, 1))
        for i in range(1, len(decoder_layers) - 1):
            decoder_layers[i] = (decoder_layers[i + 1][0], *decoder_layers[i][1:])
        decoder_layers[-1] = (config.data.input_dim[0], *decoder_layers[-1][1:])

        model_dim = config.model.model_dim
        if config.model.conditioning:
            model_dim += config.model.conditioning[1]
        if config.model.embedding:
            model_dim += config.model.embedding[1]

        self.model_input = nn.Linear(
            model_dim,
            decoder_layers[0][0] * self.sizes[0] * self.sizes[0],
        )
        self.model_unflatten = nn.Unflatten(
            1,
            (decoder_layers[0][0], self.sizes[0], self.sizes[0]),
        )
        self.model_main = SequentialConv2d(
            decoder_layers,
            config.model.activation,
            config.model.activation_parameters,
            config.model.batch_norm,
            config.model.dropout,
            output_activation=False,
            output_batch_norm=False,
            transpose=True,
            paddings=self.paddings,
        )
        final_layers: list[nn.Module] = []
        if config.model.loss == "logcosh_loss":
            final_layers.append(nn.Sigmoid())
        else:
            final_layers.append(nn.Identity())
        self.model_final = nn.Sequential(*final_layers)

    def forward(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """Forward pass."""
        i = torch.cat((x, y), dim=1) if y is not None else x
        h = self.model_input(i)
        f = self.model_unflatten(h)
        z = self.model_main(f)
        o: Tensor = self.model_final(z)
        return o


class BasicVAE(ModuleBase):
    """Basic VAE model."""

    def __init__(self, config: Configuration) -> None:
        """Initialize the module."""
        super().__init__(config)
        if config.model.conditioning is None and config.model.embedding is None:
            error = "BasicVAE requires a conditioning or an embedding layer"
            raise ValueError(error)

        self.input_dim = config.data.input_dim
        self.flat_input_dim = config.data.flat_input_dim

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.conditioning: nn.Module
        if config.model.conditioning:
            self.conditioning = nn.Linear(*config.model.conditioning)
            example_condition = torch.Tensor(
                config.data.batch_size,
                config.model.conditioning[0],
            )
        if config.model.embedding:
            self.conditioning = nn.Embedding(*config.model.embedding)
            example_condition = torch.randint(
                0,
                config.model.embedding[0],
                (config.data.batch_size,),
            )

        self.loss_function_ref = getattr(nn.functional, config.model.loss)

        self.example_input_array = (
            torch.Tensor(config.data.batch_size, *config.data.input_dim)
            if isinstance(config.data.input_dim, list)
            else torch.Tensor(config.data.batch_size, config.data.input_dim),
            example_condition,
        )

        self.save_hyperparameters()

    def loss_function(
        self,
        x: Tensor,
        x_hat: Tensor,
        mu: Tensor,
        log_var: Tensor,
    ) -> Tensor:
        """VAE loss function."""
        reproduction_loss: Tensor = self.loss_function_ref(
            x_hat,
            x,
            reduction="mean",
        )
        kld: Tensor = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return reproduction_loss + kld

    def process_loss(self, batch: Tensor) -> Tensor:
        """Process the loss of a batch."""
        x, y = batch
        y = self.conditioning(y)
        x = x.view(x.size(0), self.flat_input_dim)
        mu, log_var = self.encoder(x)
        z = self.encoder.reparameterization(mu, log_var)
        x_hat: Tensor = self.decoder(z, y)
        return self.loss_function(x, x_hat, mu, log_var)

    def forward(self, *args: Tensor) -> Tensor:
        """Forward pass."""
        x, y = args[0], args[1]
        x = x.view(x.size(0), self.flat_input_dim)
        y = self.conditioning(y)
        mu, log_var = self.encoder(x)
        z = self.encoder.reparameterization(mu, log_var)
        x_hat: Tensor = self.decoder(z, y)
        return x_hat

    def training_step(self, batch: Tensor, _batch_idx: int) -> Tensor:
        """Run training step."""
        loss = self.process_loss(batch)

        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch: Tensor, _batch_idx: int) -> Tensor:
        """Run validation step."""
        loss = self.process_loss(batch)

        self.log("val_loss", loss, sync_dist=True)
        return loss

    def test_step(self, batch: Tensor, _batch_idx: int) -> Tensor:  # noqa: PT019
        """Run test step."""
        loss = self.process_loss(batch)

        self.log("test_loss", loss, sync_dist=True)
        return loss

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        conditions: Tensor | None = None,
    ) -> Tensor:
        """Generate model output based on class."""
        if (
            self.config.model.conditioning is None
            and self.config.model.embedding is None
        ):
            error = "BasicVAE requires a conditioning or an embedding layer"
            raise RuntimeError(error)

        if conditions is None:
            error = "BasicVAE requires conditions"
            raise RuntimeError(error)

        conditions = conditions.to(self.device)
        z = torch.randn((batch_size, self.config.model.model_dim)).to(self.device)
        y = self.conditioning(conditions)
        res: Tensor = self.decoder(z, y)
        if isinstance(self.input_dim, list):
            res = res.view(batch_size, *self.input_dim)
        return res


class ConvVAE(ModuleBase):
    """Convolutional VAE model with conditioning."""

    def __init__(self, config: Configuration) -> None:
        """Initialize the module."""
        super().__init__(config)
        if config.model.conditioning is None and config.model.embedding is None:
            error = "ConvVAE requires a conditioning or an embedding layer"
            raise ValueError(error)

        self.input_dim = config.data.input_dim
        self.flat_input_dim = config.data.flat_input_dim

        self.encoder = EncoderConv2d(config)
        self.decoder = DecoderConv2d(config)
        self.conditioning: nn.Module
        if config.model.conditioning:
            self.conditioning = nn.Linear(*config.model.conditioning)
            example_condition = torch.Tensor(
                config.data.batch_size,
                1,
                config.model.conditioning[0],
            )
        if config.model.embedding:
            self.conditioning = nn.Embedding(*config.model.embedding)
            example_condition = torch.randint(
                0,
                config.model.embedding[0],
                (config.data.batch_size,),
            )

        if config.model.loss == "logcosh_loss":
            self.alpha = config.model.loss_parameters[0]
            self.beta = config.model.loss_parameters[1]
            self.loss_function_ref = self.logcosh_loss
        else:
            self.loss_function_ref = getattr(nn.functional, config.model.loss)
            self.beta = 1.0

        self.example_input_array = (
            torch.Tensor(config.data.batch_size, 1, *config.data.input_dim)
            if isinstance(config.data.input_dim, list)
            else torch.Tensor(config.data.batch_size, 1, config.data.input_dim),
            example_condition,
        )

        self.save_hyperparameters()

    def logcosh_loss(
        self,
        x: Tensor,
        x_hat: Tensor,
        **_kwargs: str,
    ) -> Tensor:
        """Logcosh loss function."""
        t = x_hat - x
        loss: Tensor = (
            self.alpha * t
            + torch.log(1.0 + torch.exp(-2 * self.alpha * t))
            - torch.log(torch.tensor(2.0))
        )
        return (1.0 / self.alpha) * loss.mean()

    def loss_function(
        self,
        x: Tensor,
        x_hat: Tensor,
        mu: Tensor,
        log_var: Tensor,
    ) -> Tensor:
        """VAE loss function."""
        reproduction_loss: Tensor = self.loss_function_ref(
            x_hat,
            x,
            reduction="mean",
        )
        kld: Tensor = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        # TODO: this does not seem to work...
        # kld: Tensor = torch.mean(
        #     -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1),
        #     dim=0,
        # )
        return reproduction_loss + self.beta * kld

    def process_loss(self, batch: Tensor) -> Tensor:
        """Process the loss of a batch."""
        x, y = batch
        y = self.conditioning(y)
        x = x.view(x.size(0), *self.input_dim)
        mu, log_var = self.encoder(x)
        z = self.encoder.reparameterization(mu, log_var)
        x_hat: Tensor = self.decoder(z, y)
        return self.loss_function(x, x_hat, mu, log_var)

    def forward(self, *args: Tensor) -> Tensor:
        """Forward pass."""
        x, y = args[0], args[1]
        x = x.view(x.size(0), *cast("list[int]", self.input_dim))
        y = self.conditioning(y)
        mu, log_var = self.encoder(x)
        z = self.encoder.reparameterization(mu, log_var)
        x_hat: Tensor = self.decoder(z, y)
        return x_hat

    def training_step(self, batch: Tensor, _batch_idx: int) -> Tensor:
        """Run training step."""
        loss = self.process_loss(batch)

        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch: Tensor, _batch_idx: int) -> Tensor:
        """Run validation step."""
        loss = self.process_loss(batch)

        self.log("val_loss", loss, sync_dist=True)
        return loss

    def test_step(self, batch: Tensor, _batch_idx: int) -> Tensor:  # noqa: PT019
        """Run test step."""
        loss = self.process_loss(batch)

        self.log("test_loss", loss, sync_dist=True)
        return loss

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        conditions: Tensor | None = None,
    ) -> Tensor:
        """Generate model output based on class."""
        if (
            self.config.model.conditioning is None
            and self.config.model.embedding is None
        ):
            error = "ConvVAE requires a conditioning or an embedding layer"
            raise RuntimeError(error)

        if conditions is None:
            error = "ConvVAE requires conditions"
            raise RuntimeError(error)

        conditions = conditions.to(self.device)
        z = torch.randn((batch_size, self.config.model.model_dim)).to(self.device)
        y = self.conditioning(conditions)
        res: Tensor = self.decoder(z, y)
        if isinstance(self.input_dim, list):
            res = res.view(batch_size, *self.input_dim)
        return res
