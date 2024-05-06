"""Transformer models."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from siliconai.ml.common.module import Module

if TYPE_CHECKING:
    from siliconai.cli.config import Configuration


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(
        self,
        model_dim: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ) -> None:
        """Initialize the module."""
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # encoding
        encoding = torch.zeros(max_len, model_dim)
        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        division_term = torch.exp(  # 1000^(2i/dim_model)
            torch.arange(0, model_dim, 2).float() * (-math.log(10000.0)) / model_dim,
        )

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        encoding[:, 0::2] = torch.sin(positions * division_term)
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        encoding[:, 1::2] = torch.cos(positions * division_term)

        # saving buffer (same as parameter without gradients needed)
        encoding = encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("positional_encoding", encoding)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        # Residual connection + pos encoding
        x = x + self.positional_encoding[: x.size(0), :]
        x_hat: Tensor = self.dropout(x)
        return x_hat


class Transformer(Module):
    """Basic transformer model."""

    def __init__(self, config: Configuration) -> None:
        """Initialize the module."""
        super().__init__(config)

        if not isinstance(config.data.input_dim, int):
            error = "Input dimension must be an integer."
            raise TypeError(error)

        if not isinstance(
            config.model.encoder_layers,
            int,
        ) or not isinstance(
            config.model.decoder_layers,
            int,
        ):
            error = "Encoder and decoder layers must be integers."
            raise TypeError(error)

        self.model_dim = config.model.model_dim
        self.has_decoder = config.model.decoder_layers > 0

        self.transformer = nn.Transformer(
            d_model=config.model.model_dim,
            nhead=config.model.heads,
            dim_feedforward=config.model.feedforward_dim,
            num_encoder_layers=config.model.encoder_layers,
            num_decoder_layers=config.model.decoder_layers,
            dropout=config.model.dropout,
            batch_first=True,
        )

        self.positional_encoder = PositionalEncoding(
            model_dim=config.model.model_dim,
            dropout=config.model.dropout,
        )

        # TODO: make configurable
        self.embedding = nn.Embedding(config.data.input_dim, config.model.model_dim)

        self.output = nn.Linear(config.model.model_dim, config.data.input_dim)

        self.loss_function_ref = getattr(nn.functional, config.model.loss)

        self.save_hyperparameters()

    def loss_function(self, x: Tensor, x_hat: Tensor) -> Tensor:
        """Calculate transformer loss."""
        reproduction_loss: Tensor = self.loss_function_ref(
            x_hat.permute(0, 2, 1),
            x,
            reduction="mean",
        )
        return reproduction_loss

    @staticmethod
    def create_pad_mask(
        matrix: Tensor,
        dtype: torch.dtype,
        pad_token: int = 0,
    ) -> Tensor:
        """Create a padding mask."""
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        mask = matrix == pad_token
        return torch.zeros_like(mask, dtype=dtype).masked_fill_(
            mask,
            float("-inf"),
        )

    def process_loss(self, batch: Tensor) -> Tensor:
        """Process the loss of a batch."""
        x_data, y_data = batch

        x = self.embedding(x_data) * math.sqrt(self.model_dim)
        x = self.positional_encoder(x)

        if self.has_decoder:
            y = self.embedding(y_data) * math.sqrt(self.model_dim)
            y = self.positional_encoder(y)

        x_mask: Tensor = self.transformer.generate_square_subsequent_mask(x.size(1))
        x_mask = x_mask.to(self.device)
        x_padding_mask: Tensor = self.create_pad_mask(x_data, dtype=x.dtype)

        x_transformer: Tensor
        if self.has_decoder:
            y_mask: Tensor = self.transformer.generate_square_subsequent_mask(y.size(1))
            y_mask = y_mask.to(self.device)
            y_padding_mask: Tensor = self.create_pad_mask(y_data, dtype=y.dtype)

            x_transformer = self.transformer(
                x,
                y,
                tgt_mask=y_mask,
                src_key_padding_mask=x_padding_mask,
                tgt_key_padding_mask=y_padding_mask,
            )
        else:
            x_transformer = self.transformer.encoder(
                x,
                mask=x_mask,
                src_key_padding_mask=x_padding_mask,
            )

        x_hat: Tensor = self.output(x_transformer)

        return self.loss_function(y_data, x_hat)

    def forward(self, *args: Tensor) -> Tensor:
        """Forward pass."""
        if self.has_decoder:
            x_data, y_data = args[0], args[1]
        else:
            x_data = args[0]

        x = self.embedding(x_data) * math.sqrt(self.model_dim)
        x = self.positional_encoder(x)

        if self.has_decoder:
            y = self.embedding(y_data) * math.sqrt(self.model_dim)
            y = self.positional_encoder(y)

        # No positional masking for evaluation
        x_padding_mask: Tensor = self.create_pad_mask(x_data, dtype=x.dtype)

        x_transformer: Tensor
        if self.has_decoder:
            y_padding_mask: Tensor = self.create_pad_mask(y_data, dtype=y.dtype)

            x_transformer = self.transformer(
                x,
                y,
                src_key_padding_mask=x_padding_mask,
                tgt_key_padding_mask=y_padding_mask,
            )
        else:
            x_transformer = self.transformer.encoder(
                x,
                src_key_padding_mask=x_padding_mask,
            )

        x_hat: Tensor = self.output(x_transformer)
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

    @torch.no_grad()
    def predict(self, input_sequence: Tensor, end_token: int = 1) -> Tensor:
        """Run predictions on the model."""
        input_tensor = input_sequence

        for _ in range(20):
            pred = self(input_tensor)

            next_item = (
                pred.topk(1)[1].view(-1)[-1].item()
            )  # num with highest probability
            next_item = torch.tensor([[next_item]], device=self.device)

            # # Concatenate previous input with predicted best word
            input_tensor = torch.cat((input_tensor, next_item), dim=1)

            # Stop if model predicts end of sentence
            if next_item.view(-1).item() == end_token:
                break

        return input_tensor.view(-1)
