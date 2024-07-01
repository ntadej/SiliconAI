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
            torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim),
        )

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        encoding[:, 0::2] = torch.sin(positions * division_term)
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        encoding[:, 1::2] = torch.cos(positions * division_term)

        # saving buffer (same as parameter without gradients needed)
        encoding = encoding.unsqueeze(0).transpose(0, 1)
        # TODO: figure out why this breaks distributed training
        # self.register_buffer("positional_encoding", encoding)
        self.positional_encoding = nn.Parameter(encoding, requires_grad=False)

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

        # store the list of discreet input dimensions
        self.input_dim_discreet: list[int]
        if isinstance(config.data.input_dim, int):
            self.input_dim_discreet = [config.data.input_dim]
        else:
            self.input_dim_discreet = config.data.input_dim
        # store the continuous input dimensions
        self.input_dim_continuous: int = len(config.data.columns_float)

        # check if encoder and decoder layers are integers
        if not isinstance(
            config.model.encoder_layers,
            int,
        ) or not isinstance(
            config.model.decoder_layers,
            int,
        ):
            error = "Encoder and decoder layers must be integers."
            raise TypeError(error)

        # cache the model parameters
        self.model_dim = config.model.model_dim
        self.has_decoder = config.model.decoder_layers > 0

        # setup the transformer
        self.transformer = nn.Transformer(
            d_model=config.model.model_dim,
            nhead=config.model.heads,
            dim_feedforward=config.model.feedforward_dim,
            num_encoder_layers=config.model.encoder_layers,
            num_decoder_layers=config.model.decoder_layers,
            dropout=config.model.dropout,
            batch_first=True,
        )

        # setup positional encoding
        self.positional_encoder = PositionalEncoding(
            model_dim=config.model.model_dim,
            dropout=config.model.dropout,
            max_len=config.data.batch_size,
        )

        # setup embedding layers
        for i, input_dim in enumerate(self.input_dim_discreet):
            setattr(
                self,
                f"embedding_{i}",
                nn.Embedding(input_dim, config.model.model_dim),
            )
        if self.input_dim_continuous:
            self.embedding_continuous = nn.Linear(
                self.input_dim_continuous,
                config.model.model_dim,
            )

        # setup the output layer
        # should have the same number of dimensions as the input
        self.output = nn.Linear(
            config.model.model_dim,
            sum(self.input_dim_discreet) + self.input_dim_continuous,
        )

        # setup the loss function for continuous features
        self.loss_function_ref = getattr(nn.functional, config.model.loss)

        # save the hyperparameters
        self.save_hyperparameters()

    @staticmethod
    def create_pad_mask(
        matrix: Tensor,
        dtype: torch.dtype,
        pad_token: int = 0,
    ) -> Tensor:
        """Create a padding mask."""
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        mask = (
            matrix[:, :, 0] == pad_token
        )  # with multiple features just take the first one for now
        return torch.zeros_like(mask, dtype=dtype).masked_fill_(
            mask,
            float("-inf"),
        )

    def loss_function(
        self,
        x_int: Tensor,
        x_float: Tensor | None,
        x_hat: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Calculate transformer loss."""
        reduction = "mean"

        # calculate the loss for the discrete features
        loss_int: Tensor = torch.nn.functional.cross_entropy(
            x_hat[:, :, : self.input_dim_discreet[0]].permute(0, 2, 1),
            x_int[:, :, 0],
            reduction=reduction,
        )
        index = self.input_dim_discreet[0]

        for i in range(1, len(self.input_dim_discreet)):
            loss_int += torch.nn.functional.cross_entropy(
                x_hat[:, :, index : index + self.input_dim_discreet[i]].permute(
                    0,
                    2,
                    1,
                ),
                x_int[:, :, i],
                reduction=reduction,
            )
            index += self.input_dim_discreet[i]

        # calculate the loss for the continuous features
        loss_float: Tensor | None = None
        if self.input_dim_continuous and x_float is not None:
            loss_float = self.loss_function_ref(
                x_hat[:, :, index : index + self.input_dim_continuous],
                x_float,
                reduction=reduction,
            )

        # total loss
        reproduction_loss: Tensor = (
            loss_int + loss_float if loss_float is not None else loss_int
        )

        return reproduction_loss, loss_int, loss_float

    def forward_pass(
        self,
        x_data_int: Tensor,
        x_data_float: Tensor | None,
        y_data_int: Tensor,
        y_data_float: Tensor | None,
        evaluate: bool = False,
    ) -> Tensor:
        """Process the loss of a batch."""
        # data is shaped in the form of [batch, sequence, features]

        # Embedding
        # we will always have one feature, having it separate helps with the sum
        x = self.embedding_0(x_data_int[:, :, 0]) * math.sqrt(self.model_dim)
        for i in range(1, len(self.input_dim_discreet)):
            x += getattr(self, f"embedding_{i}")(x_data_int[:, :, i]) * math.sqrt(
                self.model_dim,
            )
        if self.input_dim_continuous:
            x += self.embedding_continuous(x_data_float) * math.sqrt(self.model_dim)
        # all the embeddings are summed up before positional encoding
        x = self.positional_encoder(x)

        if self.has_decoder:  # only process target data if we have a decoder
            y = self.embedding_0(y_data_int[:, :, 0]) * math.sqrt(self.model_dim)
            for i in range(1, len(self.input_dim_discreet)):
                y += getattr(self, f"embedding_{i}")(y_data_int[:, :, i]) * math.sqrt(
                    self.model_dim,
                )
            if self.input_dim_continuous:
                y += self.embedding_continuous(y_data_float) * math.sqrt(self.model_dim)
            y = self.positional_encoder(y)

        # Masking
        x_mask: Tensor | None = (
            self.transformer.generate_square_subsequent_mask(
                x_data_int.size(1),
            ).to(self.device)
            if not evaluate
            else None
        )
        x_padding_mask: Tensor = self.create_pad_mask(x_data_int, dtype=x.dtype)

        x_transformer: Tensor
        if self.has_decoder:
            y_mask: Tensor | None = (
                self.transformer.generate_square_subsequent_mask(
                    y_data_int.size(1),
                ).to(self.device)
                if not evaluate
                else None
            )
            y_padding_mask: Tensor = self.create_pad_mask(y_data_int, dtype=y.dtype)

            # in case of having both encoder and decoder run the full transformer
            x_transformer = self.transformer(
                x,
                y,
                tgt_mask=y_mask,
                src_key_padding_mask=x_padding_mask,
                tgt_key_padding_mask=y_padding_mask,
            )
        else:
            # in case of having only encoder run the encoder directly
            x_transformer = self.transformer.encoder(
                x,
                mask=x_mask,
                src_key_padding_mask=x_padding_mask,
            )

        x_hat: Tensor = self.output(x_transformer)
        return x_hat

    def process_loss(self, batch: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        """Process the loss of a batch."""
        x_data_int, x_data_float, y_data_int, y_data_float = batch
        x_hat = self.forward_pass(x_data_int, x_data_float, y_data_int, y_data_float)
        return self.loss_function(y_data_int, y_data_float, x_hat)

    def forward(self, *args: Tensor) -> Tensor:
        """Forward pass."""
        if self.has_decoder:
            x_data_int, x_data_float, y_data_int, y_data_float = (
                args[0],
                args[1],
                args[2],
                args[3],
            )
        else:
            x_data_int, x_data_float = args[0], args[1]

        return self.forward_pass(
            x_data_int,
            x_data_float,
            y_data_int if self.has_decoder else x_data_int,
            y_data_float if self.has_decoder else x_data_float,
            evaluate=True,
        )

    def training_step(self, batch: Tensor, _batch_idx: int) -> Tensor:
        """Run training step."""
        loss, loss_int, loss_float = self.process_loss(batch)

        self.log("train_loss", loss, sync_dist=True)
        self.log("train_loss_int", loss_int, sync_dist=True)
        if loss_float is not None:
            self.log("train_loss_float", loss_float, sync_dist=True)
        return loss

    def validation_step(self, batch: Tensor, _batch_idx: int) -> Tensor:
        """Run validation step."""
        loss, loss_int, loss_float = self.process_loss(batch)

        self.log("val_loss", loss, sync_dist=True)
        self.log("val_loss_int", loss_int, sync_dist=True)
        if loss_float is not None:
            self.log("val_loss_float", loss_float, sync_dist=True)
        return loss

    def test_step(self, batch: Tensor, _batch_idx: int) -> Tensor:  # noqa: PT019
        """Run test step."""
        loss, loss_int, loss_float = self.process_loss(batch)

        self.log("test_loss", loss, sync_dist=True)
        self.log("test_loss_int", loss_int, sync_dist=True)
        if loss_float is not None:
            self.log("test_loss_float", loss_float, sync_dist=True)
        return loss

    @torch.no_grad()
    def predict(
        self,
        input_sequence_int: Tensor,
        input_sequence_float: Tensor,
        end_token: int,
    ) -> tuple[Tensor, Tensor]:
        """Run predictions on the model."""
        # _rich_traceback_guard = True

        end_tensor = torch.tensor([0, end_token]).to(self.device)
        input_tensor_int, input_tensor_float = input_sequence_int, input_sequence_float

        for _ in range(21):  # TODO: make this a parameter
            pred = self(input_tensor_int, input_tensor_float)

            next_items_int = []
            index = 0
            for dim in self.input_dim_discreet:
                next_items_int.append(
                    pred[:, -1:, index : index + dim].topk(1)[1],
                )
                index += dim

            next_item_int = torch.concat(next_items_int, dim=-1)
            next_item_float = pred[:, -1:, index : index + self.input_dim_continuous]

            # Concatenate previous input with predicted best word
            input_tensor_int = torch.cat((input_tensor_int, next_item_int), dim=1)
            input_tensor_float = torch.cat((input_tensor_float, next_item_float), dim=1)

            # Stop if model predicts end of sentence
            if torch.all(torch.isin(next_item_int[:, :, 0].view(-1), end_tensor)):
                break

        return (input_tensor_int, input_tensor_float)
