"""Transformer models."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from siliconai.common.enums import ColumnType
from siliconai.ml.common.module import Module

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from siliconai.cli.config import Configuration
    from siliconai.data.tokenizers import SequenceTokenizer


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(
        self,
        model_dim: int,
        dropout: float = 0.1,
        max_seq_size: int = 25,
        batch_first: bool = True,
    ) -> None:
        """Initialize the module."""
        super().__init__()
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)

        # encoding
        positions = torch.arange(0, max_seq_size, dtype=torch.float).unsqueeze(1)
        division_term = torch.exp(  # 1000^(2i/dim_model)
            torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim),
        )

        if self.batch_first:
            encoding = torch.zeros(1, max_seq_size, model_dim)
            # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
            encoding[0, :, 0::2] = torch.sin(positions * division_term)
            # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
            encoding[0, :, 1::2] = torch.cos(positions * division_term)
        else:
            encoding = torch.zeros(max_seq_size, 1, model_dim)
            # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
            encoding[:, 0, 0::2] = torch.sin(positions * division_term)
            # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
            encoding[:, 0, 1::2] = torch.cos(positions * division_term)

        # saving buffer (same as parameter without gradients needed)
        # TODO: figure out why this breaks distributed training
        # self.register_buffer("positional_encoding", encoding)
        self.positional_encoding = nn.Parameter(encoding, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        # Residual connection + pos encoding
        if self.batch_first:
            x = x + self.positional_encoding[:, : x.size(1), :]
        else:
            x = x + self.positional_encoding[: x.size(0), :, :]

        x_hat: Tensor = self.dropout(x)
        return x_hat


class RZTXEncoderLayer(nn.Module):
    r"""RZTXEncoderLayer - an encoder layer with residual weights for faster convergece.

    This encoder layer is based on the paper
    "ReZero is All You Need: Fast Convergence at Large Depth".
    Thomas Bachlechner, Bodhisattwa Prasad Majumder, Huanru Henry Mao,
    Garrison W. Cottrell, Julian McAuley. 2020.

    Args:
    ----
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        use_res_init: Use residual initialization
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn
            an additive bias. Default: ``True``.
    Examples::
        >>> encoder_layer = RZTXEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable[[Tensor], Tensor] = F.relu,
        batch_first: bool = False,
        bias: bool = True,
    ) -> None:
        """Initialize the layer module."""
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.resweight = nn.Parameter(torch.Tensor([0]))

        self.activation: Callable[[Tensor], Tensor]
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set state."""
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)  # type: ignore

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
        ----
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: if True, the self-attention layer will be causal
                (default: False).

        """
        src_key_padding_mask = F._canonical_mask(  # noqa: SLF001
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),  # noqa: SLF001
            other_name="src_mask",
            target_type=src.dtype,
        )

        src_mask = F._canonical_mask(  # noqa: SLF001
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        # Self attention layer
        x = src
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]  # no attention weights
        x = x * self.resweight
        src = src + self.dropout1(x)

        # Pointwise FF Layer
        x = src
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x * self.resweight
        out: Tensor = src + self.dropout2(x)
        return out


class TransformerBase(Module):
    """Transformer module base."""

    def __init__(self, config: Configuration) -> None:
        """Initialize the module."""
        super().__init__(config)

        # batch first
        self.batch_first = True

        # enable concatenation of the input instead of sum
        self.cat = False

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

        # check if sequence length is non-zero
        if config.model.sequence_length <= 0:
            error = "Sequence length must be greater than zero."
            raise ValueError(error)

        # cache the model parameters
        self.model_dim = config.model.model_dim
        self.output_dim = (
            sum(config.data.input_dim)
            if isinstance(
                config.data.input_dim,
                list,
            )
            else config.data.input_dim
        ) + 1  # padding token

        # setup the transformer
        encoder_layer: nn.Module
        if config.model.transformer_residual_weights:
            encoder_layer = RZTXEncoderLayer(
                d_model=self.model_dim,
                nhead=config.model.heads,
                dim_feedforward=config.model.feedforward_dim,
                dropout=config.model.dropout,
                activation=config.model.activation,
                batch_first=self.batch_first,
            )
            encoder_norm = None
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.model_dim,
                nhead=config.model.heads,
                dim_feedforward=config.model.feedforward_dim,
                dropout=config.model.dropout,
                activation=config.model.activation,
                batch_first=self.batch_first,
            )
            encoder_norm = nn.LayerNorm(self.model_dim)
        self.encoder = nn.TransformerEncoder(
            encoder_layer,  # type: ignore
            config.model.encoder_layers,
            encoder_norm,
            enable_nested_tensor=not config.model.transformer_residual_weights,
        )

        # setup positional encoding
        self.positional_encoder = PositionalEncoding(
            model_dim=self.model_dim,
            dropout=config.model.dropout,
            max_seq_size=config.model.sequence_length,
            batch_first=self.batch_first,
        )

        # setup the output layer
        # should have the same number of dimensions as the input
        self.output = nn.Linear(self.model_dim, self.output_dim)

        # setup the loss function for continuous features
        self.loss_function_ref = getattr(nn.functional, config.model.loss)

        # save the hyperparameters
        self.save_hyperparameters()

    @staticmethod
    def create_sequence_mask(
        data: Tensor,
        device: torch.device,
        evaluate: bool = False,
    ) -> Tensor | None:
        """Create a sequence mask."""
        mask: Tensor | None = (
            nn.Transformer.generate_square_subsequent_mask(data.size(1)).to(device)
            if not evaluate
            else None
        )
        return mask

    @staticmethod
    def create_pad_mask(
        data: Tensor,
        dtype: torch.dtype,
        pad_token: int = 0,
    ) -> Tensor:
        """Create a padding mask."""
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        mask = (
            data == pad_token if len(data.shape) == 2 else data[:, :, 0] == pad_token  # noqa: PLR2004
        )  # with multiple features just take the first one for now
        return torch.zeros_like(mask, dtype=dtype).masked_fill_(
            mask,
            float("-inf"),
        )

    def process_loss(
        self,
        batch: Tensor,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Process the loss of a batch."""
        raise NotImplementedError

    def training_step(self, batch: Tensor, _batch_idx: int) -> Tensor:
        """Run training step."""
        loss, loss_int, loss_float = self.process_loss(batch)

        self.log("train_loss", loss, sync_dist=True)
        if loss_int is not None:
            self.log("train_loss_int", loss_int, sync_dist=True)
        if loss_float is not None:
            self.log("train_loss_float", loss_float, sync_dist=True)
        return loss

    def validation_step(self, batch: Tensor, _batch_idx: int) -> Tensor:
        """Run validation step."""
        loss, loss_int, loss_float = self.process_loss(batch)

        self.log("val_loss", loss, sync_dist=True)
        if loss_int is not None:
            self.log("val_loss_int", loss_int, sync_dist=True)
        if loss_float is not None:
            self.log("val_loss_float", loss_float, sync_dist=True)
        return loss

    def test_step(self, batch: Tensor, _batch_idx: int) -> Tensor:  # noqa: PT019
        """Run test step."""
        loss, loss_int, loss_float = self.process_loss(batch)

        self.log("test_loss", loss, sync_dist=True)
        if loss_int is not None:
            self.log("test_loss_int", loss_int, sync_dist=True)
        if loss_float is not None:
            self.log("test_loss_float", loss_float, sync_dist=True)
        return loss


class DiscreteTransformer(TransformerBase):
    """Discrete transformer model."""

    def __init__(self, config: Configuration) -> None:
        """Initialize the module."""
        super().__init__(config)

        # store the list of discreet input dimensions
        self.input_dim: list[int]
        if isinstance(config.data.input_dim, int):
            self.input_dim = [config.data.input_dim]
        else:
            self.input_dim = config.data.input_dim

        # setup embedding layers
        for i, input_dim in enumerate(self.input_dim):
            setattr(
                self,
                f"embedding_{i}",
                nn.Embedding(input_dim, config.model.model_dim),
            )

    def loss_function(
        self,
        x: Tensor,
        x_hat: Tensor,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Calculate transformer loss."""
        reduction = "mean"
        index = 0

        # calculate the loss for the discrete features
        loss = torch.nn.functional.cross_entropy(
            x_hat[:, :, : self.input_dim[0]].permute(0, 2, 1),
            x[:, :, 0],
            reduction=reduction,
        )
        index = self.input_dim[0]

        for i in range(1, len(self.input_dim)):
            loss += torch.nn.functional.cross_entropy(
                x_hat[:, :, index : index + self.input_dim[i]].permute(0, 2, 1),
                x[:, :, i],
                reduction=reduction,
            )
            index += self.input_dim[i]

        return loss, loss, None

    def embedding(self, data: Tensor) -> Tensor:
        """Do the embedding."""
        # we will always have one feature, having it separate helps with the sum
        components: list[Tensor] = []
        components.append(
            self.embedding_0(data[:, :, 0]) * math.sqrt(self.model_dim),
        )
        components += [
            getattr(self, f"embedding_{i}")(data[:, :, i]) * math.sqrt(self.model_dim)
            for i in range(1, len(self.input_dim))
        ]

        # all the embeddings are summed up or concatenated before positional encoding
        if self.cat:
            data = torch.cat(components, dim=2)
        else:
            data = torch.clone(components[0])
            for i in range(1, len(components)):
                data += components[i]

        return data

    def forward_pass(
        self,
        x_data: Tensor,
        evaluate: bool = False,
    ) -> Tensor:
        """Process the loss of a batch."""
        # data is shaped in the form of [batch, sequence, features]

        # Embedding
        x = self.embedding(x_data)
        x = self.positional_encoder(x)

        # Masking
        x_mask: Tensor | None = self.create_sequence_mask(x_data, self.device, evaluate)
        x_padding_mask: Tensor = self.create_pad_mask(x_data, dtype=x.dtype)

        # Transformer
        x_transformer = self.encoder(
            x,
            mask=x_mask,
            src_key_padding_mask=x_padding_mask,
            is_causal=not evaluate,
        )

        # Output layer
        x_hat: Tensor = self.output(x_transformer)
        return x_hat

    def process_loss(
        self,
        batch: Tensor,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Process the loss of a batch."""
        x_data, y_data = batch
        x_hat = self.forward_pass(x_data)
        return self.loss_function(y_data, x_hat)

    def forward(self, *args: Tensor) -> Tensor:
        """Forward pass."""
        x_data = args[0]
        return self.forward_pass(x_data, evaluate=True)

    @torch.no_grad()
    def predict(
        self,
        input_sequence: Tensor,
        end_token: int,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Run predictions on the model."""
        # _rich_traceback_guard = True

        end_tensor = torch.tensor([0, end_token]).to(self.device)
        input_tensor = input_sequence

        for _ in range(self.config.model.sequence_length):
            pred = self(input_tensor)

            next_items = []
            index = 0
            for dim in self.input_dim:
                next_items.append(
                    pred[:, -1:, index : index + dim].topk(1)[1],
                )
                index += dim

            next_item = torch.concat(next_items, dim=-1)

            # Concatenate previous input with predicted best word
            input_tensor = torch.cat((input_tensor, next_item), dim=1)

            # Stop if model predicts end of sentence
            if torch.all(
                torch.isin(next_item[:, :, 0].view(-1), end_tensor),
            ):
                break

        return (input_tensor, None)


class ChainTransformer(TransformerBase):
    """Chain transformer model."""

    def __init__(self, config: Configuration) -> None:
        """Initialize the module."""
        super().__init__(config)

        if isinstance(config.data.input_dim, list):
            dim = sum(config.data.input_dim)
        else:
            dim = config.data.input_dim
        dim += 1  # add padding token
        self.input_dim = dim

        # setup the embedding layer
        self.embedding = nn.Embedding(self.input_dim, config.model.model_dim)

    def loss_function(
        self,
        x: Tensor,
        x_hat: Tensor,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Calculate transformer loss."""
        reduction = "mean"

        # calculate the loss for the discrete features
        loss = torch.nn.functional.cross_entropy(
            x_hat.permute(0, 2, 1),
            x,
            reduction=reduction,
        )

        return loss, loss, None

    def forward_pass(
        self,
        x_data: Tensor,
        evaluate: bool = False,
    ) -> Tensor:
        """Process the loss of a batch."""
        # data is shaped in the form of [batch, sequence, features]

        # Embedding
        x = self.embedding(x_data) * math.sqrt(self.model_dim)
        x = self.positional_encoder(x)

        # Masking
        x_mask: Tensor | None = self.create_sequence_mask(x_data, self.device, evaluate)
        x_padding_mask: Tensor = self.create_pad_mask(x_data, dtype=x.dtype)

        # Transformer
        x_transformer = self.encoder(
            x,
            mask=x_mask,
            src_key_padding_mask=x_padding_mask,
            is_causal=not evaluate,
        )

        # Output layer
        x_hat: Tensor = self.output(x_transformer)
        return x_hat

    def process_loss(
        self,
        batch: Tensor,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Process the loss of a batch."""
        x_data, y_data = batch
        x_hat = self.forward_pass(x_data)
        return self.loss_function(y_data, x_hat)

    def forward(self, *args: Tensor) -> Tensor:
        """Forward pass."""
        x_data = args[0]
        return self.forward_pass(x_data, evaluate=True)

    @torch.no_grad()
    def predict(
        self,
        input_sequence: Tensor,
        tokenizer: SequenceTokenizer,
    ) -> Tensor:
        """Run predictions on the model."""
        _rich_traceback_guard = True

        temperature: float = 1.0
        topk: int = 0

        end_tensor = torch.tensor(
            [
                tokenizer.dictionary.word2idx[self.config.data.padding_token],
                tokenizer.dictionary.word2idx[self.config.data.end_token],
            ],
        ).to(self.device)
        input_tensor = input_sequence

        ncolumns = len(self.config.data.columns_integer) + len(
            [c for c in self.config.data.columns_type if c == ColumnType.Numerical],
        )
        column_labels = []
        for column, column_type in zip(
            self.config.data.columns_integer,
            self.config.data.columns_type,
            strict=False,
        ):
            if column_type is ColumnType.Numerical:
                column_labels.append("numerical")
                column_labels.append("numerical")
            else:
                column_labels.append(column)

        column_indices_dict = {}
        for column_label in set(column_labels):
            start_index = tokenizer.summary_dict[column_label]["start"]
            end_index = tokenizer.summary_dict[column_label]["end"]
            column_indices_dict[column_label] = torch.tensor(
                [
                    [0, *range(start_index, end_index + 1)]
                    for i in range(len(input_tensor))
                ],
            ).to(self.device)

        for i in range(self.config.model.sequence_length - input_sequence.size(1)):
            column = i % ncolumns
            column_label = column_labels[column]
            column_indices = column_indices_dict[column_label]

            pred = self(input_tensor)
            # last item
            pred_last = pred[:, -1, :]
            # column filtering
            pred_filtered = torch.gather(pred_last, 1, column_indices)
            # log-softmax
            log_probability = F.log_softmax(pred_filtered, dim=-1)
            # temperature scaling
            log_probability_scaled = log_probability.div(temperature)
            # probability
            probability_scaled = log_probability_scaled.exp()
            # topk + rescale probability
            if topk > 0:
                probability_topk, probability_indices = probability_scaled.topk(topk)
                probability_topk = F.normalize(probability_topk, p=1)
                probability_indices = torch.gather(
                    column_indices,
                    1,
                    probability_indices,
                )
            else:
                probability_topk = probability_scaled
                probability_indices = column_indices
            # run multinomial sampling
            sampled_indices = torch.multinomial(probability_topk, 1, replacement=True)
            sampled_tokens = torch.gather(probability_indices, 1, sampled_indices)

            # Concatenate previous input with predicted best word
            input_tensor = torch.cat((input_tensor, sampled_tokens), dim=1)

            # Stop if model predicts end of sentence
            if torch.all(
                torch.isin(sampled_tokens, end_tensor),
            ):
                break

        return input_tensor
