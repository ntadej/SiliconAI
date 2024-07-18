"""Configuration utilities."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, cast

import numpy as np
import tomli_w

from siliconai.common.enums import DataType, ModelType

from .logging import Table, config_table, error_panel, info_panel


class TyperState:
    """Execution configuration state."""

    def __init__(self) -> None:
        """Initialize configuration state."""
        self.config_file: Path = Path("config.toml")
        self.debug: bool = False


class GlobalConfiguration:
    """Global configuration."""

    def __init__(
        self,
        location: Path,
        debug: bool = False,
        full_information: bool = False,
    ) -> None:
        """Initialize configuration."""
        self.location: Path = location

        with location.open(mode="rb") as f:
            config = tomllib.load(f)

        self.debug: bool = debug
        self.data_path: Path = Path("data")
        self.output_path: Path = Path("run")

        if (
            "output" in config
            and "path" in config["output"]
            and config["output"]["path"]
        ):
            self.output_path = Path(config["output"]["path"])

        if "data" in config and "path" in config["data"] and config["data"]["path"]:
            self.data_path = Path(config["data"]["path"])

        info_panel(self.to_table(full_information), title="Global Configuration")

    def to_object(self) -> dict[str, Any]:
        """Convert configuration to object."""
        return {
            "data": {
                "path": str(self.data_path),
            },
            "output": {
                "path": str(self.output_path),
                "debug": self.debug,
            },
        }

    def to_table(self, full_information: bool = False) -> Table:
        """Convert configuration to table."""
        table = config_table()

        table.add_row("Location:", print_path(self.location))
        table.add_row("Data path:", print_path(self.data_path))
        table.add_row("Output path:", print_path(self.output_path))

        if full_information:
            table.add_row()
            table.add_row("Debug:", str(self.debug))

        return table

    @classmethod
    def load(
        cls,
        state: TyperState,
        full_information: bool = True,
    ) -> GlobalConfiguration:
        """Load configuration from CLI state."""
        if not state.config_file.exists():
            config_missing(state.config_file)

        return cls(state.config_file, state.debug, full_information)

    @classmethod
    def generate_empty(cls, location: Path) -> None:
        """Generate empty config file."""
        if location.exists():
            error_message = (
                f"Configuration file [blue]'{location}'[/blue] already exists."
            )
            raise error_panel(error_message)

        config = {
            "data": {
                "path": "data",
            },
            "output": {
                "path": "run",
            },
        }

        with location.open("wb") as f:
            tomli_w.dump(config, f)

        cls(location)


class Configuration:
    """Task configuration."""

    def __init__(self, location: Path, global_config: GlobalConfiguration) -> None:
        """Initialize task configuration."""
        self.location: Path = location

        if not location.exists():
            task_config_missing(location)

        with location.open(mode="rb") as f:
            config = tomllib.load(f)

        match config:
            case {
                "name": str(),
                "data": dict(),
                "model": dict(),
            }:
                pass
            case _:
                error = f"invalid task configuration: {config}"
                raise ValueError(error)

        self.name: str = config["name"]
        self.global_config: GlobalConfiguration = global_config
        self.output_path: Path = global_config.output_path / self.output_name
        self.data: DataConfiguration = DataConfiguration(config["data"], global_config)
        self.model: ModelConfiguration = ModelConfiguration(
            config["model"],
            global_config,
        )
        self.training: TrainingConfiguration = TrainingConfiguration(
            config["training"],
            global_config,
        )

        self._run_number: int = 0

        info_panel(self.to_table(), title="Task Configuration")
        info_panel(self.data.to_table(), title="Data Configuration")
        info_panel(self.model.to_table(), title="Model Configuration")
        info_panel(self.training.to_table(), title="Training Configuration")

    def __repr__(self) -> str:
        """Return the string representation of the configuration."""
        return self.name

    @property
    def output_name(self) -> str:
        """Return the sanitized output name."""
        return self.name.replace(" ", "_")

    def run_number(self, training: bool = False) -> int:
        """Return the run number."""
        if self._run_number:
            return self._run_number

        if not self.output_path.exists():
            self.output_path.mkdir(parents=True)

        run_path = self.output_path / "run"
        if not run_path.exists():
            with run_path.open("w") as f:
                f.write("1")
            self._run_number = 1
        elif training:
            with run_path.open("r") as f:
                self._run_number = int(f.read()) + 1
            with run_path.open("w") as f:
                f.write(str(self._run_number))
        else:
            with run_path.open("r") as f:
                self._run_number = int(f.read())

        return self._run_number

    def to_object(self) -> dict[str, Any]:
        """Convert configuration to object."""
        return {
            "name": self.name,
            "output_name": self.output_name,
            "output_path": self.output_path,
        }

    def to_table(self) -> Table:
        """Convert configuration to table."""
        table = config_table()

        table.add_row("Name:", self.name)
        table.add_row("Location:", print_path(self.location))
        table.add_row("Output name:", self.output_name)
        table.add_row("Output path:", print_path(self.output_path))

        return table


class DataConfiguration:
    """Data configuration."""

    def __init__(
        self,
        config: dict[str, Any],
        global_config: GlobalConfiguration,
    ) -> None:
        """Initialize data configuration."""
        match config:
            case {
                "type": str(),
                "input_dim": list() | int(),
                "batch_size": int(),
            }:
                pass
            case _:
                error = f"invalid task configuration: {config}"
                raise ValueError(error)

        self.type: DataType = DataType(config["type"])
        self.input_file: Path | None = None
        self.input_dim: list[int] | int = config["input_dim"]
        self.split_ratio: list[float] = config.get("split_ratio", [0.7, 0.15, 0.15])
        self.batch_size: int = config["batch_size"]
        self.workers: int = config.get("workers", 4)

        self.conversion: bool = False
        self.conversion_input_file: Path | None = None

        if "conversion" in config:
            self.conversion = True
            self.conversion_input_file = (
                global_config.data_path / config["conversion"]["input"]
            )
            self.input_file = global_config.data_path / config["conversion"]["output"]
            self.columns_integer = config["conversion"].get("columns_integer", [])
            self.columns_float = config["conversion"].get("columns_float", [])

        if "input_file" in config:
            self.input_file = global_config.data_path / config["input_file"]

        if self.columns_integer and not isinstance(self.input_dim, int):
            assert len(self.columns_integer) == len(self.input_dim)

    @property
    def flat_input_dim(self) -> int:
        """Return the flat input dimension."""
        if isinstance(self.input_dim, int):
            return self.input_dim
        return int(np.prod(self.input_dim))

    def to_object(self) -> dict[str, Any]:
        """Convert configuration to object."""
        return {
            "type": self.type.value,
            "input_file": str(self.input_file),
            "columns_integer": self.columns_integer,
            "columns_float": self.columns_float,
            "input_dim": self.input_dim,
            "split_ratio": self.split_ratio,
            "batch_size": self.batch_size,
            "workers": self.workers,
            "conversion": self.conversion,
            "conversion_input_file": str(self.conversion_input_file),
        }

    def to_table(self) -> Table:
        """Convert configuration to table."""
        table = config_table()

        table.add_row("Type:", self.type.value)
        if self.input_file:
            table.add_row("Input file:", print_path(self.input_file))
        if self.columns_integer:
            table.add_row("Integer columns:", str(self.columns_integer))
        if self.columns_float:
            table.add_row("Floating-point columns:", str(self.columns_float))
        table.add_row("Input dimension:", str(self.input_dim))
        table.add_row("Split ratio:", str(self.split_ratio))
        table.add_row("Batch size:", str(self.batch_size))
        table.add_row("Workers:", str(self.workers))
        table.add_row("Conversion needed:", str(self.conversion))
        if self.conversion:
            table.add_row()
            table.add_row(
                "Conversion input file:",
                print_path(self.conversion_input_file),
            )

        return table


class ModelConfiguration:
    """Model configuration."""

    def __init__(
        self,
        config: dict[str, Any],
        _global_config: GlobalConfiguration,
    ) -> None:
        """Initialize model configuration."""
        match config:
            case {
                "type": str(),
                "model_dim": int(),
                "encoder_layers": int() | list(),
                # "decoder_layers": int() | list(),
                "activation": str(),
                "dropout": float(),
            }:
                pass
            case _:
                error = f"invalid task configuration: {config}"
                raise ValueError(error)

        self.type: ModelType = ModelType(config["type"])
        self.sequence_length: int = config.get("sequence_length", 0)
        self.model_dim: int = int(config["model_dim"])
        self.heads: int = int(config.get("heads", 0))
        self.feedforward_dim: int = int(config.get("feedforward_dim", 0))
        self.encoder_layers: int | list[int] | list[tuple[int, int, int, int]] = (
            self.process_layers(config.get("encoder_layers", 0))
        )
        self.decoder_layers: int | list[int] | list[tuple[int, int, int, int]] = (
            self.process_layers(config.get("decoder_layers", 0))
        )
        self.transformer_residual_weights: bool = config.get(
            "residual_weights",
            False,
        )
        self.activation: str = config["activation"]
        self.activation_parameters: list[float] = config.get(
            "activation_parameters",
            [],
        )
        self.batch_norm: bool = config.get("batch_norm", False)
        self.dropout: float = config["dropout"]
        self.loss: str = config.get("loss", "binary_cross_entropy_with_logits")
        self.loss_parameters: list[float] = config.get(
            "loss_parameters",
            [],
        )

        self.conditioning: tuple[int, int] | None = None
        if "conditioning" in config:
            self.conditioning = (
                config["conditioning"]["input_dim"],
                config["conditioning"]["model_dim"],
            )

        self.embedding: tuple[int, int] | None = None
        if "embedding" in config:
            self.embedding = (
                config["embedding"]["input_dim"],
                config["embedding"]["model_dim"],
            )

    @staticmethod
    def process_layers(
        layers: int | list[int | list[int]],
    ) -> int | list[int] | list[tuple[int, int, int, int]]:
        """Process layers."""
        if isinstance(layers, int):
            return layers

        int_layers = []
        tuple_layers: list[tuple[int, int, int, int]] = []
        for layer in layers:
            if isinstance(layer, int):
                int_layers.append(layer)
            else:
                tuple_size = 4
                if len(layer) != tuple_size:
                    error = "Invalid layer configuration."
                    raise ValueError(error)
                tuple_layers.append(cast(tuple[int, int, int, int], tuple(layer)))

        if int_layers and tuple_layers:
            error = "Invalid layer configuration."
            raise ValueError(error)

        return tuple_layers if tuple_layers else int_layers

    def to_object(self) -> dict[str, Any]:
        """Convert configuration to object."""
        return {
            "type": self.type.value,
            "sequence_length": self.sequence_length,
            "model_dim": self.model_dim,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "heads": self.heads,
            "feedforward_dim": self.feedforward_dim,
            "transformer_residual_weights": self.transformer_residual_weights,
            "activation": self.activation,
            "activation_parameters": self.activation_parameters,
            "batch_norm": self.batch_norm,
            "dropout": self.dropout,
            "loss": self.loss,
            "loss_parameters": self.loss_parameters,
            "conditioning": self.conditioning,
            "embedding": self.embedding,
        }

    def to_table(self) -> Table:
        """Convert configuration to table."""
        table = config_table()

        table.add_row("Type:", self.type.value)
        if self.sequence_length:
            table.add_row("Sequence length:", str(self.sequence_length))
        table.add_row("Latent dimension:", str(self.model_dim))
        table.add_row("Encoder layers:", str(self.encoder_layers))
        table.add_row("Decoder layers:", str(self.decoder_layers))
        if self.heads:
            table.add_row("Number of heads:", str(self.heads))
        if self.feedforward_dim:
            table.add_row("Feedworward dimension:", str(self.feedforward_dim))
        if self.type in [ModelType.DiscreteTransformer]:
            table.add_row(
                "Transformer residual weights:",
                str(self.transformer_residual_weights),
            )
        table.add_row("Activation function:", self.activation)
        if self.activation_parameters:
            table.add_row("Activation parameters:", str(self.activation_parameters))
        if self.type not in [ModelType.DiscreteTransformer]:
            table.add_row("Batch normalization:", str(self.batch_norm))
        table.add_row("Dropout rate:", str(self.dropout))
        table.add_row("Loss function:", self.loss)
        if self.loss_parameters:
            table.add_row("Loss function parameters:", str(self.loss_parameters))

        if self.conditioning:
            table.add_row()
            table.add_row("Conditioning input dimension:", str(self.conditioning[0]))
            table.add_row("Conditioning latent dimension:", str(self.conditioning[1]))

        if self.embedding:
            table.add_row()
            table.add_row("Embedding class count:", str(self.embedding[0]))
            table.add_row("Embedding latent dimension:", str(self.embedding[1]))

        return table


class TrainingConfiguration:
    """Training configuration."""

    def __init__(
        self,
        config: dict[str, Any],
        _global_config: GlobalConfiguration,
    ) -> None:
        """Initialize training configuration."""
        match config:
            case {
                "epochs": int(),
                "early_stopping": int(),
                "optimizer": str(),
                "learning_rate": float(),
                "weight_decay": float(),
            }:
                pass
            case _:
                error = f"invalid task configuration: {config}"
                raise ValueError(error)

        self.epochs: int = int(config["epochs"])
        self.early_stopping: int = int(config["early_stopping"])
        self.learning_rate: float = float(config["learning_rate"])
        self.weight_decay: float = float(config["weight_decay"])
        self.optimizer: str = config["optimizer"]

    def to_object(self) -> dict[str, Any]:
        """Convert configuration to object."""
        return {
            "epochs": self.epochs,
            "early_stopping": self.early_stopping,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }

    def to_table(self) -> Table:
        """Convert configuration to table."""
        table = config_table()

        table.add_row("Epochs:", str(self.epochs))
        table.add_row("Early stopping patience:", str(self.early_stopping))
        table.add_row("Optimizer:", self.optimizer)
        table.add_row("Learning rate:", str(self.learning_rate))
        table.add_row("Weight decay:", str(self.weight_decay))

        return table


def config_missing(config_file: Path) -> None:
    """Print config missing message."""
    error_message = (
        f"Configuration file [blue]'{config_file}'[/blue] does not exist.\n"
        "Please run"
        " [blue]'siliconai config [bold]--generate[/bold]'[/blue]"
        " to generate it.\n"
        "Optionally you can specify the path using the"
        " [blue]'[bold]--global-config[/bold]'[/blue] option"
        " or using the environment variable"
        " [blue bold]SILICONAI_GLOBAL_CONFIG[/blue bold].]"
    )
    raise error_panel(error_message)


def task_config_missing(config_file: Path) -> None:
    """Print config missing message."""
    error_message = (
        f"Task configuration file [blue]'{config_file}'[/blue] does not exist.\n"
        "Optionally you can specify the path using the"
        " [blue]'[bold]--config[/bold]'[/blue] option"
        " or using the environment variable"
        " [blue bold]SILICONAI_CONFIG[/blue bold].]"
    )
    raise error_panel(error_message)


def print_path(path: Path | None) -> str:
    """Print path."""
    if not path:
        return "None"

    string = str(path)
    if string.startswith("/"):
        return string

    return f"./{string}"
