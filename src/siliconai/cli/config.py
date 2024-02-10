"""Configuration utilities."""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

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

        table.add_row("Location:", str(self.location))
        table.add_row("Data path:", str(self.data_path))
        table.add_row("Output path:", str(self.output_path))

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
        self.data: DataConfiguration = DataConfiguration(config["data"], global_config)
        self.model: ModelConfiguration = ModelConfiguration(
            config["model"],
            global_config,
        )
        self.training: TrainingConfiguration = TrainingConfiguration(
            config["training"],
            global_config,
        )

        info_panel(self.to_table(), title="Task Configuration")
        info_panel(self.data.to_table(), title="Data Configuration")
        info_panel(self.model.to_table(), title="Model Configuration")
        info_panel(self.training.to_table(), title="Training Configuration")

    def to_object(self) -> dict[str, Any]:
        """Convert configuration to object."""
        return {
            "name": self.name,
        }

    def to_table(self) -> Table:
        """Convert configuration to table."""
        table = config_table()

        table.add_row("Name:", self.name)
        table.add_row("Location:", str(self.location))

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
        self.input_dim: list[int] | int = config["input_dim"]
        self.batch_size: int = config["batch_size"]
        self.conversion: bool = False
        self.conversion_input_file: Path | None = None
        self.conversion_output_file: Path | None = None

        if "conversion" in config:
            self.conversion = True
            self.conversion_input_file = (
                global_config.data_path / config["conversion"]["input"]
            )
            self.conversion_output_file = (
                global_config.data_path / config["conversion"]["output"]
            )

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
            "input_dim": self.input_dim,
            "batch_size": self.batch_size,
            "conversion": self.conversion,
            "conversion_input_file": str(self.conversion_input_file),
            "conversion_output_file": str(self.conversion_output_file),
        }

    def to_table(self) -> Table:
        """Convert configuration to table."""
        table = config_table()

        table.add_row("Type:", self.type.value)
        table.add_row("Input dimension:", str(self.input_dim))
        table.add_row("Batch size:", str(self.batch_size))
        table.add_row("Conversion needed:", str(self.conversion))
        if self.conversion:
            table.add_row()
            table.add_row("Conversion input file:", str(self.conversion_input_file))
            table.add_row("Conversion output file:", str(self.conversion_output_file))

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
                "latent_dim": int(),
                "encoder_layers": list(),
                "decoder_layers": list(),
                "activation": str(),
                "batch_norm": bool(),
                "dropout": float(),
            }:
                pass
            case _:
                error = f"invalid task configuration: {config}"
                raise ValueError(error)

        self.type: ModelType = ModelType(config["type"])
        self.latent_dim: int = int(config["latent_dim"])
        self.encoder_layers: list[int] = config["encoder_layers"]
        self.decoder_layers: list[int] = config["decoder_layers"]
        self.activation: str = config["activation"]
        self.activation_parameters: list[float] = config.get(
            "activation_parameters",
            [],
        )
        self.batch_norm: bool = config["batch_norm"]
        self.dropout: float = config["dropout"]

        self.embedding: tuple[int, int] | None = None
        if "embedding" in config:
            self.embedding = (
                config["embedding"]["class_count"],
                config["embedding"]["latent_dim"],
            )

    def to_object(self) -> dict[str, Any]:
        """Convert configuration to object."""
        return {
            "type": self.type.value,
            "latent_dim": self.latent_dim,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "activation": self.activation,
            "batch_norm": self.batch_norm,
            "dropout": self.dropout,
            "embedding": self.embedding,
        }

    def to_table(self) -> Table:
        """Convert configuration to table."""
        table = config_table()

        table.add_row("Type:", self.type.value)
        table.add_row("Latent dimension:", str(self.latent_dim))
        table.add_row("Encoder layers:", str(self.encoder_layers))
        table.add_row("Decoder layers:", str(self.decoder_layers))
        table.add_row("Activation function:", self.activation)
        if self.activation_parameters:
            table.add_row("Activation parameters:", str(self.activation_parameters))
        table.add_row("Batch normalization:", str(self.batch_norm))
        table.add_row("Dropout rate:", str(self.dropout))

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
