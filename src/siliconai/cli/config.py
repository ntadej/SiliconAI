"""Configuration utilities."""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import tomli_w

from siliconai.common.enums import DataType

from .logging import Table, config_table, error_panel, info_panel


class TyperState:
    """Execution configuration state."""

    def __init__(self) -> None:
        """Initialize configuration state."""
        self.config_file: Path = Path("config.toml")
        self.debug: bool = False


class Configuration:
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
    def load(cls, state: TyperState, full_information: bool = True) -> Configuration:
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


class TaskConfiguration:
    """Task configuration."""

    def __init__(self, location: Path, global_config: Configuration) -> None:
        """Initialize task configuration."""
        self.location: Path = location

        if not location.exists():
            task_config_missing(location)

        with location.open(mode="rb") as f:
            config = tomllib.load(f)

        match config:
            case {
                "data": dict(),
            }:
                pass
            case _:
                error = f"invalid task configuration: {config}"
                raise ValueError(error)

        self.data: DataConfiguration = DataConfiguration(config["data"], global_config)

        info_panel(self.to_table(), title="Task Configuration")
        info_panel(self.data.to_table(), title="Data Configuration")

    def to_object(self) -> dict[str, Any]:
        """Convert configuration to object."""
        return {}

    def to_table(self) -> Table:
        """Convert configuration to table."""
        table = config_table()

        table.add_row("Location:", str(self.location))

        return table


class DataConfiguration:
    """Data configuration."""

    def __init__(self, config: dict[str, Any], global_config: Configuration) -> None:
        """Initialize data configuration."""
        match config:
            case {
                "type": str(),
            }:
                pass
            case _:
                error = f"invalid task configuration: {config}"
                raise ValueError(error)

        self.type: DataType = DataType(config["type"])
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

    def to_object(self) -> dict[str, Any]:
        """Convert configuration to object."""
        return {
            "type": self.type.value,
            "conversion": self.conversion,
            "conversion_input_file": str(self.conversion_input_file),
            "conversion_output_file": str(self.conversion_output_file),
        }

    def to_table(self) -> Table:
        """Convert configuration to table."""
        table = config_table()

        table.add_row("Type:", self.type.value)
        table.add_row("Conversion needed:", str(self.conversion))
        if self.conversion:
            table.add_row()
            table.add_row("Conversion input file:", str(self.conversion_input_file))
            table.add_row("Conversion output file:", str(self.conversion_output_file))

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
