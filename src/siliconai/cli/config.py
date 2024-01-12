"""Configuration utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .logging import error_panel, info_panel


class TyperState:
    """Execution configuration state."""

    def __init__(self: TyperState) -> None:
        """Initialize configuration state."""
        self.config_file: Path = Path("config.yml")
        self.debug: bool = False


class Configuration:
    """Configuration helper."""

    def __init__(self: Configuration) -> None:
        """Initialize configuration helper."""
        self.debug: bool = False
        self.data_path: Path = Path("data")
        self.log_path: Path = Path("run")

    def to_object(self: Configuration) -> dict[str, Any]:
        """Convert configuration to object."""
        return {
            "Data": {
                "path": str(self.data_path),
            },
            "Logging": {
                "path": str(self.log_path),
                "debug": self.debug,
            },
        }


def config_missing(config_file: Path) -> None:
    """Print config missing message."""
    error_message = (
        f"Configuration file [blue]'{config_file}'[/blue] does not exist.\n"
        "Please run"
        " [blue]'siliconai config [bold]--generate[/bold]'[/blue]"
        " to generate it.\n"
        "Optionally you can specify the path using the"
        " [blue]'[bold]--config[/bold]'[/blue] option"
        " or using the environment variable"
        " [blue bold]SILICONAI_CONFIG[/blue bold].]"
    )
    raise error_panel(error_message)


def generate_empty_config(config_file: Path) -> None:
    """Generate empty config file."""
    if config_file.exists():
        error_message = (
            f"Configuration file [blue]'{config_file}'[/blue] already exists."
        )
        raise error_panel(error_message)

    config = {
        "logging": {
            "path": "run",
        },
        "data": {
            "path": "data",
        },
    }

    with config_file.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print_config_file(config_file)


def print_config_file(config_file: Path) -> None:
    """Print config file content."""
    if not config_file.exists():
        config_missing(config_file)

    with config_file.open() as f:
        config = yaml.safe_load(f)

    info_panel(
        yaml.dump(config, default_flow_style=False, sort_keys=False).strip("\n"),
        title=f"Configuration file: [bold]{config_file}[/bold]",
    )


def init_config(state: TyperState) -> Configuration:
    """Initialise configuration from CLI state."""
    if not state.config_file.exists():
        config_missing(state.config_file)

    with state.config_file.open() as f:
        config = yaml.safe_load(f)

    configuration = Configuration()
    if (
        "logging" in config
        and "path" in config["logging"]
        and config["logging"]["path"]
    ):
        configuration.log_path = Path(config["logging"]["path"])

    if "data" in config and "path" in config["data"] and config["data"]["path"]:
        configuration.data_path = Path(config["data"]["path"])

    if state:
        configuration.debug = state.debug

    info_panel(
        yaml.dump(
            configuration.to_object(),
            default_flow_style=False,
            sort_keys=False,
        ).strip("\n"),
        title="Configuration",
    )

    return configuration
