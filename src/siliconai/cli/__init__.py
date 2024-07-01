"""Vremenar Utils CLI."""

import sys
from pathlib import Path
from sys import argv
from typing import Annotated

import typer

from siliconai import __version__
from siliconai.common.enums import DataLoadingType

from .config import Configuration, GlobalConfiguration, TyperState, config_missing
from .logging import setup_logger

if not sys.warnoptions:  # pragma: no cover
    import warnings

    warnings.simplefilter("default")

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import lightning as L  # noqa: F401

application = typer.Typer()
state = TyperState()


def version_callback(value: bool) -> None:
    """Version callback."""
    if value:
        typer.echo(f"Vremenar Utils, version {__version__}")
        raise typer.Exit()


@application.callback()
def main(
    ctx: typer.Context,
    config: Annotated[
        Path,
        typer.Option(
            "-g",
            "--global-config",
            envvar="SILICONAI_GLOBAL_CONFIG",
            help="Global configuration file.",
        ),
    ] = Path(
        "config.toml",
    ),
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Run with debug printouts.",
        ),
    ] = False,
    version: Annotated[  # noqa: ARG001
        bool,
        typer.Option(
            "--version",
            help="Show version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Vremenar Utilities CLI app."""
    if ctx.invoked_subcommand != "config" and not config.exists():  # pragma: no cover
        if "--help" in argv:
            return
        config_missing(config)

    state.config_file = config
    state.debug = debug


@application.command()
def config(
    generate: Annotated[
        bool,
        typer.Option("--generate", help="Generate empty configuration."),
    ] = False,
) -> None:
    """Print or generate configuration."""
    if generate:
        GlobalConfiguration.generate_empty(state.config_file)
    else:
        GlobalConfiguration.load(state, full_information=False)


@application.command()
def test_gpu() -> None:
    """Test GPU support."""
    global_config = GlobalConfiguration.load(state)
    logger = setup_logger(global_config, "test_gpu")

    logger.info("Hello World!")

    import torch

    logger.info("torch.cuda.is_available() = %s", torch.cuda.is_available())


@application.command()
def convert_inputs(
    config_file: Annotated[
        Path,
        typer.Option(
            "-c",
            "--config",
            envvar="SILICONAI_CONFIG",
            help="Task configuration file.",
        ),
    ],
    diagnostics: Annotated[
        bool,
        typer.Option(
            "-d",
            "--diagnostics",
            help="Prepare diagnostics plots.",
        ),
    ] = False,
) -> None:
    """Prepare inputs for training."""
    global_config = GlobalConfiguration.load(state)
    config = Configuration(config_file, global_config)
    logger = setup_logger(global_config, "convert_inputs")

    if not config.data.conversion:
        logger.info("No conversion needed.")
        return

    from siliconai.data.input import InputConverter

    loader = InputConverter(logger, config.data)
    loader.load()

    if diagnostics:
        loader.diagnostics()


@application.command()
def train(
    config_file: Annotated[
        Path,
        typer.Option(
            "-c",
            "--config",
            envvar="SILICONAI_CONFIG",
            help="Task configuration file.",
        ),
    ],
    diagnostics: Annotated[
        bool,
        typer.Option(
            "-d",
            "--diagnostics",
            help="Prepare diagnostics plots.",
        ),
    ] = False,
) -> None:
    """Train the model."""
    global_config = GlobalConfiguration.load(state)
    config = Configuration(config_file, global_config)
    logger = setup_logger(global_config, "train")

    from siliconai.ml.training.training import train

    train(logger, config, diagnostics)


@application.command()
def validate(
    config_file: Annotated[
        Path,
        typer.Option(
            "-c",
            "--config",
            envvar="SILICONAI_CONFIG",
            help="Task configuration file.",
        ),
    ],
    data_type: Annotated[
        DataLoadingType,
        typer.Option(
            "-t",
            "--type",
            help="Data type to validate.",
        ),
    ] = DataLoadingType.test.value,
) -> None:
    """Validate the model."""
    global_config = GlobalConfiguration.load(state)
    config = Configuration(config_file, global_config)
    logger = setup_logger(global_config, "validate")

    from siliconai.ml.training.utils import common_setup
    from siliconai.plotting.validation import validate

    common_setup()
    validate(logger, config, data_type)
