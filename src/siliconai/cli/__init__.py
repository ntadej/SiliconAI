"""Vremenar Utils CLI."""
import sys
from pathlib import Path
from sys import argv
from typing import Annotated

import typer

from siliconai import __version__

from .config import (
    TyperState,
    config_missing,
    generate_empty_config,
    init_config,
    init_task_config,
    print_config_file,
)
from .logging import setup_logger

if not sys.warnoptions:  # pragma: no cover
    import warnings

    warnings.simplefilter("default")

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
        "config.yml",
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
        generate_empty_config(state.config_file)
    else:
        print_config_file(state.config_file)
        init_config(state)


@application.command()
def test_gpu() -> None:
    """Test GPU support."""
    config = init_config(state)
    logger = setup_logger(config, "test_gpu")

    logger.info("Hello World!")

    import torch

    logger.info("torch.cuda.is_available() = %s", torch.cuda.is_available())


@application.command()
def prepare_inputs(
    config_file: Annotated[
        Path,
        typer.Option(
            "-c",
            "--config",
            envvar="SILICONAI_CONFIG",
            help="Task configuration file.",
        ),
    ] = Path(
        "task.yml",
    ),
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
    config = init_config(state)
    task_config = init_task_config(config, config_file)
    logger = setup_logger(config, "prepare_inputs")

    from siliconai.data.input import InputLoader

    loader = InputLoader(
        logger,
        task_config.input_type,
        task_config.input_file,
        task_config.output_file,
    )
    loader.load()

    if diagnostics:
        loader.diagnostics()
