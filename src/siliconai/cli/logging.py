"""Common logging setup."""
from __future__ import annotations

from logging import DEBUG, INFO, Formatter, Logger, getLogger
from logging.handlers import RotatingFileHandler
from typing import TYPE_CHECKING, Any

from rich import print
from rich.color import Color
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.style import Style
from rich.table import Table
from typer import Exit

if TYPE_CHECKING:  # pragma: no cover
    from .config import Configuration


def config_table() -> Table:
    return Table.grid("Key", "Value", padding=(0, 3))


def info_panel(message: str | Table, title: str = "Information") -> None:
    """Print info message in a panel."""
    print(
        Panel(
            message,
            title=title,
            title_align="left",
            border_style=Style(color=Color.parse("blue")),
        ),
    )


def error_panel(message: str) -> Exit:
    """Print error message in a panel."""
    print(
        Panel(
            message,
            title="Error",
            title_align="left",
            border_style=Style(color=Color.parse("red")),
        ),
    )
    return Exit(1)


def progress_bar(**kwargs: Any) -> Progress:  # noqa: ANN401
    """Return progress bar."""
    return Progress(
        TextColumn("[progress.description]{task.description:>27} "),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        **kwargs,
    )


def download_bar(**kwargs: Any) -> Progress:  # noqa: ANN401
    """Return download bar."""
    return Progress(
        TextColumn("[progress.description]{task.description:>27} "),
        BarColumn(bar_width=None),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        **kwargs,
    )


def setup_logger(config: Configuration, name: str | None = None) -> Logger:
    """Prepare logger and write the log file."""
    if not config.output_path.exists():
        config.output_path.mkdir(parents=True)

    if name:
        file_formatter = Formatter(
            "%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_path = config.output_path / f"siliconai_{name}.log"
        file_handler = RotatingFileHandler(
            file_path,
            mode="a",
            maxBytes=10 * 1024 * 1024,
            backupCount=3,
        )
        file_handler.setFormatter(file_formatter)

    stream_handler = RichHandler(
        show_path=config.debug,
        log_time_format="%Y-%m-%d %H:%M:%S",
    )

    logger = getLogger()
    if name:
        logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    if config.debug:  # pragma: no cover
        logger.setLevel(DEBUG)
    else:
        logger.setLevel(INFO)

    # override logging from other modules
    custom_loggers = [getLogger("lightning.pytorch"), getLogger("lightning.fabric")]
    for log in custom_loggers:
        log.handlers.clear()
        if name:
            log.addHandler(file_handler)
        log.addHandler(stream_handler)

    return logger


__all__ = ["Logger", "Table"]
