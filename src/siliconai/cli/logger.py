# Copyright (C) 2024 Tadej Novak
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# SPDX-License-Identifier: MPL-2.0

"""Common logging setup."""

from __future__ import annotations

from logging import DEBUG, INFO, Formatter, Logger, getLogger
from logging.handlers import RotatingFileHandler
from typing import TYPE_CHECKING, Any

from rich import print as rprint
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

if TYPE_CHECKING:
    from siliconai.cli.config import GlobalConfiguration


def config_table() -> Table:
    return Table.grid("Key", "Value", padding=(0, 3))


def info_panel(message: str | Table, title: str = "Information") -> None:
    """Print info message in a panel."""
    rprint(
        Panel(
            message,
            title=title,
            title_align="left",
            border_style=Style(color=Color.parse("blue")),
        ),
    )


def error_panel(message: str) -> Exit:
    """Print error message in a panel."""
    rprint(
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


def setup_logger(global_config: GlobalConfiguration, name: str | None = None) -> Logger:
    """Prepare logger and write the log file."""
    if not global_config.output_path.exists():
        global_config.output_path.mkdir(parents=True)

    if name:
        file_formatter = Formatter(
            "%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_path = global_config.output_path / f"siliconai_{name}.log"
        file_handler = RotatingFileHandler(
            file_path,
            mode="a",
            maxBytes=10 * 1024 * 1024,
            backupCount=3,
        )
        file_handler.setFormatter(file_formatter)

    stream_handler = RichHandler(
        show_path=global_config.debug,
        log_time_format="%Y-%m-%d %H:%M:%S",
    )

    logger = getLogger()
    if name:
        logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    if global_config.debug:  # pragma: no cover
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
