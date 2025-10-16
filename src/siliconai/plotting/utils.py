# Copyright (C) 2024 Tadej Novak
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# SPDX-License-Identifier: MPL-2.0

"""Plotting utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from matplotlib.backends.backend_pdf import PdfPages

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class PDFDocument(PdfPages):
    """PDF document helper class that can be used as a context manager."""

    def save(self, fig: Figure, **kwargs: dict[str, Any]) -> None:
        """Save a figure to the PDF document."""
        super().savefig(fig, **kwargs)  # type: ignore[no-untyped-call]
