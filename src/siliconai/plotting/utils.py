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
        super().savefig(fig, **kwargs)  # type: ignore
