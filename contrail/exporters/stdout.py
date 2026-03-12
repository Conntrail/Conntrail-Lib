"""
StdoutExporter — pretty-prints colour-coded TraceRecord summaries to terminal.

Stability colours:
  confident (0.0–0.25) → green
  boundary  (0.25–0.6) → amber/yellow
  fragile   (0.6–1.0)  → red
"""
from __future__ import annotations

from contrail.exporters.base import BaseExporter
from contrail.record import TraceRecord

# ANSI colour codes
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

_STABILITY_COLOUR = {
    "confident": _GREEN,
    "boundary": _YELLOW,
    "fragile": _RED,
}


class StdoutExporter(BaseExporter):
    """
    Prints a colour-coded trace summary to stdout.
    Useful during development. No configuration required.
    """

    async def write(self, record: TraceRecord) -> None:
        """Print a formatted summary for one TraceRecord."""
        raise NotImplementedError("Phase 4")
