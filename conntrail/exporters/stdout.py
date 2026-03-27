"""
StdoutExporter — pretty-prints colour-coded TraceRecord summaries to terminal.

Stability colours:
  confident (0.0–0.25) → green
  boundary  (0.25–0.6) → amber/yellow
  fragile   (0.6–1.0)  → red
"""
from __future__ import annotations

from conntrail.exporters.base import BaseExporter
from conntrail.record import TraceRecord

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
        colour = _STABILITY_COLOUR.get(record.stability, _RESET)
        label = f"{colour}{_BOLD}{record.stability.upper()}{_RESET}"
        print(
            f"\n{_BOLD}[CONNTRAIL]{_RESET} {record.node_id} | {label} | "
            f"entropy={record.entropy_score:.2f} | attr: {record.attribution_dimension}"
        )
        print(f"  → Route:   {record.original_route}")
        if record.counterfactual_route:
            print(f"  → Alt:     {record.counterfactual_route}")
        print(f"  → Summary: {record.plain_language_summary}")
