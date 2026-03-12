"""
JsonlExporter — writes one TraceRecord per line to a local .jsonl file.

Zero external dependencies. Works offline. Safe for production use.
"""
from __future__ import annotations

import os
from pathlib import Path

from contrail.exporters.base import BaseExporter
from contrail.record import TraceRecord


class JsonlExporter(BaseExporter):
    """
    Appends TraceRecords as JSON lines to a local file.

    Args:
        export_path: Directory where the .jsonl file will be written.
            File is named ``contrail_traces_<date>.jsonl``.
    """

    def __init__(self, export_path: str = "./contrail_traces") -> None:
        self.export_path = Path(export_path)
        self._file = None

    async def write(self, record: TraceRecord) -> None:
        """Append a TraceRecord as a JSON line."""
        raise NotImplementedError("Phase 4")

    async def close(self) -> None:
        """Flush and close the output file."""
        raise NotImplementedError("Phase 4")

    def _get_file_path(self) -> Path:
        """Return the current output file path, creating directory if needed."""
        raise NotImplementedError("Phase 4")
