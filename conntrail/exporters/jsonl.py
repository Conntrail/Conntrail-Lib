"""
JsonlExporter — writes one TraceRecord per line to a local .jsonl file.

Zero external dependencies. Works offline. Safe for production use.
"""
from __future__ import annotations

import os
from pathlib import Path

from conntrail.exporters.base import BaseExporter
from conntrail.record import TraceRecord


class JsonlExporter(BaseExporter):
    """
    Appends TraceRecords as JSON lines to a local file.

    Args:
        export_path: Directory where the .jsonl file will be written.
            File is named ``conntrail_traces_<date>.jsonl``.
    """

    def __init__(self, export_path: str = "./conntrail_traces") -> None:
        self.export_path = Path(export_path)
        self._file = None

    async def write(self, record: TraceRecord) -> None:
        """Append a TraceRecord as a JSON line."""
        import json

        path = self._get_file_path()
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict()) + "\n")

    async def close(self) -> None:
        """No-op: file is opened/closed per write for append safety."""

    def _get_file_path(self) -> Path:
        """Return the current output file path, creating directory if needed."""
        from datetime import date

        self.export_path.mkdir(parents=True, exist_ok=True)
        return self.export_path / f"conntrail_traces_{date.today().isoformat()}.jsonl"
