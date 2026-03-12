"""
LangSmithExporter — pushes TraceRecords as LangSmith spans.

Requires: pip install contrail[langsmith]
Requires: LANGSMITH_API_KEY environment variable.
"""
from __future__ import annotations

from contrail.exporters.base import BaseExporter
from contrail.record import TraceRecord


class LangSmithExporter(BaseExporter):
    """
    Wraps TraceRecord into a LangSmith-compatible span and pushes via
    the LangSmith SDK. Adds all Contrail fields as span metadata.

    Args:
        project_name: LangSmith project to push traces to.
    """

    def __init__(self, project_name: str = "contrail") -> None:
        self.project_name = project_name
        self._client = None  # lazy-initialised on first write

    async def write(self, record: TraceRecord) -> None:
        """Push a TraceRecord to LangSmith as a span."""
        raise NotImplementedError("Phase 7")
