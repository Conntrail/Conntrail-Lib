"""
NodeInterceptor — wraps a LangGraph node and fires contrast analysis asynchronously.

The hot path is never blocked. The original node call completes and returns
before contrast analysis begins. Trace results attach to state metadata
on the next state update cycle.
"""
from __future__ import annotations

from typing import Any, Callable

from contrail.config import ContrailConfig


class NodeInterceptor:
    """
    Wraps a LangGraph node function.

    On each call:
      1. Captures the input state.
      2. Calls the original node (hot path — not delayed).
      3. Fires contrast analysis as an asyncio task (non-blocking).
      4. Returns the original node output immediately.

    TraceRecords are attached to ``state["__contrail_traces__"]`` when
    the async task completes.

    Args:
        node_fn: The original LangGraph node callable.
        node_id: The node's name in the graph (for TraceRecord.node_id).
        config: ContrailConfig controlling sampling, export, etc.
    """

    TRACE_KEY: str = "__contrail_traces__"

    def __init__(
        self,
        node_fn: Callable,
        node_id: str,
        config: ContrailConfig,
    ) -> None:
        self.node_fn = node_fn
        self.node_id = node_id
        self.config = config
        self.__name__ = node_id  # preserve name for LangGraph introspection

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the wrapped node. Fire contrast analysis async if sampled.

        Returns the original node output without modification.
        """
        raise NotImplementedError("Phase 5")

    def _should_sample(self) -> bool:
        """Return True if this call should be traced, based on sample_rate."""
        raise NotImplementedError("Phase 5")

    async def _run_contrast_analysis(
        self,
        input_state: dict[str, Any],
        original_output: dict[str, Any],
    ) -> None:
        """
        Run the full contrast analysis pipeline async.
        Attaches the resulting TraceRecord via the configured exporter.
        This method is always called via asyncio.create_task().
        """
        raise NotImplementedError("Phase 5")
