"""
NodeInterceptor — wraps a LangGraph node and fires contrast analysis asynchronously.

The hot path is never blocked. The original node call completes and returns
before contrast analysis begins. Trace results attach to state metadata
on the next state update cycle.
"""
from __future__ import annotations

import asyncio
import logging
import random
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Callable

from conntrail.config import ConntrailConfig

logger = logging.getLogger("conntrail")

# Per-invocation trace collector. trace_graph sets this before each ainvoke/invoke call
# so that interceptors can deposit records without touching LangGraph state.
_active_trace_list: ContextVar[list | None] = ContextVar("_active_trace_list", default=None)


class NodeInterceptor:
    """
    Wraps a LangGraph node function.

    On each call:
      1. Captures the input state.
      2. Calls the original node (hot path — not delayed).
      3. Fires contrast analysis as an asyncio task (non-blocking).
      4. Returns the original node output immediately.

    Args:
        node_fn: The original LangGraph node callable.
        node_id: The node's name in the graph (for TraceRecord.node_id).
        config: ConntrailConfig controlling sampling, export, etc.
        input_key: State key that holds the text input for contrast generation.
            Falls back to the first non-empty string in state if not found.
        route_key: State key that holds the routing decision. Auto-detected if None.
    """

    TRACE_KEY: str = "__conntrail_traces__"

    def __init__(
        self,
        node_fn: Callable,
        node_id: str,
        config: ConntrailConfig,
        input_key: str = "message",
        route_key: str | None = None,
    ) -> None:
        self.node_fn = node_fn
        self.node_id = node_id
        self.config = config
        self.input_key = input_key
        self.route_key = route_key
        self.__name__ = node_id  # preserve name for LangGraph introspection
        self._exporter = None

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the wrapped node. Fire contrast analysis if sampled.

        async_mode=True  → fire as background task (never blocks hot path, no state injection)
        async_mode=False → await analysis, inject TraceRecord into returned state
        """
        if asyncio.iscoroutinefunction(self.node_fn):
            output = await self.node_fn(state)
        else:
            output = self.node_fn(state)

        if self._should_sample():
            if self.config.async_mode:
                asyncio.create_task(self._run_contrast_analysis(state, output))
            else:
                try:
                    record = await self._build_trace_record(state, output)
                except Exception as exc:
                    logger.warning(
                        "conntrail: analysis failed for node %r: %s", self.node_id, exc
                    )
                    record = None

                if record is not None:
                    await self._get_exporter().write(record)
                    if (
                        self.config.on_alert
                        and record.entropy_score >= self.config.entropy_alert_threshold
                    ):
                        self.config.on_alert(record)
                    # Deposit into the per-invocation collector (set by trace_graph's ainvoke wrapper)
                    trace_list = _active_trace_list.get()
                    if trace_list is not None:
                        trace_list.append(record)

        return output

    def _should_sample(self) -> bool:
        """Return True if this call should be traced, based on sample_rate."""
        return random.random() < self.config.sample_rate

    async def _build_trace_record(
        self,
        input_state: dict[str, Any],
        original_output: dict[str, Any],
    ) -> "TraceRecord | None":
        """
        Run the contrast analysis pipeline and return a TraceRecord, or None on failure.
        """
        from conntrail.analyser import DivergenceAnalyser
        from conntrail.contrast import ContrastGenerator
        from conntrail.record import TraceRecord
        from conntrail.utils.providers import get_chat_model

        input_text, resolved_key = self._extract_input_text(input_state)
        if not input_text:
            logger.debug("conntrail: no input text found in state for node %r", self.node_id)
            return None

        llm = get_chat_model(self.config.contrast_model, max_tokens=512)
        gen = ContrastGenerator(llm=llm)
        contrasts = await gen.generate(input_text)

        analyser = DivergenceAnalyser()
        result = await analyser.analyse(
            self.node_fn,
            input_state,
            contrasts,
            input_key=resolved_key,
            route_key=self.route_key,
        )

        stability = TraceRecord.stability_label(result.entropy_score)
        return TraceRecord(
            trace_id=TraceRecord.make_id(),
            node_id=self.node_id,
            timestamp=datetime.now(timezone.utc),
            original_input=input_text,
            original_route=result.original_route,
            entropy_score=result.entropy_score,
            stability=stability,
            attribution_dimension=result.attribution_dimension,
            plain_language_summary=TraceRecord.build_summary(
                node_id=self.node_id,
                route=result.original_route,
                stability=stability,
                entropy=result.entropy_score,
                attribution=result.attribution_dimension,
                counterfactual=result.counterfactual_route,
            ),
            raw_contrasts=contrasts,
            raw_outputs=result.contrast_routes,
            counterfactual_route=result.counterfactual_route,
        )

    async def _run_contrast_analysis(
        self,
        input_state: dict[str, Any],
        original_output: dict[str, Any],
    ) -> None:
        """
        Fire-and-forget wrapper for async_mode=True. Builds and exports the record.
        Called via asyncio.create_task() — never raises.
        """
        try:
            record = await self._build_trace_record(input_state, original_output)
            if record is None:
                return
            await self._get_exporter().write(record)
            if (
                self.config.on_alert
                and record.entropy_score >= self.config.entropy_alert_threshold
            ):
                self.config.on_alert(record)
        except Exception as exc:
            logger.warning("conntrail: analysis failed for node %r: %s", self.node_id, exc)

    def _extract_input_text(self, state: dict[str, Any]) -> tuple[str, str]:
        """Extract the text to generate contrasts from.

        Tries self.input_key first, then falls back to the first non-empty
        string value found in state. Handles LangChain message lists by
        extracting the last human message content.

        Returns:
            (text, key_used) — the extracted text and the state key it came from.
            Returns ("", self.input_key) if no text is found.
        """
        val = state.get(self.input_key)
        if isinstance(val, str) and val:
            return val, self.input_key
        # LangChain message list (e.g. MessagesState)
        if isinstance(val, list) and val:
            text = self._extract_from_messages(val)
            if text:
                return text, self.input_key
        # Fallback: first non-empty string in state — return the actual key so
        # the analyser injects contrast variants into the correct state slot.
        for k, v in state.items():
            if isinstance(v, str) and v:
                return v, k
        return "", self.input_key

    @staticmethod
    def _extract_from_messages(messages: list) -> str:
        """Extract text from the last human message in a LangChain messages list."""
        try:
            from langchain_core.messages import HumanMessage
            # Walk backwards to find the most recent human message
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
                    return msg.content
        except ImportError:
            pass
        # Fallback: last item with a .content string attribute
        for msg in reversed(messages):
            content = getattr(msg, "content", None)
            if isinstance(content, str) and content:
                return content
        return ""

    def _get_exporter(self):
        """Lazily create and cache the exporter based on config.export_format."""
        if self._exporter is None:
            fmt = self.config.export_format
            if fmt == "stdout":
                from conntrail.exporters.stdout import StdoutExporter
                self._exporter = StdoutExporter()
            elif fmt == "jsonl":
                from conntrail.exporters.jsonl import JsonlExporter
                self._exporter = JsonlExporter(self.config.export_path)
            else:
                # langsmith — Phase 7, fall back to stdout in the meantime
                from conntrail.exporters.stdout import StdoutExporter
                self._exporter = StdoutExporter()
        return self._exporter
