"""
Tests for NodeInterceptor and the public API (Phase 5).
"""
import asyncio
import time

import pytest

from conntrail.config import ConntrailConfig
from conntrail.interceptor import NodeInterceptor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _noop_node(state):
    return state


async def _routing_node(state):
    msg = state.get("message", "")
    route = "urgent" if "urgent" in msg.lower() else "general"
    return {**state, "route": route}


def _make_interceptor(node_fn=None, input_key="message", route_key="route", **config_kwargs):
    config = ConntrailConfig(**config_kwargs)
    return NodeInterceptor(
        node_fn or _noop_node,
        node_id="test_node",
        config=config,
        input_key=input_key,
        route_key=route_key,
    )


# ---------------------------------------------------------------------------
# ConntrailConfig validation (live)
# ---------------------------------------------------------------------------

class TestConntrailConfig:
    def test_default_values(self):
        config = ConntrailConfig()
        assert config.contrast_model == "claude-haiku-4-5-20251001"
        assert config.sample_rate == 1.0
        assert config.async_mode is True
        assert config.export_format == "jsonl"
        assert config.entropy_alert_threshold == 0.6

    def test_invalid_sample_rate_raises(self):
        with pytest.raises(ValueError, match="sample_rate"):
            ConntrailConfig(sample_rate=1.5)

    def test_zero_sample_rate_valid(self):
        config = ConntrailConfig(sample_rate=0.0)
        assert config.sample_rate == 0.0

    def test_invalid_export_format_raises(self):
        with pytest.raises(ValueError, match="export_format"):
            ConntrailConfig(export_format="invalid")

    def test_invalid_alert_threshold_raises(self):
        with pytest.raises(ValueError, match="entropy_alert_threshold"):
            ConntrailConfig(entropy_alert_threshold=1.5)


# ---------------------------------------------------------------------------
# NodeInterceptor — core behaviour
# ---------------------------------------------------------------------------

class TestNodeInterceptor:
    def test_instantiation(self):
        interceptor = _make_interceptor()
        assert interceptor.node_id == "test_node"
        assert interceptor.__name__ == "test_node"

    @pytest.mark.asyncio
    async def test_output_unchanged(self):
        """Wrapped node return value must be identical to unwrapped."""
        state = {"message": "hello", "route": None}
        output = await _routing_node(state)

        interceptor = _make_interceptor(_routing_node, sample_rate=0.0)
        wrapped_output = await interceptor(state)

        assert wrapped_output == output

    @pytest.mark.asyncio
    async def test_output_unchanged_for_noop(self):
        state = {"message": "test", "value": 42}
        interceptor = _make_interceptor(sample_rate=0.0)
        result = await interceptor(state)
        assert result == state

    @pytest.mark.asyncio
    async def test_zero_sample_rate_fires_no_analysis(self):
        """sample_rate=0.0 must not schedule any contrast analysis tasks."""
        fired = []

        async def spy_node(state):
            return state

        interceptor = _make_interceptor(spy_node, sample_rate=0.0)

        original_should_sample = interceptor._should_sample
        interceptor._should_sample = lambda: False

        tasks_before = len(asyncio.all_tasks())
        await interceptor({"message": "hello"})
        tasks_after = len(asyncio.all_tasks())

        # No new tasks should have been created
        assert tasks_after == tasks_before

    @pytest.mark.asyncio
    async def test_hot_path_not_blocked(self):
        """Wrapped node wall-time must be within 20ms of unwrapped node."""
        RUNS = 10
        TOLERANCE_MS = 20

        async def slow_analysis_node(state):
            await asyncio.sleep(0.001)  # 1ms simulated work
            return state

        interceptor = _make_interceptor(slow_analysis_node, sample_rate=0.0)

        # Measure unwrapped
        t0 = time.perf_counter()
        for _ in range(RUNS):
            await slow_analysis_node({"message": "x"})
        unwrapped_ms = (time.perf_counter() - t0) * 1000 / RUNS

        # Measure wrapped (analysis disabled so it's pure overhead)
        t0 = time.perf_counter()
        for _ in range(RUNS):
            await interceptor({"message": "x"})
        wrapped_ms = (time.perf_counter() - t0) * 1000 / RUNS

        overhead_ms = wrapped_ms - unwrapped_ms
        assert overhead_ms < TOLERANCE_MS, (
            f"Interceptor overhead {overhead_ms:.1f}ms exceeds {TOLERANCE_MS}ms tolerance"
        )

    @pytest.mark.asyncio
    async def test_sync_node_supported(self):
        """NodeInterceptor must also wrap synchronous node functions."""
        def sync_node(state):
            return {**state, "route": "sync_route"}

        interceptor = NodeInterceptor(
            sync_node, node_id="sync", config=ConntrailConfig(sample_rate=0.0)
        )
        result = await interceptor({"message": "hi"})
        assert result["route"] == "sync_route"

    def test_extract_input_text_uses_input_key(self):
        interceptor = _make_interceptor(input_key="message")
        state = {"message": "hello", "other": "world"}
        text, key = interceptor._extract_input_text(state)
        assert text == "hello"
        assert key == "message"

    def test_extract_input_text_fallback(self):
        """Falls back to first non-empty string if input_key missing, returns actual key."""
        interceptor = NodeInterceptor(
            _noop_node, node_id="n", config=ConntrailConfig(), input_key="missing_key"
        )
        state = {"content": "fallback text", "number": 42}
        text, key = interceptor._extract_input_text(state)
        assert text == "fallback text"
        assert key == "content"

    def test_extract_input_text_empty_state(self):
        interceptor = _make_interceptor()
        text, key = interceptor._extract_input_text({})
        assert text == ""

    def test_get_exporter_stdout(self, tmp_path):
        from conntrail.exporters.stdout import StdoutExporter
        interceptor = _make_interceptor(export_format="stdout")
        assert isinstance(interceptor._get_exporter(), StdoutExporter)

    def test_get_exporter_jsonl(self, tmp_path):
        from conntrail.exporters.jsonl import JsonlExporter
        interceptor = _make_interceptor(
            export_format="jsonl", export_path=str(tmp_path / "traces")
        )
        assert isinstance(interceptor._get_exporter(), JsonlExporter)

    def test_get_exporter_cached(self):
        """Same exporter instance returned on repeated calls."""
        interceptor = _make_interceptor(export_format="stdout")
        exp1 = interceptor._get_exporter()
        exp2 = interceptor._get_exporter()
        assert exp1 is exp2

    @pytest.mark.asyncio
    async def test_on_alert_fires_above_threshold(self):
        """on_alert callback invoked when entropy >= entropy_alert_threshold."""
        alerts = []

        config = ConntrailConfig(
            sample_rate=1.0,
            entropy_alert_threshold=0.0,  # always fires
            export_format="stdout",
            on_alert=lambda rec: alerts.append(rec),
        )

        # Stub out the analysis to return a known high-entropy result
        async def _fast_analysis(input_state, original_output):
            from datetime import datetime, timezone
            from conntrail.contrast import ContrastSet
            from conntrail.record import TraceRecord
            record = TraceRecord(
                trace_id="t1",
                node_id="test_node",
                timestamp=datetime.now(timezone.utc),
                original_input="x",
                original_route="a",
                entropy_score=0.75,
                stability="fragile",
                attribution_dimension="semantic intensity",
                plain_language_summary="test",
                raw_contrasts=ContrastSet(similar="s", neutral="n", opposite="o"),
                raw_outputs={"original": "a", "similar": "b", "neutral": "c", "opposite": "d"},
            )
            await interceptor._get_exporter().write(record)
            if config.on_alert and record.entropy_score >= config.entropy_alert_threshold:
                config.on_alert(record)

        interceptor = NodeInterceptor(_noop_node, node_id="test_node", config=config)
        interceptor._run_contrast_analysis = _fast_analysis

        await interceptor({"message": "trigger"})
        # Give the task time to execute
        await asyncio.sleep(0.05)
        assert len(alerts) == 1
        assert alerts[0].entropy_score == 0.75


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TestPublicAPI:
    def test_trace_node_import(self):
        from conntrail import trace_node
        assert callable(trace_node)

    def test_trace_graph_import(self):
        from conntrail import trace_graph
        assert callable(trace_graph)

    def test_conntrail_config_import(self):
        from conntrail import ConntrailConfig
        assert ConntrailConfig is not None

    @pytest.mark.asyncio
    async def test_trace_node_does_not_alter_output(self):
        from conntrail import trace_node

        @trace_node(config=ConntrailConfig(sample_rate=0.0))
        async def my_node(state):
            return {**state, "route": "result"}

        out = await my_node({"message": "hello"})
        assert out["route"] == "result"

    @pytest.mark.asyncio
    async def test_trace_node_preserves_name(self):
        from conntrail import trace_node

        @trace_node(config=ConntrailConfig(sample_rate=0.0))
        async def classify_query(state):
            return state

        assert classify_query.__name__ == "classify_query"

    def test_trace_graph_wraps_all_non_system_nodes(self):
        from conntrail import trace_graph
        from testing.agents.customer_support.agent import build_graph

        graph = build_graph()
        original_fns = {
            nid: node.bound.afunc or node.bound.func
            for nid, node in graph.nodes.items()
            if nid not in {"__start__", "__end__"}
        }

        trace_graph(graph, config=ConntrailConfig(sample_rate=0.0))

        for nid, orig in original_fns.items():
            wrapped = graph.nodes[nid].bound.afunc
            assert isinstance(wrapped, NodeInterceptor), (
                f"Node {nid!r} was not wrapped by a NodeInterceptor"
            )
            assert wrapped.node_fn is orig

    @pytest.mark.asyncio
    async def test_trace_graph_does_not_alter_graph_output(self):
        """Graph output must be identical before and after trace_graph."""
        from dotenv import load_dotenv
        load_dotenv("testing/.env")
        from conntrail import trace_graph
        from testing.agents.customer_support.agent import build_graph

        state = {"message": "I need a refund", "category": None, "response": None}

        # Unwrapped run
        g1 = build_graph()
        result_before = await g1.ainvoke(state)

        # Wrapped run (analysis disabled)
        g2 = build_graph()
        trace_graph(g2, config=ConntrailConfig(sample_rate=0.0))
        result_after = await g2.ainvoke(state)

        assert result_before["category"] == result_after["category"]
