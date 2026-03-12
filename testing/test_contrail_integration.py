"""
Phase 6 integration tests — Contrail against all 4 real agents.

These run after Phase 5 is complete (NodeInterceptor + Public API).
Tests the full 6-test matrix: non-intrusion, trace completeness, entropy
calibration, attribution plausibility, latency overhead, JSONL round-trip.
"""
import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from testing.harness.assertions import (
    assert_entropy_range,
    assert_latency_overhead,
    assert_non_intrusive,
    assert_trace_record,
)
from testing.harness.fixtures import ALL_INPUTS, ENTROPY_RANGES


# --- Parametrize over all 4 agents ---
AGENT_CONFIGS = [
    {
        "name": "customer_support",
        "build_fn": "testing.agents.customer_support.agent.build_graph",
        "input_fn": lambda t: {"message": t.text, "category": None, "response": None},
    },
    {
        "name": "multi_agent_supervisor",
        "build_fn": "testing.agents.multi_agent_supervisor.agent.build_graph",
        "input_fn": lambda t: {"task": t.text, "assigned_agent": None, "result": None},
    },
    {
        "name": "adaptive_rag",
        "build_fn": "testing.agents.adaptive_rag.agent.build_graph",
        "input_fn": lambda t: {"query": t.text, "strategy": None, "retrieved_docs": [], "answer": None},
    },
]


def _import_build_fn(dotted_path: str):
    module_path, fn_name = dotted_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, fn_name)


@pytest.mark.integration
@pytest.mark.parametrize("agent_config", AGENT_CONFIGS, ids=[a["name"] for a in AGENT_CONFIGS])
class TestNonIntrusion:
    """Contrail wrapping must not alter agent output."""

    async def test_output_unchanged(self, agent_config):
        """Agent output must be identical before and after Contrail wrapping."""
        # NOTE: Requires Contrail to be implemented (Phase 5+)
        pytest.importorskip("contrail")
        from contrail import ContrailConfig, trace_graph

        build_graph = _import_build_fn(agent_config["build_fn"])
        test_inputs = ALL_INPUTS[agent_config["name"]]
        confident_input = next(t for t in test_inputs if t.category == "confident")

        baseline_graph = build_graph()
        wrapped_graph = trace_graph(
            build_graph(),
            config=ContrailConfig(sample_rate=1.0, export_format="stdout"),
        )

        input_state = agent_config["input_fn"](confident_input)

        baseline_result = await baseline_graph.ainvoke(input_state)
        wrapped_result = await wrapped_graph.ainvoke(input_state)

        # Remove contrail traces from comparison
        wrapped_clean = {k: v for k, v in wrapped_result.items() if k != "__contrail_traces__"}
        assert baseline_result == wrapped_clean, (
            f"Agent output changed after Contrail wrapping for {agent_config['name']}"
        )


@pytest.mark.integration
@pytest.mark.parametrize("agent_config", AGENT_CONFIGS, ids=[a["name"] for a in AGENT_CONFIGS])
class TestTraceCompleteness:
    """Every node call must produce a TraceRecord."""

    async def test_traces_present(self, agent_config):
        pytest.importorskip("contrail")
        from contrail import ContrailConfig, trace_graph

        build_graph = _import_build_fn(agent_config["build_fn"])
        test_inputs = ALL_INPUTS[agent_config["name"]]
        confident_input = next(t for t in test_inputs if t.category == "confident")

        wrapped_graph = trace_graph(
            build_graph(),
            config=ContrailConfig(sample_rate=1.0, export_format="stdout"),
        )

        result = await wrapped_graph.ainvoke(agent_config["input_fn"](confident_input))

        traces = result.get("__contrail_traces__", [])
        assert len(traces) > 0, f"No traces found for {agent_config['name']}"
        for trace in traces:
            assert_trace_record(trace)


@pytest.mark.integration
@pytest.mark.parametrize("agent_config", AGENT_CONFIGS, ids=[a["name"] for a in AGENT_CONFIGS])
class TestEntropyCalibration:
    """Entropy scores must match expected ranges per input category."""

    @pytest.mark.parametrize("category", ["confident", "boundary", "fragile"])
    async def test_entropy_in_range(self, agent_config, category):
        pytest.importorskip("contrail")
        from contrail import ContrailConfig, trace_graph

        build_graph = _import_build_fn(agent_config["build_fn"])
        inputs_for_category = [
            t for t in ALL_INPUTS[agent_config["name"]] if t.category == category
        ]
        if not inputs_for_category:
            pytest.skip(f"No {category!r} inputs for {agent_config['name']}")

        wrapped_graph = trace_graph(
            build_graph(),
            config=ContrailConfig(sample_rate=1.0, export_format="stdout"),
        )

        calibration_failures = 0
        for test_input in inputs_for_category:
            result = await wrapped_graph.ainvoke(agent_config["input_fn"](test_input))
            traces = result.get("__contrail_traces__", [])
            if not traces:
                continue
            # Check the routing node trace (first trace)
            trace = traces[0]
            min_e, max_e = ENTROPY_RANGES[category]
            if not (min_e <= trace.entropy_score <= max_e):
                calibration_failures += 1

        # Allow up to 20% calibration failures per category
        total = len(inputs_for_category)
        failure_rate = calibration_failures / total if total > 0 else 0
        assert failure_rate <= 0.2, (
            f"{agent_config['name']} {category}: {calibration_failures}/{total} "
            f"inputs failed entropy calibration (>{20}% threshold)"
        )


@pytest.mark.integration
@pytest.mark.parametrize("agent_config", AGENT_CONFIGS, ids=[a["name"] for a in AGENT_CONFIGS])
class TestAttributionPlausibility:
    """Attribution dimension labels must be non-empty and human-readable."""

    async def test_attribution_labels_readable(self, agent_config):
        pytest.importorskip("contrail")
        from contrail import ContrailConfig, trace_graph

        build_graph = _import_build_fn(agent_config["build_fn"])
        test_inputs = ALL_INPUTS[agent_config["name"]]

        wrapped_graph = trace_graph(
            build_graph(),
            config=ContrailConfig(sample_rate=1.0, export_format="stdout"),
        )

        for test_input in test_inputs[:2]:  # Check first 2 inputs per agent
            result = await wrapped_graph.ainvoke(agent_config["input_fn"](test_input))
            traces = result.get("__contrail_traces__", [])
            for trace in traces:
                assert trace.attribution_dimension, "attribution_dimension must not be empty"
                # Must be a real word/phrase, not a UUID or error code
                assert len(trace.attribution_dimension) > 2, (
                    f"attribution_dimension too short: {trace.attribution_dimension!r}"
                )
                assert trace.attribution_dimension[0].isalpha(), (
                    f"attribution_dimension should start with a letter: {trace.attribution_dimension!r}"
                )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("agent_config", AGENT_CONFIGS, ids=[a["name"] for a in AGENT_CONFIGS])
class TestLatencyOverhead:
    """Hot-path latency increase must be < 5% after Contrail wrapping."""

    async def test_hot_path_latency(self, agent_config):
        pytest.importorskip("contrail")
        from contrail import ContrailConfig, trace_graph
        import time

        build_graph = _import_build_fn(agent_config["build_fn"])
        test_inputs = ALL_INPUTS[agent_config["name"]]
        confident_input = next(t for t in test_inputs if t.category == "confident")
        input_state = agent_config["input_fn"](confident_input)

        N_RUNS = 5  # Reduced for CI — increase to 10+ for thorough testing

        # Baseline timing
        baseline_graph = build_graph()
        baseline_times = []
        for _ in range(N_RUNS):
            start = time.perf_counter()
            await baseline_graph.ainvoke(input_state)
            baseline_times.append((time.perf_counter() - start) * 1000)
        baseline_avg = sum(baseline_times) / len(baseline_times)

        # Wrapped timing (async mode — contrast runs don't block hot path)
        wrapped_graph = trace_graph(
            build_graph(),
            config=ContrailConfig(sample_rate=1.0, async_mode=True, export_format="stdout"),
        )
        wrapped_times = []
        for _ in range(N_RUNS):
            start = time.perf_counter()
            await wrapped_graph.ainvoke(input_state)
            wrapped_times.append((time.perf_counter() - start) * 1000)
        wrapped_avg = sum(wrapped_times) / len(wrapped_times)

        assert_latency_overhead(baseline_avg, wrapped_avg, max_overhead_pct=5.0)


@pytest.mark.integration
@pytest.mark.parametrize("agent_config", AGENT_CONFIGS, ids=[a["name"] for a in AGENT_CONFIGS])
class TestJSONLRoundTrip:
    """JSONL export → read back → reconstructed records match originals."""

    async def test_jsonl_roundtrip(self, agent_config):
        pytest.importorskip("contrail")
        from contrail import ContrailConfig, trace_graph
        from contrail.record import TraceRecord

        build_graph = _import_build_fn(agent_config["build_fn"])
        test_inputs = ALL_INPUTS[agent_config["name"]]
        confident_input = next(t for t in test_inputs if t.category == "confident")

        with tempfile.TemporaryDirectory() as tmpdir:
            wrapped_graph = trace_graph(
                build_graph(),
                config=ContrailConfig(
                    sample_rate=1.0,
                    export_format="jsonl",
                    export_path=tmpdir,
                ),
            )

            await wrapped_graph.ainvoke(agent_config["input_fn"](confident_input))

            # Find and read the JSONL file
            jsonl_files = list(Path(tmpdir).glob("*.jsonl"))
            assert len(jsonl_files) > 0, "No JSONL files produced"

            for jsonl_file in jsonl_files:
                with open(jsonl_file) as f:
                    lines = f.readlines()
                assert len(lines) > 0, "JSONL file is empty"
                for line in lines:
                    record_dict = json.loads(line)
                    # Reconstruct and validate
                    reconstructed = TraceRecord.from_dict(record_dict)
                    assert_trace_record(reconstructed)
                    assert reconstructed.node_id == record_dict["node_id"]
                    assert abs(reconstructed.entropy_score - record_dict["entropy_score"]) < 1e-9
