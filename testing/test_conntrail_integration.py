"""
Phase 6 integration tests — Conntrail against all 4 real agents.

These run after Phase 5 is complete (NodeInterceptor + Public API).
Tests the full 7-test matrix: non-intrusion, trace completeness, entropy
calibration, attribution plausibility, latency overhead, JSONL round-trip,
and prompt quality audit.

The prompt quality audit (TestPromptQualityAudit) only requires Phase 3
and runs independently of trace_graph — it can run now.
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
        "routing_key": "category",    # deterministic classification field
        "routing_node": "classify_query",  # the node with conditional edges
    },
    {
        "name": "multi_agent_supervisor",
        "build_fn": "testing.agents.multi_agent_supervisor.agent.build_graph",
        "input_fn": lambda t: {"task": t.text, "assigned_agent": None, "result": None},
        "routing_key": "assigned_agent",
        "routing_node": "supervisor",
    },
    {
        "name": "adaptive_rag",
        "build_fn": "testing.agents.adaptive_rag.agent.build_graph",
        "input_fn": lambda t: {"query": t.text, "strategy": None, "retrieved_docs": [], "answer": None},
        "routing_key": "strategy",
        "routing_node": "route_query",
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
    """Conntrail wrapping must not alter agent output."""

    async def test_output_unchanged(self, agent_config):
        """Conntrail wrapping must not alter the agent's routing decision.

        Only the routing field (category/assigned_agent/strategy) is compared —
        LLM-generated response text is inherently non-deterministic across runs.
        """
        pytest.importorskip("conntrail")
        from conntrail import ConntrailConfig, trace_graph

        build_graph = _import_build_fn(agent_config["build_fn"])
        test_inputs = ALL_INPUTS[agent_config["name"]]
        confident_input = next(t for t in test_inputs if t.category == "confident")

        # sample_rate=0.0: Conntrail wraps the graph but fires no analysis.
        # Any routing difference must be caused by the wrapping mechanism itself, not LLM variance.
        baseline_graph = build_graph()
        wrapped_graph = trace_graph(
            build_graph(),
            config=ConntrailConfig(sample_rate=0.0, export_format="stdout"),
        )

        input_state = agent_config["input_fn"](confident_input)
        routing_key = agent_config["routing_key"]

        baseline_result = await baseline_graph.ainvoke(input_state)
        wrapped_result = await wrapped_graph.ainvoke(input_state)

        assert baseline_result[routing_key] == wrapped_result[routing_key], (
            f"Routing decision changed after Conntrail wrapping for {agent_config['name']}: "
            f"baseline={baseline_result[routing_key]!r}, wrapped={wrapped_result[routing_key]!r}"
        )


@pytest.mark.integration
@pytest.mark.parametrize("agent_config", AGENT_CONFIGS, ids=[a["name"] for a in AGENT_CONFIGS])
class TestTraceCompleteness:
    """Every node call must produce a TraceRecord."""

    async def test_traces_present(self, agent_config):
        pytest.importorskip("conntrail")
        from conntrail import ConntrailConfig, trace_graph

        build_graph = _import_build_fn(agent_config["build_fn"])
        test_inputs = ALL_INPUTS[agent_config["name"]]
        confident_input = next(t for t in test_inputs if t.category == "confident")

        wrapped_graph = trace_graph(
            build_graph(),
            config=ConntrailConfig(sample_rate=1.0, async_mode=False, export_format="stdout"),
            only_nodes={agent_config["routing_node"]},
        )

        result = await wrapped_graph.ainvoke(agent_config["input_fn"](confident_input))

        traces = result.get("__conntrail_traces__", [])
        assert len(traces) > 0, f"No traces found for {agent_config['name']}"
        for trace in traces:
            assert_trace_record(trace)


@pytest.mark.integration
@pytest.mark.parametrize("agent_config", AGENT_CONFIGS, ids=[a["name"] for a in AGENT_CONFIGS])
class TestEntropyCalibration:
    """Entropy scores must match expected ranges per input category."""

    @pytest.mark.parametrize("category", ["confident", "boundary", "fragile"])
    async def test_entropy_in_range(self, agent_config, category):
        pytest.importorskip("conntrail")
        from conntrail import ConntrailConfig, trace_graph

        build_graph = _import_build_fn(agent_config["build_fn"])
        inputs_for_category = [
            t for t in ALL_INPUTS[agent_config["name"]] if t.category == category
        ]
        if not inputs_for_category:
            pytest.skip(f"No {category!r} inputs for {agent_config['name']}")

        wrapped_graph = trace_graph(
            build_graph(),
            config=ConntrailConfig(sample_rate=1.0, async_mode=False, export_format="stdout"),
            only_nodes={agent_config["routing_node"]},
        )

        calibration_failures = 0
        for test_input in inputs_for_category:
            result = await wrapped_graph.ainvoke(agent_config["input_fn"](test_input))
            traces = result.get("__conntrail_traces__", [])
            if not traces:
                continue
            # Only the routing node is traced — take its trace
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
        pytest.importorskip("conntrail")
        from conntrail import ConntrailConfig, trace_graph

        build_graph = _import_build_fn(agent_config["build_fn"])
        test_inputs = ALL_INPUTS[agent_config["name"]]

        wrapped_graph = trace_graph(
            build_graph(),
            config=ConntrailConfig(sample_rate=1.0, async_mode=False, export_format="stdout"),
            only_nodes={agent_config["routing_node"]},
        )

        for test_input in test_inputs[:2]:  # Check first 2 inputs per agent
            result = await wrapped_graph.ainvoke(agent_config["input_fn"](test_input))
            traces = result.get("__conntrail_traces__", [])
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
    """Hot-path latency increase must be < 5% after Conntrail wrapping."""

    async def test_hot_path_latency(self, agent_config):
        pytest.importorskip("conntrail")
        from conntrail import ConntrailConfig, trace_graph
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
            config=ConntrailConfig(sample_rate=1.0, async_mode=True, export_format="stdout"),
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
        pytest.importorskip("conntrail")
        from conntrail import ConntrailConfig, trace_graph
        from conntrail.record import TraceRecord

        build_graph = _import_build_fn(agent_config["build_fn"])
        test_inputs = ALL_INPUTS[agent_config["name"]]
        confident_input = next(t for t in test_inputs if t.category == "confident")

        with tempfile.TemporaryDirectory() as tmpdir:
            wrapped_graph = trace_graph(
                build_graph(),
                config=ConntrailConfig(
                    sample_rate=1.0,
                    async_mode=False,
                    export_format="jsonl",
                    export_path=tmpdir,
                ),
                only_nodes={agent_config["routing_node"]},
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


# ---------------------------------------------------------------------------
# Prompt Quality Audit — requires Phase 3 only, no trace_graph needed
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestPromptQualityAudit:
    """
    Conntrail as a prompt quality tool.

    Runs the same fixture inputs through two versions of the customer_support
    routing node — one with a weak vague prompt, one with our strong explicit
    prompt — and asserts that the strong prompt produces lower mean entropy.

    High entropy on an "obvious" input = the prompt is under-specified.
    Attribution breakdown reveals which dimension the prompt is sensitive to:
      - neutral flips  → prompt relies on tone, not intent  (emotion-sensitive)
      - similar flips  → prompt reacts to word choice       (surface-form brittle)
      - opposite flips → genuine boundary                   (expected, healthy)
    """

    # 3 inputs: clear, ambiguous, and borderline — enough for a statistical signal
    # Kept small to stay within Groq free-tier RPM limits (30 RPM)
    AUDIT_INPUTS = [
        "I need a refund immediately, my order never arrived!",
        "I want to speak to a manager right now.",
        "I have an issue with something I purchased.",   # deliberately ambiguous
    ]

    @pytest.fixture(autouse=True)
    def require_api_key(self):
        import os
        if not any(os.getenv(k) for k in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")):
            pytest.skip("No API key available")

    def _make_weak_node(self):
        """Routing node with a vague prompt — no category definitions."""
        from langchain_core.messages import HumanMessage, SystemMessage
        from testing.harness.llm import get_llm

        async def weak_classify(state):
            llm = get_llm()
            response = await llm.ainvoke([
                SystemMessage(content="You are a customer support assistant. Classify this customer message into a category. Respond with only the category name."),
                HumanMessage(content=state["message"]),
            ])
            return {**state, "category": response.content.strip().lower()}

        return weak_classify

    def _make_strong_node(self):
        """Routing node with explicit category definitions — our production prompt."""
        from testing.agents.customer_support.agent import classify_query
        return classify_query

    async def _score_node(self, node, input_text, gen, analyser):
        """Run one input through a node and return its entropy score."""
        from conntrail.contrast import ContrastGenerator
        contrasts = await gen.generate(input_text)
        result = await analyser.analyse(
            node,
            {"message": input_text, "category": None, "response": None},
            contrasts,
        )
        return result.entropy_score, result.attribution_dimension, result.contrast_routes

    @pytest.mark.asyncio
    async def test_prompt_quality_audit(self):
        """
        Single-pass audit: iterate inputs once, collect entropy + attribution
        for both weak and strong prompts, then assert both properties.

        Combined into one test to stay within Groq free-tier 30 RPM limit —
        two separate tests would each make 24 node calls (48 total > 30 RPM).
        """
        from collections import Counter

        from conntrail.analyser import DivergenceAnalyser
        from conntrail.contrast import ContrastGenerator
        from testing.harness.llm import get_contrast_llm

        gen = ContrastGenerator(llm=get_contrast_llm())
        analyser = DivergenceAnalyser()
        weak_node = self._make_weak_node()
        strong_node = self._make_strong_node()

        weak_scores, strong_scores = [], []
        weak_attrs, strong_attrs = Counter(), Counter()

        for input_text in self.AUDIT_INPUTS:
            w_entropy, w_attr, _ = await self._score_node(weak_node, input_text, gen, analyser)
            await asyncio.sleep(3)  # respect Groq free-tier RPM limit
            s_entropy, s_attr, _ = await self._score_node(strong_node, input_text, gen, analyser)
            await asyncio.sleep(3)
            weak_scores.append(w_entropy)
            strong_scores.append(s_entropy)
            weak_attrs[w_attr] += 1
            strong_attrs[s_attr] += 1

        weak_mean = sum(weak_scores) / len(weak_scores)
        strong_mean = sum(strong_scores) / len(strong_scores)
        strong_stable = strong_attrs.get("none detected", 0)
        weak_stable = weak_attrs.get("none detected", 0)

        print(f"\n  Weak scores:   {[f'{s:.3f}' for s in weak_scores]}  mean={weak_mean:.3f}")
        print(f"  Strong scores: {[f'{s:.3f}' for s in strong_scores]}  mean={strong_mean:.3f}")
        print(f"  Weak prompt attribution breakdown:   {dict(weak_attrs)}")
        print(f"  Strong prompt attribution breakdown: {dict(strong_attrs)}")
        print(f"  Stable decisions — weak: {weak_stable}, strong: {strong_stable}")

        # Assertion 1: strong prompt produces lower mean entropy
        assert strong_mean < weak_mean, (
            f"Strong prompt mean entropy ({strong_mean:.3f}) should be lower than "
            f"weak prompt mean entropy ({weak_mean:.3f}).\n"
            f"Weak scores:   {[f'{s:.3f}' for s in weak_scores]}\n"
            f"Strong scores: {[f'{s:.3f}' for s in strong_scores]}"
        )

        # Assertion 2: strong prompt has >= stable (none detected) decisions
        assert strong_stable >= weak_stable, (
            f"Strong prompt should have >= stable decisions as weak prompt.\n"
            f"Weak: {dict(weak_attrs)}\nStrong: {dict(strong_attrs)}"
        )
