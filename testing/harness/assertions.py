"""
Reusable assertion helpers for Conntrail TraceRecord validation.
"""
from typing import Any


def assert_trace_record(record: Any, *, node_id: str | None = None) -> None:
    """Assert that a TraceRecord is structurally complete and valid."""
    assert record is not None, "TraceRecord must not be None"

    required_fields = [
        "trace_id",
        "node_id",
        "timestamp",
        "original_input",
        "original_route",
        "entropy_score",
        "stability",
        "attribution_dimension",
        "plain_language_summary",
        "raw_contrasts",
        "raw_outputs",
    ]
    for field in required_fields:
        assert hasattr(record, field), f"TraceRecord missing field: {field}"

    assert isinstance(record.entropy_score, float), "entropy_score must be float"
    assert 0.0 <= record.entropy_score <= 1.0, (
        f"entropy_score {record.entropy_score} out of range [0, 1]"
    )
    assert record.stability in ("confident", "boundary", "fragile"), (
        f"Invalid stability label: {record.stability}"
    )
    assert isinstance(record.attribution_dimension, str) and record.attribution_dimension, (
        "attribution_dimension must be a non-empty string"
    )
    assert isinstance(record.plain_language_summary, str) and record.plain_language_summary, (
        "plain_language_summary must be a non-empty string"
    )

    if node_id is not None:
        assert record.node_id == node_id, (
            f"Expected node_id={node_id!r}, got {record.node_id!r}"
        )


def assert_entropy_range(
    record: Any,
    *,
    min_entropy: float = 0.0,
    max_entropy: float = 1.0,
    label: str = "",
) -> None:
    """Assert that a TraceRecord's entropy falls within the expected range."""
    score = record.entropy_score
    assert min_entropy <= score <= max_entropy, (
        f"{label}: entropy {score:.3f} not in [{min_entropy}, {max_entropy}]"
        f" (stability={record.stability})"
    )


def assert_non_intrusive(baseline_result: Any, wrapped_result: Any) -> None:
    """
    Assert that wrapping with Conntrail does not alter agent output.
    Compares the final output dict, excluding __conntrail_traces__.
    """
    def clean(output: dict) -> dict:
        return {k: v for k, v in output.items() if k != "__conntrail_traces__"}

    baseline_clean = clean(baseline_result.output)
    wrapped_clean = clean(wrapped_result.output)

    assert baseline_result.route_taken == wrapped_result.route_taken, (
        f"Route changed: baseline={baseline_result.route_taken!r}, "
        f"wrapped={wrapped_result.route_taken!r}"
    )
    assert baseline_result.nodes_visited == wrapped_result.nodes_visited, (
        f"Nodes visited changed: baseline={baseline_result.nodes_visited}, "
        f"wrapped={wrapped_result.nodes_visited}"
    )
    assert baseline_clean == wrapped_clean, (
        f"Output changed after Conntrail wrapping.\n"
        f"Baseline: {baseline_clean}\n"
        f"Wrapped:  {wrapped_clean}"
    )


def assert_latency_overhead(
    baseline_ms: float,
    wrapped_ms: float,
    *,
    max_overhead_pct: float = 5.0,
) -> None:
    """Assert that hot-path latency overhead is within acceptable bounds."""
    overhead_pct = ((wrapped_ms - baseline_ms) / baseline_ms) * 100
    assert overhead_pct <= max_overhead_pct, (
        f"Latency overhead {overhead_pct:.1f}% exceeds {max_overhead_pct}% limit "
        f"(baseline={baseline_ms:.1f}ms, wrapped={wrapped_ms:.1f}ms)"
    )
