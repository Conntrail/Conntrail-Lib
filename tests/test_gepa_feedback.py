"""Unit tests for CPEFeedbackFunction and cpe_feedback — no API keys required."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from conntrail.contrast import ContrastSet
from conntrail.gepa.feedback import CPEFeedbackFunction, cpe_feedback
from conntrail.gepa.schema import PromptAttemptRecord
from conntrail.record import TraceRecord


def _make_trace(
    entropy: float,
    stability: str,
    attribution: str = "urgency/sentiment",
) -> TraceRecord:
    return TraceRecord(
        trace_id="test-id",
        node_id="router",
        timestamp=datetime.now(timezone.utc),
        original_input="test input",
        original_route="route_a",
        entropy_score=entropy,
        stability=stability,
        attribution_dimension=attribution,
        plain_language_summary="test summary",
        raw_contrasts=ContrastSet(similar="s", neutral="n", opposite="o"),
        raw_outputs={},
        counterfactual_route=None,
    )


def _make_attempt(traces: list[TraceRecord], score: float | None = None) -> PromptAttemptRecord:
    attempt = PromptAttemptRecord(attempt_id="a1", prompt_candidate="test prompt")
    attempt.traces = traces
    attempt.scalar_score = score
    return attempt


# --- cpe_feedback ---

def test_feedback_no_traces():
    attempt = _make_attempt([])
    result = cpe_feedback(attempt)
    assert "no conntrail traces" in result.lower()


def test_feedback_dominant_fragile():
    traces = [_make_trace(0.9, "fragile")] * 3 + [_make_trace(0.1, "confident")]
    result = cpe_feedback(_make_attempt(traces))
    assert "underspecified" in result or "fragile" in result.lower()


def test_feedback_dominant_boundary():
    traces = [_make_trace(0.4, "boundary")] * 3 + [_make_trace(0.1, "confident")]
    result = cpe_feedback(_make_attempt(traces))
    assert "boundary" in result.lower() or "inconsistent" in result.lower()


def test_feedback_dominant_confident():
    traces = [_make_trace(0.1, "confident")] * 4
    result = cpe_feedback(_make_attempt(traces))
    assert "stable" in result.lower() or "confident" in result.lower()


def test_feedback_includes_attribution():
    traces = [_make_trace(0.8, "fragile", "surface form")] * 2
    result = cpe_feedback(_make_attempt(traces))
    assert "surface form" in result


def test_feedback_includes_task_score_when_not_confident():
    traces = [_make_trace(0.7, "fragile")] * 2
    result = cpe_feedback(_make_attempt(traces, score=0.42))
    assert "0.420" in result or "task score" in result.lower()


def test_feedback_no_task_score_line_when_confident():
    traces = [_make_trace(0.1, "confident")] * 3
    result = cpe_feedback(_make_attempt(traces, score=0.99))
    assert "task score" not in result.lower()


def test_feedback_header_contains_trace_count():
    traces = [_make_trace(0.5, "boundary")] * 7
    result = cpe_feedback(_make_attempt(traces))
    assert "7" in result


# --- PromptAttemptRecord properties ---

def test_mean_entropy():
    attempt = _make_attempt([_make_trace(0.2, "confident"), _make_trace(0.8, "fragile")])
    assert abs(attempt.mean_entropy - 0.5) < 1e-9


def test_mean_entropy_none_when_no_traces():
    assert _make_attempt([]).mean_entropy is None


def test_fragile_count():
    traces = [
        _make_trace(0.9, "fragile"),
        _make_trace(0.1, "confident"),
        _make_trace(0.8, "fragile"),
    ]
    assert _make_attempt(traces).fragile_count == 2


def test_boundary_count():
    traces = [_make_trace(0.4, "boundary")] * 3 + [_make_trace(0.1, "confident")]
    assert _make_attempt(traces).boundary_count == 3


def test_dominant_attribution():
    traces = [
        _make_trace(0.8, "fragile", "semantic intensity"),
        _make_trace(0.7, "fragile", "urgency/sentiment"),
        _make_trace(0.9, "fragile", "semantic intensity"),
    ]
    assert _make_attempt(traces).dominant_attribution == "semantic intensity"


def test_dominant_attribution_none_when_no_traces():
    assert _make_attempt([]).dominant_attribution is None


# --- CPEFeedbackFunction ---

class _FakeCollector:
    def __init__(self, attempts):
        self._attempts = attempts

    @property
    def all_attempts(self):
        return self._attempts


def test_feedback_fn_no_attempts():
    fn = CPEFeedbackFunction(_FakeCollector([]))
    score, feedback = fn({}, {})
    assert score == 0.0
    assert "no attempts" in feedback.lower()


def test_feedback_fn_uses_task_metric():
    attempt = _make_attempt([_make_trace(0.5, "boundary")], score=None)
    collector = _FakeCollector([attempt])
    fn = CPEFeedbackFunction(collector, task_metric_fn=lambda g, p: 0.75)
    score, _ = fn({}, {})
    assert score == 0.75


def test_feedback_fn_falls_back_to_scalar_score():
    attempt = _make_attempt([_make_trace(0.3, "boundary")], score=0.6)
    collector = _FakeCollector([attempt])
    fn = CPEFeedbackFunction(collector)
    score, _ = fn({}, {})
    assert score == 0.6


def test_feedback_fn_falls_back_to_stability_proxy():
    attempt = _make_attempt([_make_trace(0.4, "boundary")])  # no scalar_score
    collector = _FakeCollector([attempt])
    fn = CPEFeedbackFunction(collector)
    score, _ = fn({}, {})
    assert abs(score - (1.0 - 0.4)) < 1e-9


def test_feedback_fn_returns_cpe_feedback_string():
    traces = [_make_trace(0.9, "fragile")] * 3
    attempt = _make_attempt(traces, score=0.5)
    collector = _FakeCollector([attempt])
    fn = CPEFeedbackFunction(collector)
    _, feedback = fn({}, {})
    assert "CPE Analysis" in feedback
