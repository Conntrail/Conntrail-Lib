"""
Tests for TraceRecord (Phase 4).
"""
import pytest

from contrail.record import TraceRecord


class TestStabilityLabel:
    """stability_label() is a pure function — test it now."""

    def test_zero_entropy_is_confident(self):
        assert TraceRecord.stability_label(0.0) == "confident"

    def test_low_entropy_is_confident(self):
        assert TraceRecord.stability_label(0.24) == "confident"

    def test_boundary_threshold(self):
        assert TraceRecord.stability_label(0.25) == "boundary"

    def test_mid_entropy_is_boundary(self):
        assert TraceRecord.stability_label(0.45) == "boundary"

    def test_high_threshold_is_boundary(self):
        assert TraceRecord.stability_label(0.60) == "boundary"

    def test_above_threshold_is_fragile(self):
        assert TraceRecord.stability_label(0.61) == "fragile"

    def test_max_entropy_is_fragile(self):
        assert TraceRecord.stability_label(1.0) == "fragile"


class TestBuildSummary:
    """build_summary() is a pure function — test it now."""

    def test_basic_summary(self):
        summary = TraceRecord.build_summary(
            node_id="router",
            route="escalate",
            stability="fragile",
            entropy=0.75,
            attribution="urgency",
            counterfactual="general",
        )
        assert "router" in summary
        assert "escalate" in summary
        assert "fragile" in summary
        assert "0.75" in summary
        assert "urgency" in summary
        assert "general" in summary

    def test_summary_no_counterfactual(self):
        summary = TraceRecord.build_summary(
            node_id="router",
            route="general",
            stability="confident",
            entropy=0.1,
            attribution="sentiment",
            counterfactual=None,
        )
        assert "router" in summary
        assert "confident" in summary
        assert "If that dimension" not in summary


class TestTraceRecordSerialization:
    def test_to_dict_not_implemented(self):
        """PHASE 4: to_dict() will be implemented."""
        pytest.skip("Implement in Phase 4")

    def test_from_dict_roundtrip(self):
        """PHASE 4: serialise → deserialise preserves all fields."""
        pytest.skip("Implement in Phase 4")
