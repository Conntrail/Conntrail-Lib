"""
Tests for TraceRecord (Phase 4).
"""
import json
from datetime import datetime, timezone

import pytest

from conntrail.contrast import ContrastSet
from conntrail.record import TraceRecord


def _make_record(**overrides) -> TraceRecord:
    """Build a minimal but complete TraceRecord for testing."""
    defaults = dict(
        trace_id="abc-123",
        node_id="router",
        timestamp=datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        original_input="I need a refund",
        original_route="refund",
        entropy_score=0.25,
        stability="boundary",
        attribution_dimension="semantic intensity",
        plain_language_summary="The 'router' node routed to 'refund' with boundary confidence.",
        raw_contrasts=ContrastSet(
            similar="I want my money back",
            neutral="I have a question about my order",
            opposite="Great service, no issues!",
        ),
        raw_outputs={
            "original": "refund",
            "similar": "refund",
            "neutral": "general",
            "opposite": "general",
        },
        counterfactual_route=None,
    )
    defaults.update(overrides)
    return TraceRecord(**defaults)


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
    def test_to_dict_returns_dict(self):
        record = _make_record()
        result = record.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_json_serialisable(self):
        record = _make_record()
        # Must not raise
        json.dumps(record.to_dict())

    def test_to_dict_all_fields_present(self):
        record = _make_record()
        d = record.to_dict()
        for key in (
            "trace_id", "node_id", "timestamp", "original_input",
            "original_route", "entropy_score", "stability",
            "attribution_dimension", "plain_language_summary",
            "raw_contrasts", "raw_outputs", "counterfactual_route",
        ):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_timestamp_is_isoformat(self):
        record = _make_record()
        ts = record.to_dict()["timestamp"]
        assert isinstance(ts, str)
        # Must parse back without error
        datetime.fromisoformat(ts)

    def test_to_dict_raw_contrasts_is_dict(self):
        record = _make_record()
        contrasts = record.to_dict()["raw_contrasts"]
        assert isinstance(contrasts, dict)
        assert set(contrasts.keys()) == {"similar", "neutral", "opposite"}

    def test_to_dict_counterfactual_none(self):
        record = _make_record(counterfactual_route=None)
        assert record.to_dict()["counterfactual_route"] is None

    def test_to_dict_counterfactual_present(self):
        record = _make_record(counterfactual_route="general")
        assert record.to_dict()["counterfactual_route"] == "general"

    def test_from_dict_roundtrip(self):
        """Serialise → deserialise preserves all fields."""
        original = _make_record()
        restored = TraceRecord.from_dict(original.to_dict())

        assert restored.trace_id == original.trace_id
        assert restored.node_id == original.node_id
        assert restored.timestamp == original.timestamp
        assert restored.original_input == original.original_input
        assert restored.original_route == original.original_route
        assert restored.entropy_score == original.entropy_score
        assert restored.stability == original.stability
        assert restored.attribution_dimension == original.attribution_dimension
        assert restored.plain_language_summary == original.plain_language_summary
        assert restored.raw_contrasts.similar == original.raw_contrasts.similar
        assert restored.raw_contrasts.neutral == original.raw_contrasts.neutral
        assert restored.raw_contrasts.opposite == original.raw_contrasts.opposite
        assert restored.raw_outputs == original.raw_outputs
        assert restored.counterfactual_route == original.counterfactual_route

    def test_from_dict_roundtrip_with_counterfactual(self):
        original = _make_record(counterfactual_route="escalation")
        restored = TraceRecord.from_dict(original.to_dict())
        assert restored.counterfactual_route == "escalation"


class TestStdoutExporter:
    @pytest.mark.asyncio
    async def test_write_does_not_raise(self, capsys):
        from conntrail.exporters.stdout import StdoutExporter
        exporter = StdoutExporter()
        await exporter.write(_make_record())
        out = capsys.readouterr().out
        assert "router" in out
        assert "refund" in out

    @pytest.mark.asyncio
    async def test_write_includes_entropy(self, capsys):
        from conntrail.exporters.stdout import StdoutExporter
        exporter = StdoutExporter()
        await exporter.write(_make_record(entropy_score=0.75))
        out = capsys.readouterr().out
        assert "0.75" in out

    @pytest.mark.asyncio
    async def test_write_shows_counterfactual(self, capsys):
        from conntrail.exporters.stdout import StdoutExporter
        exporter = StdoutExporter()
        await exporter.write(_make_record(counterfactual_route="general"))
        out = capsys.readouterr().out
        assert "general" in out

    @pytest.mark.asyncio
    async def test_write_no_counterfactual_no_alt_line(self, capsys):
        from conntrail.exporters.stdout import StdoutExporter
        exporter = StdoutExporter()
        await exporter.write(_make_record(counterfactual_route=None))
        out = capsys.readouterr().out
        assert "Alt:" not in out


class TestJsonlExporter:
    @pytest.mark.asyncio
    async def test_write_creates_file(self, tmp_path):
        from conntrail.exporters.jsonl import JsonlExporter
        exporter = JsonlExporter(export_path=str(tmp_path / "traces"))
        await exporter.write(_make_record())
        files = list((tmp_path / "traces").glob("*.jsonl"))
        assert len(files) == 1

    @pytest.mark.asyncio
    async def test_write_valid_json_line(self, tmp_path):
        from conntrail.exporters.jsonl import JsonlExporter
        exporter = JsonlExporter(export_path=str(tmp_path / "traces"))
        await exporter.write(_make_record())
        file = next((tmp_path / "traces").glob("*.jsonl"))
        lines = file.read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["node_id"] == "router"

    @pytest.mark.asyncio
    async def test_write_appends_multiple_records(self, tmp_path):
        from conntrail.exporters.jsonl import JsonlExporter
        exporter = JsonlExporter(export_path=str(tmp_path / "traces"))
        await exporter.write(_make_record(trace_id="id-1"))
        await exporter.write(_make_record(trace_id="id-2"))
        file = next((tmp_path / "traces").glob("*.jsonl"))
        lines = file.read_text().strip().splitlines()
        assert len(lines) == 2
        ids = [json.loads(l)["trace_id"] for l in lines]
        assert ids == ["id-1", "id-2"]

    @pytest.mark.asyncio
    async def test_roundtrip_via_jsonl(self, tmp_path):
        from conntrail.exporters.jsonl import JsonlExporter
        original = _make_record()
        exporter = JsonlExporter(export_path=str(tmp_path / "traces"))
        await exporter.write(original)
        file = next((tmp_path / "traces").glob("*.jsonl"))
        data = json.loads(file.read_text().strip())
        restored = TraceRecord.from_dict(data)
        assert restored.trace_id == original.trace_id
        assert restored.entropy_score == original.entropy_score
        assert restored.raw_contrasts.opposite == original.raw_contrasts.opposite

    @pytest.mark.asyncio
    async def test_file_named_with_date(self, tmp_path):
        from datetime import date

        from conntrail.exporters.jsonl import JsonlExporter
        exporter = JsonlExporter(export_path=str(tmp_path / "traces"))
        await exporter.write(_make_record())
        file = next((tmp_path / "traces").glob("*.jsonl"))
        assert date.today().isoformat() in file.name
