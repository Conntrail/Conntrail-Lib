"""
Tests for ContrastGenerator (Phase 2).
"""
import pytest

from contrail.contrast import ContrastGenerator, ContrastGenerationError, ContrastSet


class TestContrastSet:
    def test_as_list_order(self):
        cs = ContrastSet(similar="a", neutral="b", opposite="c")
        assert cs.as_list() == ["a", "b", "c"]

    def test_all_non_empty_true(self):
        cs = ContrastSet(
            similar="this is similar text",
            neutral="this is neutral text",
            opposite="this is opposite text",
        )
        assert cs.all_non_empty() is True

    def test_all_non_empty_false_on_short(self):
        cs = ContrastSet(similar="hi", neutral="ok", opposite="no")
        assert cs.all_non_empty() is False


class TestContrastGenerator:
    def test_instantiation(self):
        gen = ContrastGenerator()
        assert gen.model == "claude-haiku-4-5-20251001"

    def test_custom_model(self):
        gen = ContrastGenerator(model="gpt-4o-mini")
        assert gen.model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_generate_not_implemented_yet(self):
        gen = ContrastGenerator()
        with pytest.raises(NotImplementedError):
            await gen.generate("test input")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_generate_urgency_input(self):
        """PHASE 2: urgent input produces non-empty ContrastSet."""
        pytest.skip("Implement in Phase 2")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_generate_technical_input(self):
        """PHASE 2: technical query produces distinct variants."""
        pytest.skip("Implement in Phase 2")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_generate_ambiguous_input(self):
        """PHASE 2: ambiguous input handled gracefully."""
        pytest.skip("Implement in Phase 2")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_generate_emotional_input(self):
        """PHASE 2: emotional input produces sentiment-flipped opposite."""
        pytest.skip("Implement in Phase 2")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_generate_minimal_input(self):
        """PHASE 2: minimal single-character input handled without crash."""
        pytest.skip("Implement in Phase 2")

    def test_parse_response_not_implemented(self):
        gen = ContrastGenerator()
        with pytest.raises(NotImplementedError):
            gen._parse_response("{}")
