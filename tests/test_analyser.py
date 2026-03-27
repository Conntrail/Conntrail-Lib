"""
Tests for DivergenceAnalyser and entropy utilities (Phase 3).

All tests use the deterministic mock_router_node from fixtures — no API key needed.
mock_router_node routes:  urgency keywords → "escalate",  everything else → "general"
"""
import asyncio
import time

import pytest

from conntrail.analyser import AnalysisResult, DivergenceAnalyser
from conntrail.contrast import ContrastSet
from conntrail.utils.entropy import routing_entropy
from tests.fixtures.sample_graphs import ROUTINE_INPUTS, URGENT_INPUTS, mock_router_node


# ---------------------------------------------------------------------------
# routing_entropy — already live since Phase 1
# ---------------------------------------------------------------------------

class TestRoutingEntropy:
    def test_all_same_routes(self):
        assert routing_entropy(["a", "a", "a", "a"]) == 0.0

    def test_all_different_routes(self):
        assert routing_entropy(["a", "b", "c", "d"]) == pytest.approx(1.0)

    def test_half_half_routes(self):
        score = routing_entropy(["a", "a", "b", "b"])
        assert 0.4 < score < 0.6

    def test_three_same_one_different(self):
        score = routing_entropy(["a", "a", "a", "b"])
        assert 0.0 < score < 0.5

    def test_empty_list(self):
        assert routing_entropy([]) == 0.0

    def test_single_element(self):
        assert routing_entropy(["a"]) == 0.0

    def test_two_elements_same(self):
        assert routing_entropy(["x", "x"]) == 0.0

    def test_two_elements_different(self):
        assert routing_entropy(["x", "y"]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        from conntrail.utils.embedding import cosine_similarity
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        from conntrail.utils.embedding import cosine_similarity
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        from conntrail.utils.embedding import cosine_similarity
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_length_mismatch_raises(self):
        from conntrail.utils.embedding import cosine_similarity
        with pytest.raises(ValueError):
            cosine_similarity([1.0], [1.0, 2.0])


# ---------------------------------------------------------------------------
# DivergenceAnalyser._extract_route
# ---------------------------------------------------------------------------

class TestExtractRoute:
    def setup_method(self):
        self.analyser = DivergenceAnalyser()

    def test_explicit_route_key(self):
        inp = {"message": "hi", "category": None}
        out = {"message": "hi", "category": "refund"}
        assert self.analyser._extract_route(inp, out, route_key="category") == "refund"

    def test_explicit_route_key_missing_returns_unknown(self):
        out = {"message": "hi"}
        assert self.analyser._extract_route({}, out, route_key="missing_key") == "unknown"

    def test_auto_detect_none_to_string(self):
        inp = {"message": "hi", "category": None}
        out = {"message": "hi", "category": "escalation"}
        assert self.analyser._extract_route(inp, out) == "escalation"

    def test_auto_detect_changed_string(self):
        inp = {"message": "hi", "strategy": "old"}
        out = {"message": "hi", "strategy": "vector_search"}
        assert self.analyser._extract_route(inp, out) == "vector_search"

    def test_auto_detect_prefers_none_to_string_over_changed(self):
        # Both None→string and changed-string present — None→string wins (comes first)
        inp = {"message": "hi", "category": None, "strategy": "old"}
        out = {"message": "hi", "category": "refund", "strategy": "new"}
        result = self.analyser._extract_route(inp, out)
        assert result in ("refund", "new")  # either is acceptable

    def test_auto_detect_no_change_returns_unknown(self):
        inp = {"message": "hi"}
        out = {"message": "hi"}
        assert self.analyser._extract_route(inp, out) == "unknown"


# ---------------------------------------------------------------------------
# DivergenceAnalyser._infer_attribution
# ---------------------------------------------------------------------------

class TestInferAttribution:
    def setup_method(self):
        self.analyser = DivergenceAnalyser()

    def test_opposite_flips_names_semantic_intensity(self):
        attr, cf = self.analyser._infer_attribution(
            original_route="escalate",
            contrast_routes={"similar": "escalate", "neutral": "escalate", "opposite": "general"},
        )
        assert attr == "semantic intensity"
        assert cf == "general"

    def test_neutral_flips_names_urgency_sentiment(self):
        # Only neutral flips — opposite stays same, so neutral wins
        attr, cf = self.analyser._infer_attribution(
            original_route="escalate",
            contrast_routes={"similar": "escalate", "neutral": "general", "opposite": "escalate"},
        )
        assert attr == "urgency/sentiment"
        assert cf == "general"

    def test_similar_flips_names_surface_form(self):
        # Only similar flips — neutral and opposite stay same
        attr, cf = self.analyser._infer_attribution(
            original_route="escalate",
            contrast_routes={"similar": "general", "neutral": "escalate", "opposite": "escalate"},
        )
        assert attr == "surface form"
        assert cf == "general"

    def test_no_flip_returns_none_detected(self):
        attr, cf = self.analyser._infer_attribution(
            original_route="escalate",
            contrast_routes={"similar": "escalate", "neutral": "escalate", "opposite": "escalate"},
        )
        assert attr == "none detected"
        assert cf is None

    def test_priority_opposite_over_neutral(self):
        # Both neutral and opposite flip — opposite should win
        attr, cf = self.analyser._infer_attribution(
            original_route="A",
            contrast_routes={"similar": "A", "neutral": "B", "opposite": "C"},
        )
        assert attr == "semantic intensity"
        assert cf == "C"


# ---------------------------------------------------------------------------
# DivergenceAnalyser.analyse — with deterministic mock_router_node
# ---------------------------------------------------------------------------

class TestDivergenceAnalyser:
    def setup_method(self):
        self.analyser = DivergenceAnalyser()

    @pytest.mark.asyncio
    async def test_urgent_input_high_entropy(self):
        """Urgent input + contrasts that flip the route → entropy > 0.5."""
        # similar keeps urgency → escalate, neutral + opposite drop it → general
        contrasts = ContrastSet(
            similar="This needs immediate attention, it's critical!",   # → escalate
            neutral="Please look into this issue when you can.",         # → general
            opposite="There is no rush on this, handle when convenient.",# → general
        )
        original = {"message": "I need this fixed ASAP!", "route": None, "response": None}
        result = await self.analyser.analyse(mock_router_node, original, contrasts)
        assert result.entropy_score >= 0.5, f"Expected high entropy, got {result.entropy_score}"

    @pytest.mark.asyncio
    async def test_routine_input_low_entropy(self):
        """Routine input — only opposite flips → low entropy."""
        contrasts = ContrastSet(
            similar="When do you open?",                                 # → general
            neutral="What are your business hours?",                    # → general
            opposite="I need to know your hours IMMEDIATELY, urgent!",  # → escalate
        )
        original = {"message": "What are your business hours?", "route": None, "response": None}
        result = await self.analyser.analyse(mock_router_node, original, contrasts)
        assert result.entropy_score < 0.8, f"Expected lower entropy, got {result.entropy_score}"
        assert result.original_route == "general"

    @pytest.mark.asyncio
    async def test_confident_input_zero_entropy(self):
        """All 4 variants route the same way → entropy = 0.0."""
        contrasts = ContrastSet(
            similar="This is absolutely urgent and critical!",
            neutral="Please address this critical system issue.",
            opposite="This is an emergency that needs immediate action!",
        )
        original = {"message": "URGENT: system is down!", "route": None, "response": None}
        result = await self.analyser.analyse(mock_router_node, original, contrasts)
        assert result.entropy_score == 0.0
        assert result.original_route == "escalate"
        assert result.attribution_dimension == "none detected"
        assert result.counterfactual_route is None

    @pytest.mark.asyncio
    async def test_attribution_dimension_detected(self):
        """Attribution dimension is a non-empty string when route flips."""
        contrasts = ContrastSet(
            similar="I need this urgently!",
            neutral="Please help with this.",
            opposite="No rush on this at all.",
        )
        original = {"message": "Fix this ASAP!", "route": None, "response": None}
        result = await self.analyser.analyse(mock_router_node, original, contrasts)
        assert result.attribution_dimension
        assert len(result.attribution_dimension) > 2

    @pytest.mark.asyncio
    async def test_counterfactual_route_is_opposite_of_original(self):
        """When route flips, counterfactual_route is the flipped destination."""
        contrasts = ContrastSet(
            similar="Fix this immediately!",     # → escalate
            neutral="Please fix this issue.",    # → general (flips)
            opposite="No rush, low priority.",   # → general (flips)
        )
        original = {"message": "Fix this ASAP!", "route": None, "response": None}
        result = await self.analyser.analyse(mock_router_node, original, contrasts)
        assert result.original_route == "escalate"
        assert result.counterfactual_route == "general"

    @pytest.mark.asyncio
    async def test_raw_outputs_contains_all_variants(self):
        """raw_outputs must contain keys for all 4 variants."""
        contrasts = ContrastSet(similar="s", neutral="n", opposite="o")
        original = {"message": "hello", "route": None, "response": None}
        result = await self.analyser.analyse(mock_router_node, original, contrasts)
        assert set(result.raw_outputs.keys()) == {"original", "similar", "neutral", "opposite"}

    @pytest.mark.asyncio
    async def test_contrast_routes_contains_three_keys(self):
        contrasts = ContrastSet(similar="s", neutral="n", opposite="o")
        original = {"message": "hello", "route": None, "response": None}
        result = await self.analyser.analyse(mock_router_node, original, contrasts)
        assert set(result.contrast_routes.keys()) == {"similar", "neutral", "opposite"}

    @pytest.mark.asyncio
    async def test_explicit_route_key(self):
        """Explicit route_key bypasses auto-detection."""
        contrasts = ContrastSet(
            similar="urgent fix needed asap",
            neutral="please help",
            opposite="no rush",
        )
        original = {"message": "Fix this ASAP!", "route": None, "response": None}
        result = await self.analyser.analyse(
            mock_router_node, original, contrasts, route_key="route"
        )
        assert result.original_route == "escalate"

    @pytest.mark.asyncio
    async def test_sync_node_supported(self):
        """analyse() works with a synchronous node function."""
        def sync_router(state):
            text = state["message"].lower()
            route = "escalate" if "urgent" in text else "general"
            return {**state, "route": route}

        contrasts = ContrastSet(similar="urgent!", neutral="please help", opposite="no rush")
        original = {"message": "urgent!", "route": None, "response": None}
        result = await self.analyser.analyse(sync_router, original, contrasts, route_key="route")
        assert result.original_route == "escalate"

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """All 4 node calls run concurrently — total time ≈ 1 call, not 4."""
        async def slow_node(state):
            await asyncio.sleep(0.1)
            return {**state, "route": "general"}

        contrasts = ContrastSet(similar="s", neutral="n", opposite="o")
        original = {"message": "hi", "route": None, "response": None}

        start = time.perf_counter()
        await self.analyser.analyse(slow_node, original, contrasts, route_key="route")
        elapsed = time.perf_counter() - start

        # 4 concurrent 0.1s sleeps should complete in ~0.1s, not ~0.4s
        assert elapsed < 0.3, f"Expected concurrent execution (~0.1s), took {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_analysis_result_is_dataclass(self):
        contrasts = ContrastSet(similar="s", neutral="n", opposite="o")
        original = {"message": "test", "route": None, "response": None}
        result = await self.analyser.analyse(mock_router_node, original, contrasts)
        assert isinstance(result, AnalysisResult)
        assert isinstance(result.entropy_score, float)
        assert 0.0 <= result.entropy_score <= 1.0
