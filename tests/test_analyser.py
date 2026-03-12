"""
Tests for DivergenceAnalyser and entropy utilities (Phase 3).
"""
import pytest

from contrail.analyser import AnalysisResult, DivergenceAnalyser
from contrail.utils.entropy import routing_entropy


class TestRoutingEntropy:
    """routing_entropy() is implemented in Phase 1 — test it now."""

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


class TestCosineSimilarity:
    def test_identical_vectors(self):
        from contrail.utils.embedding import cosine_similarity
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        from contrail.utils.embedding import cosine_similarity
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        from contrail.utils.embedding import cosine_similarity
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_length_mismatch_raises(self):
        from contrail.utils.embedding import cosine_similarity
        with pytest.raises(ValueError):
            cosine_similarity([1.0], [1.0, 2.0])


class TestDivergenceAnalyser:
    def test_instantiation(self):
        analyser = DivergenceAnalyser()
        assert analyser.COSINE_THRESHOLD == 0.3

    @pytest.mark.asyncio
    async def test_analyse_not_implemented_yet(self):
        from contrail.contrast import ContrastSet
        analyser = DivergenceAnalyser()
        with pytest.raises(NotImplementedError):
            await analyser.analyse(
                node_fn=lambda s: s,
                original_input={"message": "test"},
                contrast_set=ContrastSet("a", "b", "c"),
            )

    @pytest.mark.asyncio
    async def test_urgent_input_high_entropy(self):
        """PHASE 3: urgent input against mock_router_node → entropy > 0.5."""
        pytest.skip("Implement in Phase 3")

    @pytest.mark.asyncio
    async def test_attribution_dimension_detected(self):
        """PHASE 3: attribution dimension is non-empty string."""
        pytest.skip("Implement in Phase 3")

    @pytest.mark.asyncio
    async def test_counterfactual_route_set(self):
        """PHASE 3: counterfactual_route is the opposite variant's route."""
        pytest.skip("Implement in Phase 3")

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """PHASE 3: all 4 node calls run concurrently (< 2x single call latency)."""
        pytest.skip("Implement in Phase 3")
