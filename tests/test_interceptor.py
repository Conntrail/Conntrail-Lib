"""
Tests for NodeInterceptor and the public API (Phase 5).
"""
import pytest

from contrail.config import ContrailConfig
from contrail.interceptor import NodeInterceptor


class TestContrailConfig:
    """ContrailConfig validation is implemented now — test it."""

    def test_default_values(self):
        config = ContrailConfig()
        assert config.contrast_model == "claude-haiku-4-5-20251001"
        assert config.sample_rate == 1.0
        assert config.async_mode is True
        assert config.export_format == "jsonl"
        assert config.entropy_alert_threshold == 0.6

    def test_invalid_sample_rate_raises(self):
        with pytest.raises(ValueError, match="sample_rate"):
            ContrailConfig(sample_rate=1.5)

    def test_zero_sample_rate_valid(self):
        config = ContrailConfig(sample_rate=0.0)
        assert config.sample_rate == 0.0

    def test_invalid_export_format_raises(self):
        with pytest.raises(ValueError, match="export_format"):
            ContrailConfig(export_format="invalid")

    def test_invalid_alert_threshold_raises(self):
        with pytest.raises(ValueError, match="entropy_alert_threshold"):
            ContrailConfig(entropy_alert_threshold=1.5)


class TestNodeInterceptor:
    def test_instantiation(self):
        async def dummy_node(state):
            return state

        interceptor = NodeInterceptor(dummy_node, node_id="test_node", config=ContrailConfig())
        assert interceptor.node_id == "test_node"
        assert interceptor.__name__ == "test_node"

    @pytest.mark.asyncio
    async def test_call_not_implemented_yet(self):
        async def dummy_node(state):
            return state

        interceptor = NodeInterceptor(dummy_node, node_id="test", config=ContrailConfig())
        with pytest.raises(NotImplementedError):
            await interceptor({"message": "hello"})

    @pytest.mark.asyncio
    async def test_hot_path_not_blocked(self):
        """PHASE 5: wrapped node returns in same wall time as unwrapped."""
        pytest.skip("Implement in Phase 5")

    @pytest.mark.asyncio
    async def test_output_unchanged(self):
        """PHASE 5: wrapped node output identical to unwrapped output."""
        pytest.skip("Implement in Phase 5")

    @pytest.mark.asyncio
    async def test_zero_sample_rate_fires_no_analysis(self):
        """PHASE 5: sample_rate=0.0 → no contrast analysis fired."""
        pytest.skip("Implement in Phase 5")


class TestPublicAPI:
    def test_trace_node_import(self):
        from contrail import trace_node
        assert callable(trace_node)

    def test_trace_graph_import(self):
        from contrail import trace_graph
        assert callable(trace_graph)

    def test_contrail_config_import(self):
        from contrail import ContrailConfig
        assert ContrailConfig is not None

    def test_trace_graph_not_implemented(self):
        """PHASE 5: trace_graph wraps compiled graph nodes."""
        from contrail import trace_graph
        with pytest.raises(NotImplementedError):
            trace_graph(object(), config=ContrailConfig())

    @pytest.mark.asyncio
    async def test_trace_node_decorator_not_implemented(self):
        """PHASE 5: @trace_node() wraps a node with an interceptor."""
        from contrail import trace_node
        with pytest.raises(NotImplementedError):

            @trace_node()
            async def my_node(state):
                return state

            await my_node({"message": "test"})
