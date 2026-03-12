"""
Contrail — Decision Path Tracer for LangGraph Agents.

Wraps LangGraph nodes to measure routing stability via contrastive analysis.
Produces per-node entropy scores and plain-language attribution explanations.

Quick start:
    from contrail import trace_node, trace_graph, ContrailConfig

    # Wrap a single node:
    @trace_node(config=ContrailConfig(export_format="stdout"))
    async def my_router_node(state): ...

    # Wrap an entire graph:
    graph = trace_graph(compiled_graph, config=ContrailConfig(sample_rate=0.2))
"""
from contrail.config import ContrailConfig
from contrail.wrap import trace_graph, trace_node

__all__ = ["trace_node", "trace_graph", "ContrailConfig"]
__version__ = "0.1.0"
