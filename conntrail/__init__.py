"""
Conntrail — Decision Path Tracer for LangGraph Agents.

Wraps LangGraph nodes to measure routing stability via contrastive analysis.
Produces per-node entropy scores and plain-language attribution explanations.

Quick start:
    from conntrail import trace_node, trace_graph, ConntrailConfig

    # Wrap a single node:
    @trace_node(config=ConntrailConfig(export_format="stdout"))
    async def my_router_node(state): ...

    # Wrap an entire graph:
    graph = trace_graph(compiled_graph, config=ConntrailConfig(sample_rate=0.2))
"""
from conntrail.config import ConntrailConfig
from conntrail.wrap import trace_graph, trace_node

__all__ = ["trace_node", "trace_graph", "ConntrailConfig"]
__version__ = "0.1.0"
