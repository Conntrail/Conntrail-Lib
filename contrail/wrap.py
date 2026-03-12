"""
Public API — trace_node decorator and trace_graph function.

Two integration patterns:

  Pattern 1: decorator on a single node
    @trace_node(config=ContrailConfig())
    async def my_router_node(state): ...

  Pattern 2: wrap an entire compiled graph
    graph = trace_graph(compiled_graph, config=ContrailConfig(sample_rate=0.2))
"""
from __future__ import annotations

import functools
from typing import Any, Callable

from contrail.config import ContrailConfig
from contrail.interceptor import NodeInterceptor


def trace_node(config: ContrailConfig | None = None) -> Callable:
    """
    Decorator that wraps a single LangGraph node with Contrail tracing.

    Args:
        config: ContrailConfig. Defaults to ContrailConfig() if not provided.

    Usage:
        @trace_node()
        async def my_node(state): ...

        @trace_node(config=ContrailConfig(sample_rate=0.5))
        async def my_node(state): ...
    """
    resolved_config = config or ContrailConfig()

    def decorator(fn: Callable) -> Callable:
        node_id = getattr(fn, "__name__", "unknown_node")
        interceptor = NodeInterceptor(fn, node_id=node_id, config=resolved_config)

        @functools.wraps(fn)
        async def wrapper(state: dict[str, Any]) -> dict[str, Any]:
            return await interceptor(state)

        return wrapper

    return decorator


def trace_graph(compiled_graph: Any, config: ContrailConfig | None = None) -> Any:
    """
    Wrap all nodes in a compiled LangGraph with Contrail tracing.

    Introspects the compiled graph's node list, wraps each node with a
    NodeInterceptor, and returns the modified graph. Does not recompile.

    Args:
        compiled_graph: A LangGraph CompiledStateGraph instance.
        config: ContrailConfig. Defaults to ContrailConfig() if not provided.

    Returns:
        The same compiled_graph with all nodes wrapped.
    """
    raise NotImplementedError("Phase 5")
