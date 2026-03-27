"""
BaseTestRunner — runs a LangGraph agent and captures routing information.
Used by all per-agent test files to establish baseline and post-wrap behaviour.
"""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class RunResult:
    """Result of a single agent execution."""
    input: dict
    output: dict
    route_taken: str | None          # final route/node if detectable
    nodes_visited: list[str]         # ordered list of nodes visited
    wall_time_ms: float
    conntrail_traces: list[Any] = field(default_factory=list)  # TraceRecords if present
    error: Exception | None = None


class BaseTestRunner:
    """
    Runs a compiled LangGraph agent and captures execution metadata.

    Usage:
        runner = BaseTestRunner(compiled_graph)
        result = await runner.run({"messages": [HumanMessage(content="...")]})
    """

    def __init__(self, graph, config: dict | None = None):
        self.graph = graph
        self.run_config = config or {}

    async def run(self, input_state: dict) -> RunResult:
        """Execute the graph and return a RunResult."""
        start = time.perf_counter()
        nodes_visited: list[str] = []
        final_output: dict = {}
        error: Exception | None = None

        try:
            # Stream events to capture node visit order
            async for event in self.graph.astream_events(
                input_state,
                version="v2",
                config=self.run_config,
            ):
                if event["event"] == "on_chain_start" and event.get("name"):
                    node_name = event["name"]
                    if node_name not in ("LangGraph", "__start__", "__end__"):
                        nodes_visited.append(node_name)

                if event["event"] == "on_chain_end" and event.get("name") == "LangGraph":
                    final_output = event.get("data", {}).get("output", {})

        except Exception as e:
            error = e

        wall_time_ms = (time.perf_counter() - start) * 1000

        # Extract conntrail traces from output metadata if present
        conntrail_traces = []
        if "__conntrail_traces__" in final_output:
            conntrail_traces = final_output["__conntrail_traces__"]

        # Infer route from last node visited
        route_taken = nodes_visited[-1] if nodes_visited else None

        return RunResult(
            input=input_state,
            output=final_output,
            route_taken=route_taken,
            nodes_visited=nodes_visited,
            wall_time_ms=wall_time_ms,
            conntrail_traces=conntrail_traces,
            error=error,
        )

    def run_sync(self, input_state: dict) -> RunResult:
        """Synchronous wrapper for use in non-async test contexts."""
        return asyncio.run(self.run(input_state))

    async def run_baseline_vs_wrapped(
        self,
        input_state: dict,
        wrap_fn: Callable,
    ) -> tuple[RunResult, RunResult]:
        """
        Run the same input through the agent before and after Conntrail wrapping.
        Returns (baseline_result, wrapped_result) for non-intrusion assertions.
        """
        baseline = await self.run(input_state)
        wrapped_graph = wrap_fn(self.graph)
        wrapped_runner = BaseTestRunner(wrapped_graph, self.run_config)
        wrapped = await wrapped_runner.run(input_state)
        return baseline, wrapped
