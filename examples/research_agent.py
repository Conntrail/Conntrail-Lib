#!/usr/bin/env python3
"""
Example 3: Research Agent with Selective Tracing and Programmatic Access

This example demonstrates:
- Multi-step research agent with multiple routing nodes
- Using `only_nodes` parameter to trace only specific routing nodes
- Using `async_mode=False` to access traces programmatically after execution
- Printing trace records to analyze decision stability

The agent has two routing decisions:
1. query_classifier: Routes to factual, opinion, or complex
2. source_selector: Routes to web, knowledge_base, or both

This example shows how Conntrail can be used for:
- Debugging agent behavior during development
- Analyzing which routing decisions are stable vs fragile
- Building custom dashboards from trace data

Run with:
    uv run examples/research_agent.py

Features demonstrated:
- only_nodes={"query_classifier", "source_selector"} - trace just the routers
- async_mode=False - synchronous trace collection
- Programmatic access to trace records after execution
"""

from __future__ import annotations

import asyncio
import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

# Import Conntrail components
from conntrail import trace_graph, ConntrailConfig

# Load environment variables from testing/.env
env_path = os.path.join(os.path.dirname(__file__), "..", "testing", ".env")
load_dotenv(env_path)


# =============================================================================
# State Definition
# =============================================================================


class ResearchState(TypedDict):
    """State for the multi-step research agent."""

    query: str  # User's research query
    query_type: str | None  # Classified type: factual, opinion, complex
    source_strategy: str | None  # Selected sources: web, knowledge_base, both
    result: str | None  # Research result


# =============================================================================
# Node 1: Query Classifier
# =============================================================================


def query_classifier(state: ResearchState) -> ResearchState:
    """
    Classify the research query type.

    Routes:
    - factual: Questions with objective answers ("What is the capital of France?")
    - opinion: Subjective questions ("What is the best programming language?")
    - complex: Multi-part or ambiguous questions requiring breakdown

    Uses keyword matching for reliability in this demo.
    """
    query = state["query"].lower()

    # Factual queries - questions with objective answers
    factual_patterns = [
        "what is",
        "what are",
        "when did",
        "where is",
        "who is",
        "how many",
        "how much",
        "define",
        "explain",
    ]
    if any(pattern in query for pattern in factual_patterns):
        # Additional check for subjective terms
        subjective_terms = ["best", "worst", "better", "worse", "opinion", "think"]
        if any(term in query for term in subjective_terms):
            # Ambiguous: could be factual or opinion
            query_type = "complex"
        else:
            query_type = "factual"

    # Opinion queries - subjective questions
    elif any(
        term in query for term in ["best", "worst", "better", "should i", "recommend", "opinion"]
    ):
        query_type = "opinion"

    # Complex queries - multi-part or requiring clarification
    elif any(term in query for term in ["compare", "difference between", "vs", "versus", "and"]):
        query_type = "complex"

    else:
        # Default to complex for unclear queries
        query_type = "complex"

    return {**state, "query_type": query_type}


def route_by_query_type(state: ResearchState) -> Literal["select_sources", "breakdown_complex"]:
    """Route based on query classification."""
    if state["query_type"] == "complex":
        return "breakdown_complex"
    return "select_sources"


# =============================================================================
# Node 2: Complex Query Breakdown (intermediate step)
# =============================================================================


def breakdown_complex(state: ResearchState) -> ResearchState:
    """Break down complex queries into simpler parts."""
    return {**state, "result": f"Breaking down complex query: '{state['query']}'"}


def route_after_breakdown(state: ResearchState) -> Literal["select_sources"]:
    """After breakdown, always go to source selection."""
    return "select_sources"


# =============================================================================
# Node 3: Source Selector
# =============================================================================


def source_selector(state: ResearchState) -> ResearchState:
    """
    Select appropriate sources for the research.

    Routes:
    - web: For time-sensitive or current information
    - knowledge_base: For established facts and documentation
    - both: For comprehensive research needs
    """
    query = state["query"].lower()
    query_type = state.get("query_type", "complex")

    # Web search for current/time-sensitive topics
    web_keywords = [
        "latest",
        "news",
        "recent",
        "today",
        "2024",
        "2025",
        "current",
        "update",
        "trending",
        "price",
        "stock",
    ]
    needs_web = any(kw in query for kw in web_keywords)

    # Knowledge base for established facts
    kb_keywords = [
        "definition",
        "history",
        "theory",
        "algorithm",
        "standard",
        "protocol",
        "specification",
        "documentation",
    ]
    needs_kb = any(kw in query for kw in kb_keywords)

    # Opinion questions often need both
    if query_type == "opinion":
        source_strategy = "both"
    elif needs_web and needs_kb:
        source_strategy = "both"
    elif needs_web:
        source_strategy = "web"
    elif needs_kb:
        source_strategy = "kb"
    else:
        # Default to both for comprehensive coverage
        source_strategy = "both"

    return {**state, "source_strategy": source_strategy}


def route_by_source(state: ResearchState) -> Literal["search_web", "search_kb", "search_both"]:
    """Route based on source selection."""
    return f"search_{state['source_strategy']}"


# =============================================================================
# Research Action Nodes
# =============================================================================


def search_web(state: ResearchState) -> ResearchState:
    """Simulate web search."""
    return {**state, "result": f"[Web Search] Results for: {state['query']}"}


def search_kb(state: ResearchState) -> ResearchState:
    """Simulate knowledge base search."""
    return {**state, "result": f"[Knowledge Base] Results for: {state['query']}"}


def search_both(state: ResearchState) -> ResearchState:
    """Simulate searching both web and knowledge base."""
    return {**state, "result": f"[Web + KB] Comprehensive results for: {state['query']}"}


# =============================================================================
# Build and Wrap the Graph
# =============================================================================


def build_graph():
    """Build the multi-step research agent graph."""
    builder = StateGraph(ResearchState)

    # Add all nodes
    builder.add_node("query_classifier", query_classifier)
    builder.add_node("breakdown_complex", breakdown_complex)
    builder.add_node("source_selector", source_selector)
    builder.add_node("search_web", search_web)
    builder.add_node("search_kb", search_kb)
    builder.add_node("search_both", search_both)

    # Add edges
    builder.add_edge(START, "query_classifier")

    # Query classification routes
    builder.add_conditional_edges(
        "query_classifier",
        route_by_query_type,
        {
            "select_sources": "source_selector",
            "breakdown_complex": "breakdown_complex",
        },
    )

    # After breakdown, go to source selection
    builder.add_conditional_edges(
        "breakdown_complex",
        route_after_breakdown,
        {"select_sources": "source_selector"},
    )

    # Source selection routes
    builder.add_conditional_edges(
        "source_selector",
        route_by_source,
        {
            "search_web": "search_web",
            "search_kb": "search_kb",
            "search_both": "search_both",
        },
    )

    # All searches end
    builder.add_edge("search_web", END)
    builder.add_edge("search_kb", END)
    builder.add_edge("search_both", END)

    return builder.compile()


# =============================================================================
# Trace Analysis Helper
# =============================================================================


def analyze_traces(traces: list) -> None:
    """
    Analyze and display trace records from Conntrail.

    This demonstrates how to programmatically access trace data
    when using async_mode=False.
    """
    if not traces:
        print("\n⚠️  No traces collected (may be due to sampling)")
        return

    print(f"\n{'=' * 70}")
    print(f"📊 TRACE ANALYSIS: {len(traces)} routing decisions traced")
    print(f"{'=' * 70}\n")

    for i, record in enumerate(traces, 1):
        print(f"Trace #{i}")
        print(f"  Node: {record.node_id}")
        print(f"  Input: {record.original_input[:60]}...")
        print(f"  Decision: {record.original_route}")
        print(f"  Entropy Score: {record.entropy_score:.3f}")
        print(f"  Stability: {record.stability}")
        print(f"  Attribution: {record.attribution_dimension}")

        # Interpretation
        if record.entropy_score < 0.3:
            interpretation = "✅ Very stable - model is confident"
        elif record.entropy_score < 0.6:
            interpretation = "⚠️  Moderate stability - monitor this"
        else:
            interpretation = "🚨 Fragile - model is uncertain, review needed"

        print(f"  Interpretation: {interpretation}")
        print()

    # Summary statistics
    if len(traces) > 1:
        avg_entropy = sum(r.entropy_score for r in traces) / len(traces)
        max_entropy = max(r.entropy_score for r in traces)
        min_entropy = min(r.entropy_score for r in traces)

        print(f"Summary Statistics:")
        print(f"  Average Entropy: {avg_entropy:.3f}")
        print(f"  Min Entropy: {min_entropy:.3f}")
        print(f"  Max Entropy: {max_entropy:.3f}")

        if avg_entropy < 0.3:
            print(f"  Overall Assessment: Agent decisions are STABLE ✅")
        elif avg_entropy < 0.6:
            print(f"  Overall Assessment: Agent decisions are MODERATELY STABLE ⚠️")
        else:
            print(f"  Overall Assessment: Agent decisions are UNSTABLE 🚨")

    print(f"\n{'=' * 70}\n")


# =============================================================================
# Main Execution
# =============================================================================


async def main():
    """Run the research agent with selective Conntrail tracing."""

    print("=" * 70)
    print("Conntrail Example: Research Agent with Selective Tracing")
    print("=" * 70)
    print("\nThis example demonstrates:")
    print("- Tracing ONLY specific routing nodes (query_classifier, source_selector)")
    print("- async_mode=False for programmatic trace access")
    print("- Multi-step agent with multiple routing decisions\n")

    # Build the graph
    graph = build_graph()

    # Configure Conntrail with selective tracing
    config = ConntrailConfig(
        export_format="stdout",
        sample_rate=1.0,  # Trace all calls for demo
        entropy_alert_threshold=0.6,
        contrast_model="groq-llama3-8b-8192",
        # async_mode=False allows us to collect traces programmatically
        async_mode=False,
    )

    # Wrap ONLY the routing nodes with Conntrail
    # This avoids tracing handler nodes (search_web, search_kb, etc.)
    traced_graph = trace_graph(
        graph, config=config, only_nodes={"query_classifier", "source_selector"}
    )

    # Test queries demonstrating different routing paths
    test_queries = [
        # Clear factual query - expect factual → both
        {
            "name": "Clear Factual Query",
            "query": "What is the capital of France and its population?",
            "expected_path": "factual → both",
        },
        # Clear opinion query - expect opinion → both
        {
            "name": "Opinion Query",
            "query": "What is the best programming language for beginners?",
            "expected_path": "opinion → both",
        },
        # Time-sensitive query - expect factual → web
        {
            "name": "Time-Sensitive Query",
            "query": "What is the latest news about AI today?",
            "expected_path": "factual → web",
        },
        # Complex multi-part query - expect complex → both
        {
            "name": "Complex Query",
            "query": "Compare and contrast Python vs JavaScript",
            "expected_path": "complex → both",
        },
        # Ambiguous query - could be interpreted differently
        {
            "name": "Ambiguous Query",
            "query": "Tell me about machine learning",
            "expected_path": "factual or complex → both",
        },
    ]

    print(f"Tracing nodes: query_classifier, source_selector")
    print(f"async_mode: False (traces returned in result)")
    print(f"\n{'─' * 70}\n")

    for test in test_queries:
        print(f"\n{'─' * 70}")
        print(f"Test: {test['name']}")
        print(f"Query: {test['query']}")
        print(f"Expected Path: {test['expected_path']}")
        print(f"{'─' * 70}\n")

        # Invoke the traced graph
        # With async_mode=False, traces are included in the result
        result = await traced_graph.ainvoke(
            {
                "query": test["query"],
                "query_type": None,
                "source_strategy": None,
                "result": None,
            }
        )

        print(f"\nResult:")
        print(f"  Query Type: {result['query_type']}")
        print(f"  Source Strategy: {result['source_strategy']}")
        print(f"  Output: {result['result']}")

        # Access traces programmatically
        # With async_mode=False, traces are injected into the result
        trace_key = "__conntrail_traces__"
        if trace_key in result:
            traces = result[trace_key]
            analyze_traces(traces)
        else:
            print("\n⚠️  No traces in result (may have been sampled out)")

        print("=" * 70)
        await asyncio.sleep(0.5)

    print("\n✅ Research agent tests completed!")
    print("\nKey Takeaways:")
    print("- Use `only_nodes` to trace specific routing nodes only")
    print("- Use `async_mode=False` to access traces programmatically")
    print("- Multi-step agents have multiple decision points to monitor")
    print("- Trace data helps identify which routing decisions need attention")
    print("\nThis pattern is useful for:")
    print("  - Building custom monitoring dashboards")
    print("  - Debugging agent routing behavior")
    print("  - Collecting training data for model improvements")


if __name__ == "__main__":
    asyncio.run(main())
