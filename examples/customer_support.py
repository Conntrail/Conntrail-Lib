#!/usr/bin/env python3
"""
Example 1: Customer Support Agent with Conntrail Tracing

This example demonstrates:
- Building a simple LangGraph agent with routing nodes
- Using trace_graph() with export_format="stdout" for real-time visibility
- Interpreting entropy scores to understand decision stability
- Setting up alert callbacks for fragile decisions (entropy > 0.6)

The agent routes customer queries to: refund, technical_support, billing, or general.

Entropy Score Interpretation:
- 0.0 - 0.3: Confident (stable decision, model consistently routes the same way)
- 0.3 - 0.6: Uncertain (some variation under perturbation, worth monitoring)
- 0.6 - 1.0: Fragile (highly unstable, decision changes with small input changes)

Run with:
    uv run examples/customer_support.py

Or with pip:
    python examples/customer_support.py
"""

from __future__ import annotations

import asyncio
import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

# Import Conntrail components
from conntrail import trace_graph, ConntrailConfig

# Load environment variables from testing/.env (contains GROQ_API_KEY)
# This allows using real LLM APIs if desired
env_path = os.path.join(os.path.dirname(__file__), "..", "testing", ".env")
load_dotenv(env_path)


# =============================================================================
# State Definition
# =============================================================================


class SupportState(TypedDict):
    """State for the customer support agent."""

    message: str  # Customer query
    category: str | None  # Routed category
    response: str | None  # Final response


# =============================================================================
# Simple Keyword-Based Router (Reliable, no API calls needed)
# =============================================================================


def classify_query(state: SupportState) -> SupportState:
    """
    Classify the customer query into a support category.

    Uses simple keyword matching for reliability. In production, this would
    use an LLM API call. The keywords are chosen to demonstrate different
    stability levels when traced with Conntrail.
    """
    message = state["message"].lower()

    # Refund keywords - very clear intent (should be STABLE/CONFIDENT)
    if any(word in message for word in ["refund", "money back", "return", "chargeback"]):
        category = "refund"
    # Technical keywords - also clear (should be STABLE/CONFIDENT)
    elif any(
        word in message for word in ["error", "bug", "crash", "not working", "broken", "technical"]
    ):
        category = "technical"
    # Billing keywords - clear intent (should be STABLE/CONFIDENT)
    elif any(
        word in message
        for word in ["bill", "invoice", "payment", "charged", "subscription", "billing"]
    ):
        category = "billing"
    # Ambiguous cases - borderline queries (may show UNCERTAIN/FRAGILE)
    # These have overlapping keywords that could fit multiple categories
    elif any(word in message for word in ["problem", "issue", "help", "question"]):
        # These are vague and could go to any category - expect higher entropy
        category = "general"
    else:
        category = "general"

    return {**state, "category": category}


def route_query(
    state: SupportState,
) -> Literal["handle_refund", "handle_technical", "handle_billing", "handle_general"]:
    """Route to the appropriate handler based on classification."""
    return f"handle_{state['category']}"


# =============================================================================
# Handler Nodes (Simple responses for demo)
# =============================================================================


def handle_refund(state: SupportState) -> SupportState:
    """Handle refund requests."""
    return {
        **state,
        "response": f"Refund team: Processing your request about '{state['message'][:50]}...'",
    }


def handle_technical(state: SupportState) -> SupportState:
    """Handle technical support queries."""
    return {**state, "response": f"Tech support: Troubleshooting '{state['message'][:50]}...'"}


def handle_billing(state: SupportState) -> SupportState:
    """Handle billing inquiries."""
    return {
        **state,
        "response": f"Billing team: Reviewing charges for '{state['message'][:50]}...'",
    }


def handle_general(state: SupportState) -> SupportState:
    """Handle general queries."""
    return {
        **state,
        "response": f"Support team: General inquiry about '{state['message'][:50]}...'",
    }


# =============================================================================
# Alert Callback
# =============================================================================


def on_fragile_decision(record) -> None:
    """
    Callback fired when a fragile decision is detected (entropy > 0.6).

    In production, you might:
    - Send an alert to Slack/PagerDuty
    - Log to a monitoring system
    - Flag for human review
    - Trigger model retraining
    """
    print(f"\n{'=' * 60}")
    print("🚨 ALERT: Fragile Decision Detected!")
    print(f"{'=' * 60}")
    print(f"Node: {record.node_id}")
    print(f"Input: {record.original_input[:80]}...")
    print(f"Route: {record.original_route}")
    print(f"Entropy Score: {record.entropy_score:.3f} (threshold: 0.6)")
    print(f"Stability: {record.stability}")
    print(f"Attribution: {record.attribution_dimension}")
    print(f"{'=' * 60}\n")


# =============================================================================
# Build and Wrap the Graph
# =============================================================================


def build_graph():
    """Build the customer support agent graph."""
    builder = StateGraph(SupportState)

    # Add nodes
    builder.add_node("classify_query", classify_query)
    builder.add_node("handle_refund", handle_refund)
    builder.add_node("handle_technical", handle_technical)
    builder.add_node("handle_billing", handle_billing)
    builder.add_node("handle_general", handle_general)

    # Add edges
    builder.add_edge(START, "classify_query")
    builder.add_conditional_edges(
        "classify_query",
        route_query,
        {
            "handle_refund": "handle_refund",
            "handle_technical": "handle_technical",
            "handle_billing": "handle_billing",
            "handle_general": "handle_general",
        },
    )
    builder.add_edge("handle_refund", END)
    builder.add_edge("handle_technical", END)
    builder.add_edge("handle_billing", END)
    builder.add_edge("handle_general", END)

    return builder.compile()


# =============================================================================
# Main Execution
# =============================================================================


async def main():
    """Run the customer support agent with Conntrail tracing."""

    print("=" * 70)
    print("Conntrail Example: Customer Support Agent")
    print("=" * 70)
    print("\nThis example shows how Conntrail traces routing decisions")
    print("and alerts on fragile (unstable) decisions.\n")

    # Build the graph
    graph = build_graph()

    # Configure Conntrail with stdout export and alert callback
    config = ConntrailConfig(
        # Use stdout to see traces in real-time
        export_format="stdout",
        # Sample 100% for demo (use 0.1-0.2 in production)
        sample_rate=1.0,
        # Alert threshold for fragile decisions
        entropy_alert_threshold=0.6,
        # Callback fired when entropy >= threshold
        on_alert=on_fragile_decision,
        # Fast/cheap model for contrast generation
        contrast_model="groq-llama3-8b-8192",
    )

    # Wrap the entire graph with Conntrail tracing
    # This patches all nodes in-place without recompiling
    traced_graph = trace_graph(graph, config=config)

    # Test queries demonstrating different stability levels
    test_queries = [
        # Clear refund request - expect LOW entropy (CONFIDENT)
        {
            "name": "Clear Refund Request",
            "message": "I want a refund for my order. The product arrived damaged.",
            "expected": "Low entropy (confident decision)",
        },
        # Clear technical issue - expect LOW entropy (CONFIDENT)
        {
            "name": "Clear Technical Issue",
            "message": "The app crashes every time I try to save. Getting error code 500.",
            "expected": "Low entropy (confident decision)",
        },
        # Ambiguous query - expect HIGHER entropy (UNCERTAIN/FRAGILE)
        # Vague terms like "problem" and "help" could apply to any category
        {
            "name": "Ambiguous Query",
            "message": "I have a problem and need help with something.",
            "expected": "Higher entropy (uncertain decision - vague input)",
        },
    ]

    for test in test_queries:
        print(f"\n{'─' * 70}")
        print(f"Test: {test['name']}")
        print(f"Query: {test['message']}")
        print(f"Expected: {test['expected']}")
        print(f"{'─' * 70}\n")

        # Invoke the traced graph
        result = await traced_graph.ainvoke(
            {
                "message": test["message"],
                "category": None,
                "response": None,
            }
        )

        print(f"\nResult: {result['category']} → {result['response'][:60]}...")
        print("\n" + "=" * 70)
        await asyncio.sleep(0.5)  # Brief pause for readability

    print("\n✅ All tests completed!")
    print("\nKey Takeaways:")
    print("- Clear, specific queries have LOW entropy (stable decisions)")
    print("- Vague, ambiguous queries have HIGH entropy (fragile decisions)")
    print("- The on_alert callback can notify you when models are uncertain")
    print("- Use export_format='jsonl' for audit trails and persistence")


if __name__ == "__main__":
    asyncio.run(main())
