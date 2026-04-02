#!/usr/bin/env python3
"""
Example 2: Content Moderation Agent with Audit Trail

This example demonstrates:
- Building a content moderation agent with safety-critical routing
- Using export_format="jsonl" for persistent audit trails
- Configuring sample_rate (0.5) to balance cost vs coverage
- Alert handling for fragile decisions in safety contexts
- Edge case testing with borderline content

The agent routes content to: approve, flag_for_review, or reject.

Safety-Critical Considerations:
- False negatives (approving harmful content) are dangerous
- False positives (rejecting benign content) hurt user experience
- FRAGILE decisions (high entropy) indicate the model is uncertain
- These should always trigger human review in production

Run with:
    uv run examples/content_moderation.py

Output:
- Console: Real-time trace information
- File: ./conntrail_traces/content_moderation_*.jsonl (audit trail)
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


class ModerationState(TypedDict):
    """State for the content moderation agent."""

    content: str  # Content to moderate
    decision: str | None  # Moderation decision
    reason: str | None  # Explanation for decision


# =============================================================================
# Simple Keyword-Based Moderator (Reliable, no API calls needed)
# =============================================================================


def moderate_content(state: ModerationState) -> ModerationState:
    """
    Moderate content and decide: approve, flag_for_review, or reject.

    Uses keyword matching with confidence levels:
    - HIGH confidence violations → reject
    - HIGH confidence safe content → approve
    - BORDERLINE or ambiguous → flag_for_review

    In production, this would use a fine-tuned moderation model.
    """
    content = state["content"].lower()

    # HIGH confidence violations - clear harmful content
    severe_violations = [
        "kill",
        "murder",
        "attack",
        "bomb",
        "terrorist",
        "child abuse",
        "illegal drugs",
        "hate speech",
    ]
    if any(term in content for term in severe_violations):
        return {**state, "decision": "reject", "reason": "High-confidence violation detected"}

    # HIGH confidence safe content - clearly benign
    safe_indicators = [
        "hello",
        "thank you",
        "please help",
        "how do i",
        "what is",
        "recipe for",
        "weather today",
    ]
    if any(term in content for term in safe_indicators):
        return {**state, "decision": "approve", "reason": "High-confidence safe content"}

    # BORDERLINE cases - context-dependent, needs human review
    # These are intentionally ambiguous and should show higher entropy
    borderline_terms = [
        "stupid",
        "idiot",
        "dumb",
        "hate",
        "annoying",
        "political",
        "controversial",
        "opinion",
        "debate",
    ]
    if any(term in content for term in borderline_terms):
        return {
            **state,
            "decision": "flag",
            "reason": "Borderline content - requires human review",
        }

    # Default: moderate confidence, flag for review
    return {
        **state,
        "decision": "flag",
        "reason": "Unable to classify with high confidence",
    }


def route_decision(
    state: ModerationState,
) -> Literal["process_approve", "process_flag", "process_reject"]:
    """Route to the appropriate action based on moderation decision."""
    return f"process_{state['decision']}"


# =============================================================================
# Action Nodes
# =============================================================================


def process_approve(state: ModerationState) -> ModerationState:
    """Process approved content."""
    return {**state, "reason": f"✅ APPROVED: {state['reason']}"}


def process_flag(state: ModerationState) -> ModerationState:
    """Process flagged content for human review."""
    return {**state, "reason": f"⚠️  FLAGGED: {state['reason']}"}


def process_reject(state: ModerationState) -> ModerationState:
    """Process rejected content."""
    return {**state, "reason": f"❌ REJECTED: {state['reason']}"}


# =============================================================================
# Alert Handler for Safety-Critical Decisions
# =============================================================================


def on_safety_alert(record) -> None:
    """
    Handle fragile decisions in content moderation.

    In a production moderation system, you might:
    - Immediately escalate to human moderators
    - Block content until reviewed
    - Log to a security incident system
    - Update model training data
    - Notify compliance team
    """
    print(f"\n{'!' * 70}")
    print("⚠️  SAFETY ALERT: Uncertain Moderation Decision")
    print(f"{'!' * 70}")
    print(f"This content had an UNSTABLE moderation decision.")
    print(f"The model's confidence is low - immediate human review recommended!")
    print(f"\nDetails:")
    print(f"  Node: {record.node_id}")
    print(f"  Decision: {record.original_route}")
    print(f"  Entropy: {record.entropy_score:.3f} (>= 0.5 indicates uncertainty)")
    print(f"  Stability: {record.stability}")
    print(f"  Content: {record.original_input[:100]}...")
    print(f"\nAttribution Dimension: {record.attribution_dimension}")
    print(f"This tells us what type of input variation causes decision changes.")
    print(f"{'!' * 70}\n")


# =============================================================================
# Build and Wrap the Graph
# =============================================================================


def build_graph():
    """Build the content moderation agent graph."""
    builder = StateGraph(ModerationState)

    # Add nodes
    builder.add_node("moderate_content", moderate_content)
    builder.add_node("process_approve", process_approve)
    builder.add_node("process_flag", process_flag)
    builder.add_node("process_reject", process_reject)

    # Add edges
    builder.add_edge(START, "moderate_content")
    builder.add_conditional_edges(
        "moderate_content",
        route_decision,
        {
            "process_approve": "process_approve",
            "process_flag": "process_flag",
            "process_reject": "process_reject",
        },
    )
    builder.add_edge("process_approve", END)
    builder.add_edge("process_flag", END)
    builder.add_edge("process_reject", END)

    return builder.compile()


# =============================================================================
# Main Execution
# =============================================================================


async def main():
    """Run the content moderation agent with Conntrail tracing."""

    print("=" * 70)
    print("Conntrail Example: Content Moderation Agent")
    print("=" * 70)
    print("\nThis example demonstrates safety-critical decision tracing")
    print("with audit trails and fragile decision alerts.\n")

    # Build the graph
    graph = build_graph()

    # Configure Conntrail with jsonl export for audit trail
    config = ConntrailConfig(
        # Export to jsonl files for audit trail
        export_format="jsonl",
        # Directory for trace files
        export_path="./conntrail_traces",
        # Sample 50% for demo (balance cost vs coverage)
        # In production with high volume, use 0.01-0.1
        sample_rate=0.5,
        # Lower threshold for safety-critical applications
        # We want to catch uncertain decisions early
        entropy_alert_threshold=0.5,
        # Alert callback for fragile decisions
        on_alert=on_safety_alert,
        # Use fast model for contrast generation
        contrast_model="groq-llama3-8b-8192",
    )

    # Wrap the graph with Conntrail tracing
    traced_graph = trace_graph(graph, config=config)

    # Test content demonstrating different moderation scenarios
    test_content = [
        # Clear violations - expect LOW entropy, REJECT
        {
            "name": "Clear Violation",
            "content": "I want to kill everyone at the company. This is a threat.",
            "expected_decision": "reject",
            "expected_entropy": "Low (clear violation)",
        },
        # Clear safe content - expect LOW entropy, APPROVE
        {
            "name": "Clearly Safe",
            "content": "Hello! Thank you for the help with my recipe yesterday.",
            "expected_decision": "approve",
            "expected_entropy": "Low (clearly safe)",
        },
        # Borderline case 1: negative language but not severe
        # Expect HIGHER entropy - model might flip between flag/reject
        {
            "name": "Borderline: Negative Language",
            "content": "This stupid policy is so annoying, I hate it.",
            "expected_decision": "flag_for_review",
            "expected_entropy": "Medium-High (sentiment is negative but not threatening)",
        },
        # Borderline case 2: political/controversial
        # Expect HIGHER entropy - context dependent
        {
            "name": "Borderline: Controversial Topic",
            "content": "My political opinion is that this debate is controversial.",
            "expected_decision": "flag_for_review",
            "expected_entropy": "Medium-High (context-dependent classification)",
        },
        # Ambiguous case: vague complaint
        # Expect potentially HIGHER entropy
        {
            "name": "Ambiguous Complaint",
            "content": "Something is wrong and I feel dumb for asking.",
            "expected_decision": "flag_for_review",
            "expected_entropy": "Medium (mild self-deprecation, unclear severity)",
        },
    ]

    print(f"Export path: {config.export_path}")
    print(f"Sample rate: {config.sample_rate} (50% of decisions will be traced)")
    print(f"Alert threshold: {config.entropy_alert_threshold}")
    print("\n" + "─" * 70 + "\n")

    for test in test_content:
        print(f"\n{'─' * 70}")
        print(f"Test: {test['name']}")
        print(f"Content: {test['content']}")
        print(f"Expected Decision: {test['expected_decision']}")
        print(f"Expected Entropy: {test['expected_entropy']}")
        print(f"{'─' * 70}\n")

        # Invoke the traced graph
        result = await traced_graph.ainvoke(
            {
                "content": test["content"],
                "decision": None,
                "reason": None,
            }
        )

        print(f"\nResult: {result['decision']}")
        print(f"Reason: {result['reason']}")
        print("\n" + "=" * 70)
        await asyncio.sleep(0.5)

    print("\n✅ Content moderation tests completed!")
    print("\nKey Takeaways:")
    print("- Safety-critical systems need LOWER entropy thresholds")
    print("- jsonl export creates persistent audit trails for compliance")
    print("- sample_rate balances cost vs observability coverage")
    print("- Borderline content naturally has higher entropy - this is expected!")
    print("\nAudit trails saved to: ./conntrail_traces/")
    print("Review the .jsonl files to see the full trace records.")


if __name__ == "__main__":
    asyncio.run(main())
