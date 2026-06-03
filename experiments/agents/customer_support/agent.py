"""
Customer Support Agent — adapted from langchain-ai/langgraph examples.

Routes incoming queries to: refund | escalation | order_info | general
Provider-agnostic: works with Groq, Anthropic, or OpenAI.

Source: https://github.com/langchain-ai/langgraph/tree/main/examples/customer-support
"""
from __future__ import annotations

import os
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph


class SupportState(TypedDict):
    message: str
    category: str | None
    response: str | None


def _get_llm():
    from testing.harness.llm import get_llm
    return get_llm()


# --- Routing node ---
async def classify_query(state: SupportState) -> SupportState:
    """Classify the incoming query into a support category."""
    llm = _get_llm()
    response = await llm.ainvoke([
        SystemMessage(content="""Classify the customer query into exactly one category.
Respond with only the category name, nothing else.

Categories:
- refund: customer wants a refund, return, or money back
- escalation: customer is angry, wants a manager, or situation is urgent/critical
- order_info: customer asking about order status, tracking, shipping
- general: any other query"""),
        HumanMessage(content=state["message"]),
    ])
    category = response.content.strip().lower()
    if category not in ("refund", "escalation", "order_info", "general"):
        category = "general"
    return {**state, "category": category}


def route_query(state: SupportState) -> Literal["handle_refund", "handle_escalation", "handle_order_info", "handle_general"]:
    return f"handle_{state['category']}"


# --- Handler nodes ---
async def handle_refund(state: SupportState) -> SupportState:
    response = await _get_llm().ainvoke([
        SystemMessage(content="You handle refund requests. Be helpful and process-oriented."),
        HumanMessage(content=state["message"]),
    ])
    return {**state, "response": response.content}


async def handle_escalation(state: SupportState) -> SupportState:
    response = await _get_llm().ainvoke([
        SystemMessage(content="You handle escalations. Be empathetic and de-escalate."),
        HumanMessage(content=state["message"]),
    ])
    return {**state, "response": response.content}


async def handle_order_info(state: SupportState) -> SupportState:
    response = await _get_llm().ainvoke([
        SystemMessage(content="You handle order inquiries. Be precise and informative."),
        HumanMessage(content=state["message"]),
    ])
    return {**state, "response": response.content}


async def handle_general(state: SupportState) -> SupportState:
    response = await _get_llm().ainvoke([
        SystemMessage(content="You handle general support queries. Be friendly and helpful."),
        HumanMessage(content=state["message"]),
    ])
    return {**state, "response": response.content}


# --- Graph builder ---
def build_graph():
    builder = StateGraph(SupportState)

    builder.add_node("classify_query", classify_query)
    builder.add_node("handle_refund", handle_refund)
    builder.add_node("handle_escalation", handle_escalation)
    builder.add_node("handle_order_info", handle_order_info)
    builder.add_node("handle_general", handle_general)

    builder.add_edge(START, "classify_query")
    builder.add_conditional_edges(
        "classify_query",
        route_query,
        {
            "handle_refund": "handle_refund",
            "handle_escalation": "handle_escalation",
            "handle_order_info": "handle_order_info",
            "handle_general": "handle_general",
        },
    )
    builder.add_edge("handle_refund", END)
    builder.add_edge("handle_escalation", END)
    builder.add_edge("handle_order_info", END)
    builder.add_edge("handle_general", END)

    return builder.compile()


if __name__ == "__main__":
    import asyncio

    async def main():
        graph = build_graph()
        result = await graph.ainvoke({
            "message": "I need a refund for my order, it never arrived!",
            "category": None,
            "response": None,
        })
        print(f"Category: {result['category']}")
        print(f"Response: {result['response']}")

    asyncio.run(main())
