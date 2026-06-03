"""
Customer Support adapter for the CPE-GEPA optimization loop.

Exposes a configurable system prompt via PromptHolder so the routing node
can be updated between iterations without rebuilding the graph.
"""
from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from testing.harness.fixtures import CUSTOMER_SUPPORT_INPUTS

# ── Prompt holder ─────────────────────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = """\
Classify the customer query into exactly one category.
Respond with only the category name, nothing else.

Categories:
- refund: customer wants a refund, return, or money back
- escalation: customer is angry, wants a manager, or situation is urgent/critical
- order_info: customer asking about order status, tracking, shipping
- general: any other query"""


class PromptHolder:
    def __init__(self, prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
        self.system_prompt = prompt


# ── State ─────────────────────────────────────────────────────────────────────

class SupportState(TypedDict):
    message: str
    category: str | None
    response: str | None


# ── Nodes ─────────────────────────────────────────────────────────────────────

def _get_llm():
    from langchain_ollama import ChatOllama
    return ChatOllama(model="qwen2.5:7b", num_predict=20, temperature=0.0)


def make_routing_node(holder: PromptHolder):
    async def classify_query(state: SupportState) -> SupportState:
        llm = _get_llm()
        response = await llm.ainvoke([
            SystemMessage(content=holder.system_prompt),
            HumanMessage(content=state["message"]),
        ])
        category = response.content.strip().lower()
        if category not in ("refund", "escalation", "order_info", "general"):
            category = "general"
        return {**state, "category": category}
    classify_query.__name__ = "classify_query"
    return classify_query


async def _handle(state: SupportState) -> SupportState:
    return {**state, "response": f"Handled: {state['category']}"}


def _route(state: SupportState) -> str:
    return f"handle_{state['category']}"


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph(holder: PromptHolder | None = None):
    if holder is None:
        holder = PromptHolder()
    builder = StateGraph(SupportState)
    builder.add_node("classify_query", make_routing_node(holder))
    for cat in ("refund", "escalation", "order_info", "general"):
        builder.add_node(f"handle_{cat}", _handle)
        builder.add_edge(f"handle_{cat}", END)
    builder.add_edge(START, "classify_query")
    builder.add_conditional_edges(
        "classify_query", _route,
        {f"handle_{c}": f"handle_{c}" for c in ("refund", "escalation", "order_info", "general")},
    )
    return builder.compile()


# ── Experiment config ──────────────────────────────────────────────────────────

INPUT_KEY = "message"
ROUTE_KEY = "category"
ONLY_NODES = {"classify_query"}
VALID_ROUTES = ("refund", "escalation", "order_info", "general")


def make_initial_state(input_text: str) -> dict:
    return {"message": input_text, "category": None, "response": None}


def get_result_route(result: dict) -> str:
    return result.get("category", "unknown") or "unknown"


# TRAINSET: drawn from fixtures with expected routes added.
# Expected route = the dominant route the model takes for each input.
TRAINSET = [
    {"input": CUSTOMER_SUPPORT_INPUTS[0].text, "expected_route": "general",     "entropy_category": "confident"},
    {"input": CUSTOMER_SUPPORT_INPUTS[1].text, "expected_route": "order_info",  "entropy_category": "confident"},
    {"input": CUSTOMER_SUPPORT_INPUTS[2].text, "expected_route": "refund",      "entropy_category": "boundary"},
    {"input": CUSTOMER_SUPPORT_INPUTS[3].text, "expected_route": "refund",      "entropy_category": "boundary"},
    {"input": CUSTOMER_SUPPORT_INPUTS[4].text, "expected_route": "refund",      "entropy_category": "fragile"},
    {"input": CUSTOMER_SUPPORT_INPUTS[5].text, "expected_route": "refund",      "entropy_category": "fragile"},
]
