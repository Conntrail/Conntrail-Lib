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

def _get_llm(model=None):
    from conntrail.utils.providers import get_chat_model
    return get_chat_model(model or "claude-haiku-4-5-20251001", max_tokens=20)


def make_routing_node(holder: PromptHolder, model=None):
    async def classify_query(state: SupportState) -> SupportState:
        llm = _get_llm(model)
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

def build_graph(holder: PromptHolder | None = None, model=None):
    if holder is None:
        holder = PromptHolder()
    builder = StateGraph(SupportState)
    builder.add_node("classify_query", make_routing_node(holder, model=model))
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

TRAINSET_LARGE = TRAINSET + [
    # escalation
    {"input": "This is completely unacceptable! I want to speak to a manager right now!", "expected_route": "escalation", "entropy_category": "confident"},
    {"input": "I've been waiting three weeks and nobody is helping me. Get me your supervisor immediately.", "expected_route": "escalation", "entropy_category": "confident"},
    {"input": "I am furious. If this isn't fixed right now I'm disputing every single charge with my bank.", "expected_route": "escalation", "entropy_category": "confident"},
    # order_info
    {"input": "Where is my package? It was supposed to arrive last Tuesday and still nothing.", "expected_route": "order_info", "entropy_category": "confident"},
    {"input": "Can you give me the tracking number for my most recent shipment?", "expected_route": "order_info", "entropy_category": "confident"},
    # general
    {"input": "Do you ship internationally?", "expected_route": "general", "entropy_category": "confident"},
    {"input": "What is your return policy for unopened items?", "expected_route": "general", "entropy_category": "confident"},
    {"input": "Do you offer gift wrapping on orders?", "expected_route": "general", "entropy_category": "confident"},
    {"input": "How do I create an account on your website?", "expected_route": "general", "entropy_category": "confident"},
    # refund
    {"input": "I received completely the wrong item in my order. I need a full refund.", "expected_route": "refund", "entropy_category": "confident"},
    # boundary
    {"input": "My order hasn't arrived and it's been three weeks. Can I cancel it and get my money back?", "expected_route": "refund", "entropy_category": "boundary"},
    {"input": "I am extremely upset. Either process my refund immediately or I will dispute this charge.", "expected_route": "escalation", "entropy_category": "boundary"},
    # fragile
    {"input": "I'm not sure whether to keep this or return it — it works but it's not quite what I expected.", "expected_route": "refund", "entropy_category": "fragile"},
    {"input": "Maybe I should send this back... it's okay but not great for the price.", "expected_route": "refund", "entropy_category": "fragile"},
]
