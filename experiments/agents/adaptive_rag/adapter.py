"""
Adaptive RAG adapter for the CPE-GEPA optimization loop.
"""
from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from testing.harness.fixtures import ADAPTIVE_RAG_INPUTS

DEFAULT_SYSTEM_PROMPT = """\
Select a retrieval strategy for this query.
Respond with only the strategy name, nothing else.

Strategies:
- direct_answer: general knowledge, definitions, facts from training data
- vector_search: questions about internal company documents, reports, or data
- web_search: current events, recent news, real-time information"""


class PromptHolder:
    def __init__(self, prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
        self.system_prompt = prompt


class RAGState(TypedDict):
    query: str
    strategy: str | None
    answer: str | None


def _get_llm(model=None):
    from conntrail.utils.providers import get_chat_model
    return get_chat_model(model or "claude-haiku-4-5-20251001", max_tokens=20)


def make_routing_node(holder: PromptHolder, model=None):
    async def route_query(state: RAGState) -> RAGState:
        llm = _get_llm(model)
        response = await llm.ainvoke([
            SystemMessage(content=holder.system_prompt),
            HumanMessage(content=state["query"]),
        ])
        strategy = response.content.strip().lower()
        if strategy not in ("direct_answer", "vector_search", "web_search"):
            strategy = "direct_answer"
        return {**state, "strategy": strategy}
    route_query.__name__ = "route_query"
    return route_query


async def _handler(state: RAGState) -> RAGState:
    return {**state, "answer": f"Answered via: {state['strategy']}"}


def _route(state: RAGState) -> str:
    return state["strategy"]


def build_graph(holder: PromptHolder | None = None, model=None):
    if holder is None:
        holder = PromptHolder()
    builder = StateGraph(RAGState)
    builder.add_node("route_query", make_routing_node(holder, model=model))
    for strategy in ("direct_answer", "vector_search", "web_search"):
        builder.add_node(strategy, _handler)
        builder.add_edge(strategy, END)
    builder.add_edge(START, "route_query")
    builder.add_conditional_edges(
        "route_query", _route,
        {s: s for s in ("direct_answer", "vector_search", "web_search")},
    )
    return builder.compile()


INPUT_KEY = "query"
ROUTE_KEY = "strategy"
ONLY_NODES = {"route_query"}
VALID_ROUTES = ("direct_answer", "vector_search", "web_search")


def make_initial_state(input_text: str) -> dict:
    return {"query": input_text, "strategy": None, "answer": None}


def get_result_route(result: dict) -> str:
    return result.get("strategy", "unknown") or "unknown"


TRAINSET = [
    {"input": ADAPTIVE_RAG_INPUTS[0].text, "expected_route": "direct_answer", "entropy_category": "confident"},
    {"input": ADAPTIVE_RAG_INPUTS[1].text, "expected_route": "vector_search", "entropy_category": "confident"},
    {"input": ADAPTIVE_RAG_INPUTS[2].text, "expected_route": "vector_search", "entropy_category": "boundary"},
    {"input": ADAPTIVE_RAG_INPUTS[3].text, "expected_route": "web_search",    "entropy_category": "fragile"},
]

TRAINSET_LARGE = TRAINSET + [
    # confident direct_answer
    {"input": "What is the chemical formula for water?",                                        "expected_route": "direct_answer", "entropy_category": "confident"},
    {"input": "Who invented the telephone?",                                                    "expected_route": "direct_answer", "entropy_category": "confident"},
    {"input": "How many continents are there on Earth?",                                        "expected_route": "direct_answer", "entropy_category": "confident"},
    {"input": "What is the definition of machine learning?",                                    "expected_route": "direct_answer", "entropy_category": "confident"},
    {"input": "What is the capital of Japan?",                                                  "expected_route": "direct_answer", "entropy_category": "confident"},
    # confident vector_search
    {"input": "What does our Q3 revenue report say about APAC performance?",                    "expected_route": "vector_search", "entropy_category": "confident"},
    {"input": "What are the company's parental leave and vacation policies?",                    "expected_route": "vector_search", "entropy_category": "confident"},
    {"input": "What is our SLA commitment for Tier 1 support tickets?",                         "expected_route": "vector_search", "entropy_category": "confident"},
    {"input": "What were the key decisions from last quarter's board meeting?",                  "expected_route": "vector_search", "entropy_category": "confident"},
    # confident web_search
    {"input": "What is the current price of Bitcoin?",                                          "expected_route": "web_search",    "entropy_category": "confident"},
    {"input": "What happened in the technology sector news today?",                             "expected_route": "web_search",    "entropy_category": "confident"},
    {"input": "What are the latest travel entry requirements for Japan?",                        "expected_route": "web_search",    "entropy_category": "confident"},
    # boundary
    {"input": "What is the current unemployment rate in the United States?",                    "expected_route": "web_search",    "entropy_category": "boundary"},
    {"input": "What is the standard formula for calculating customer churn rate?",              "expected_route": "direct_answer", "entropy_category": "boundary"},
    # fragile
    {"input": "What's our main competitor doing differently in the market this quarter?",        "expected_route": "web_search",    "entropy_category": "fragile"},
    {"input": "What were the results of our most recent employee performance review cycle?",     "expected_route": "vector_search", "entropy_category": "fragile"},
]
