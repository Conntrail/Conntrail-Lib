"""
Code Review Router adapter for the CPE-GEPA optimization loop.

Routes pull request descriptions based on change type and risk level.
Routing labels: security_review | performance_review | style_check | architecture_review
"""
from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

DEFAULT_SYSTEM_PROMPT = """\
Route this code review request to the appropriate review type.
Respond with only the category name, nothing else.

Categories:
- security_review: authentication, authorization, cryptography, injection vulnerabilities, secret handling
- performance_review: query optimization, caching, algorithms, indexing, resource usage
- style_check: naming conventions, formatting, comments, documentation, imports, dead code removal
- architecture_review: API design, schema changes, service boundaries, interfaces, major structural changes"""


class PromptHolder:
    def __init__(self, prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
        self.system_prompt = prompt


class ReviewState(TypedDict):
    pr_description: str
    review_type: str | None
    result: str | None


def _get_llm(model=None):
    from conntrail.utils.providers import get_chat_model
    return get_chat_model(model or "claude-haiku-4-5-20251001", max_tokens=20)


def make_routing_node(holder: PromptHolder, model=None):
    async def route_review(state: ReviewState) -> ReviewState:
        llm = _get_llm(model)
        response = await llm.ainvoke([
            SystemMessage(content=holder.system_prompt),
            HumanMessage(content=state["pr_description"]),
        ])
        review_type = response.content.strip().lower()
        if review_type not in ("security_review", "performance_review", "style_check", "architecture_review"):
            review_type = "style_check"
        return {**state, "review_type": review_type}
    route_review.__name__ = "route_review"
    return route_review


async def _handler(state: ReviewState) -> ReviewState:
    return {**state, "result": f"Routed to: {state['review_type']}"}


def _route(state: ReviewState) -> str:
    return state["review_type"]


def build_graph(holder: PromptHolder | None = None, model=None):
    if holder is None:
        holder = PromptHolder()
    builder = StateGraph(ReviewState)
    builder.add_node("route_review", make_routing_node(holder, model=model))
    for rt in ("security_review", "performance_review", "style_check", "architecture_review"):
        builder.add_node(rt, _handler)
        builder.add_edge(rt, END)
    builder.add_edge(START, "route_review")
    builder.add_conditional_edges(
        "route_review", _route,
        {rt: rt for rt in ("security_review", "performance_review", "style_check", "architecture_review")},
    )
    return builder.compile()


INPUT_KEY = "pr_description"
ROUTE_KEY = "review_type"
ONLY_NODES = {"route_review"}
VALID_ROUTES = ("security_review", "performance_review", "style_check", "architecture_review")


def make_initial_state(input_text: str) -> dict:
    return {"pr_description": input_text, "review_type": None, "result": None}


def get_result_route(result: dict) -> str:
    return result.get("review_type", "unknown") or "unknown"


TRAINSET = [
    # security_review
    {"input": "PR: Add JWT token validation middleware — validates tokens on all authenticated endpoints, rejects expired/tampered tokens.", "expected_route": "security_review", "entropy_category": "confident"},
    {"input": "PR: Migrate password hashing from MD5 to argon2id — affects user auth and password reset flows.", "expected_route": "security_review", "entropy_category": "confident"},
    {"input": "PR: Fix SQL injection vulnerability in user search — parameterize all raw query inputs.", "expected_route": "security_review", "entropy_category": "confident"},
    {"input": "PR: Implement OAuth 2.0 PKCE authorization flow — replaces legacy session-based auth.", "expected_route": "security_review", "entropy_category": "confident"},
    {"input": "PR: Add rate limiting to login endpoint — blocks brute force after 5 failed attempts.", "expected_route": "security_review", "entropy_category": "boundary"},
    # performance_review
    {"input": "PR: Fix N+1 query problem in user list endpoint — replace per-row queries with single JOIN.", "expected_route": "performance_review", "entropy_category": "confident"},
    {"input": "PR: Add Redis caching layer for product catalog — 24h TTL, cache-aside pattern.", "expected_route": "performance_review", "entropy_category": "confident"},
    {"input": "PR: Replace bubble sort with timsort in report generator — affects 500k-row processing.", "expected_route": "performance_review", "entropy_category": "confident"},
    {"input": "PR: Add composite index on (user_id, created_at) — resolves 3s query time on order history.", "expected_route": "performance_review", "entropy_category": "confident"},
    {"input": "PR: Lazy load images in product grid — reduces initial payload from 4MB to 800KB.", "expected_route": "performance_review", "entropy_category": "boundary"},
    # style_check
    {"input": "PR: Rename camelCase variables to snake_case throughout the payments module.", "expected_route": "style_check", "entropy_category": "confident"},
    {"input": "PR: Add missing docstrings to all public API methods in the user service.", "expected_route": "style_check", "entropy_category": "confident"},
    {"input": "PR: Fix inconsistent 2-space vs 4-space indentation across the codebase.", "expected_route": "style_check", "entropy_category": "confident"},
    {"input": "PR: Remove unused imports and commented-out dead code in legacy billing module.", "expected_route": "style_check", "entropy_category": "boundary"},
    {"input": "PR: Update inline comments to match new API behavior after v2 migration.", "expected_route": "style_check", "entropy_category": "confident"},
    # architecture_review
    {"input": "PR: Extract payment processing into standalone microservice — define service contract and API boundary.", "expected_route": "architecture_review", "entropy_category": "confident"},
    {"input": "PR: Add GraphQL schema for user profiles — new query types, mutations, and resolvers.", "expected_route": "architecture_review", "entropy_category": "confident"},
    {"input": "PR: Introduce event bus for inter-service communication — replaces synchronous HTTP calls.", "expected_route": "architecture_review", "entropy_category": "confident"},
    {"input": "PR: Redesign multi-tenant schema — evaluating row-level isolation vs separate schema approaches.", "expected_route": "architecture_review", "entropy_category": "confident"},
    {"input": "PR: Add CSRF token validation to auth middleware — touches session handling and security boundary.", "expected_route": "security_review", "entropy_category": "fragile"},
]

TRAINSET_LARGE = TRAINSET
