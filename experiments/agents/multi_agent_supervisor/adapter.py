"""
Multi-Agent Supervisor adapter for the CPE-GEPA optimization loop.
"""
from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from testing.harness.fixtures import SUPERVISOR_INPUTS

DEFAULT_SYSTEM_PROMPT = """\
You are a supervisor that routes tasks to specialists.
Choose exactly one agent. Respond with only the agent name, nothing else.

Agents:
- code_agent: writing Python/code, debugging, technical implementation
- writing_agent: prose writing, summarisation, editing, documentation
- research_agent: web research, fact-finding, current events, data lookup"""


class PromptHolder:
    def __init__(self, prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
        self.system_prompt = prompt


class SupervisorState(TypedDict):
    task: str
    assigned_agent: str | None
    result: str | None


def _get_llm(model=None):
    from conntrail.utils.providers import get_chat_model
    return get_chat_model(model or "claude-haiku-4-5-20251001", max_tokens=20)


def make_routing_node(holder: PromptHolder, model=None):
    async def supervisor_node(state: SupervisorState) -> SupervisorState:
        llm = _get_llm(model)
        response = await llm.ainvoke([
            SystemMessage(content=holder.system_prompt),
            HumanMessage(content=state["task"]),
        ])
        agent = response.content.strip().lower()
        if agent not in ("code_agent", "writing_agent", "research_agent"):
            agent = "writing_agent"
        return {**state, "assigned_agent": agent}
    supervisor_node.__name__ = "supervisor_node"
    return supervisor_node


async def _specialist(state: SupervisorState) -> SupervisorState:
    return {**state, "result": f"Handled by: {state['assigned_agent']}"}


def _route(state: SupervisorState) -> str:
    return state["assigned_agent"]


def build_graph(holder: PromptHolder | None = None, model=None):
    if holder is None:
        holder = PromptHolder()
    builder = StateGraph(SupervisorState)
    builder.add_node("supervisor_node", make_routing_node(holder, model=model))
    for agent in ("code_agent", "writing_agent", "research_agent"):
        builder.add_node(agent, _specialist)
        builder.add_edge(agent, END)
    builder.add_edge(START, "supervisor_node")
    builder.add_conditional_edges(
        "supervisor_node", _route,
        {a: a for a in ("code_agent", "writing_agent", "research_agent")},
    )
    return builder.compile()


INPUT_KEY = "task"
ROUTE_KEY = "assigned_agent"
ONLY_NODES = {"supervisor_node"}
VALID_ROUTES = ("code_agent", "writing_agent", "research_agent")


def make_initial_state(input_text: str) -> dict:
    return {"task": input_text, "assigned_agent": None, "result": None}


def get_result_route(result: dict) -> str:
    return result.get("assigned_agent", "unknown") or "unknown"


TRAINSET = [
    {"input": SUPERVISOR_INPUTS[0].text, "expected_route": "code_agent",     "entropy_category": "confident"},
    {"input": SUPERVISOR_INPUTS[1].text, "expected_route": "writing_agent",  "entropy_category": "confident"},
    {"input": SUPERVISOR_INPUTS[2].text, "expected_route": "code_agent",     "entropy_category": "boundary"},
    {"input": SUPERVISOR_INPUTS[3].text, "expected_route": "research_agent", "entropy_category": "fragile"},
]

TRAINSET_LARGE = TRAINSET + [
    # confident code_agent
    {"input": "Write a SQL query to find all users who haven't logged in for the past 90 days.", "expected_route": "code_agent",     "entropy_category": "confident"},
    {"input": "Implement a binary search algorithm in Python with unit tests.",                  "expected_route": "code_agent",     "entropy_category": "confident"},
    {"input": "Fix the memory leak in this C++ function that handles file I/O.",                 "expected_route": "code_agent",     "entropy_category": "confident"},
    {"input": "Review my Python script for performance issues before I deploy it to production.", "expected_route": "code_agent",     "entropy_category": "confident"},
    {"input": "Help me design a REST API for user authentication and session management.",        "expected_route": "code_agent",     "entropy_category": "confident"},
    # confident writing_agent
    {"input": "Write an engaging blog post introduction about the future of remote work.",        "expected_route": "writing_agent",  "entropy_category": "confident"},
    {"input": "Proofread and edit this email for clarity before I send it to the board.",         "expected_route": "writing_agent",  "entropy_category": "confident"},
    {"input": "Summarize this 10-page technical report into a concise executive summary.",        "expected_route": "writing_agent",  "entropy_category": "confident"},
    {"input": "Draft a professional apology letter to a client about a service outage.",          "expected_route": "writing_agent",  "entropy_category": "confident"},
    {"input": "Write a product description for our new SaaS analytics platform.",                 "expected_route": "writing_agent",  "entropy_category": "confident"},
    # confident research_agent
    {"input": "Find the latest statistics on global electric vehicle adoption rates.",            "expected_route": "research_agent", "entropy_category": "confident"},
    {"input": "What companies are currently leading in quantum computing development?",           "expected_route": "research_agent", "entropy_category": "confident"},
    {"input": "Compile a comparison of the top 5 CRM tools with their pricing and features.",    "expected_route": "research_agent", "entropy_category": "confident"},
    # boundary
    {"input": "Explain how the PageRank algorithm works with a concrete example.",               "expected_route": "code_agent",     "entropy_category": "boundary"},
    {"input": "Write a summary of recent developments in large language models.",                "expected_route": "writing_agent",  "entropy_category": "boundary"},
    # fragile
    {"input": "Create some content about Python best practices for a technical audience.",       "expected_route": "writing_agent",  "entropy_category": "fragile"},
]
