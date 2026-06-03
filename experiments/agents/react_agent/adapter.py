"""
ReAct Agent adapter for the CPE-GEPA optimization loop.

The react_agent's routing is a conditional edge (should_use_tool), not a node,
so we introduce a thin routing node that exposes the decision as a state key.
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from testing.harness.fixtures import REACT_AGENT_INPUTS

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful assistant. Use tools when you need calculations or current
information. Otherwise answer directly from your knowledge."""


class PromptHolder:
    def __init__(self, prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
        self.system_prompt = prompt


class ReactState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    last_route: str | None   # "call_tool" or "respond" — exposed for Conntrail


def _get_llm():
    from langchain_ollama import ChatOllama
    return ChatOllama(model="qwen2.5:7b", num_predict=100, temperature=0.0)


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression."""
    try:
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            return "Error: only arithmetic expressions allowed"
        return str(eval(expression, {"__builtins__": {}}))  # noqa: S307
    except Exception as e:
        return f"Error: {e}"


_TOOLS = [calculator]


def make_routing_node(holder: PromptHolder):
    async def agent_node(state: ReactState) -> ReactState:
        llm = _get_llm().bind_tools(_TOOLS)
        response = await llm.ainvoke([
            SystemMessage(content=holder.system_prompt),
            *state["messages"],
        ])
        route = "call_tool" if getattr(response, "tool_calls", None) else "respond"
        return {"messages": [response], "last_route": route}
    agent_node.__name__ = "agent_node"
    return agent_node


async def tool_node(state: ReactState) -> ReactState:
    tools_map = {t.name: t for t in _TOOLS}
    last = state["messages"][-1]
    from langchain_core.messages import ToolMessage
    msgs = []
    for tc in getattr(last, "tool_calls", []):
        result = tools_map[tc["name"]].invoke(tc["args"]) if tc["name"] in tools_map else "Tool not found"
        msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": msgs, "last_route": state.get("last_route")}


async def respond_node(state: ReactState) -> ReactState:
    return state


def _should_use_tool(state: ReactState) -> str:
    return state.get("last_route") or "respond"


def build_graph(holder: PromptHolder | None = None):
    if holder is None:
        holder = PromptHolder()
    builder = StateGraph(ReactState)
    builder.add_node("agent_node", make_routing_node(holder))
    builder.add_node("call_tool", tool_node)
    builder.add_node("respond", respond_node)
    builder.add_edge(START, "agent_node")
    builder.add_conditional_edges(
        "agent_node", _should_use_tool,
        {"call_tool": "call_tool", "respond": "respond"},
    )
    builder.add_edge("call_tool", "agent_node")
    builder.add_edge("respond", END)
    return builder.compile()


INPUT_KEY = "messages"
ROUTE_KEY = "last_route"
ONLY_NODES = {"agent_node"}
VALID_ROUTES = ("call_tool", "respond")


def make_initial_state(input_text: str) -> dict:
    return {"messages": [HumanMessage(content=input_text)], "last_route": None}


def get_result_route(result: dict) -> str:
    return result.get("last_route", "unknown") or "unknown"


# Trainset uses only the calculator tool (no Tavily key required).
# Avoids the brave_search hallucination seen with smaller models.
TRAINSET = [
    {"input": "What is 1234 multiplied by 5678?",   "expected_route": "call_tool", "entropy_category": "confident"},
    {"input": "What is the capital of France?",      "expected_route": "respond",   "entropy_category": "confident"},
    {"input": "What is 15 percent of 240?",          "expected_route": "call_tool", "entropy_category": "boundary"},
    {"input": "What is 2 to the power of 10?",       "expected_route": "call_tool", "entropy_category": "fragile"},
]
