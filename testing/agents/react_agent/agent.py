"""
ReAct Agent — adapted from langchain-ai/langgraph examples.

Tool-selection conditional edges: agent → (call_tool | respond)
Provider-agnostic: works with Groq, Anthropic, or OpenAI.

Source: https://github.com/langchain-ai/langgraph/tree/main/examples/react-agent
"""
from __future__ import annotations

import os
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# --- Tool definitions ---
@tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression. Input: a string like '2 + 2' or '10 * 5'."""
    try:
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            return "Error: only arithmetic expressions allowed"
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def _get_tools():
    tools = [calculator]
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            tools.append(TavilySearchResults(max_results=3, api_key=tavily_key))
        except ImportError:
            pass
    return tools


def _get_llm_with_tools():
    from testing.harness.llm import get_llm
    tools = _get_tools()
    return get_llm().bind_tools(tools), tools


# --- Nodes ---
async def agent_node(state: AgentState) -> AgentState:
    llm, _ = _get_llm_with_tools()
    response = await llm.ainvoke([
        SystemMessage(content="You are a helpful assistant. Use tools when you need calculations or current information. Otherwise answer directly."),
        *state["messages"],
    ])
    return {"messages": [response]}


async def tool_node(state: AgentState) -> AgentState:
    _, tools = _get_llm_with_tools()
    tools_map = {t.name: t for t in tools}
    last_message = state["messages"][-1]
    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        if tool_name in tools_map:
            result = await tools_map[tool_name].ainvoke(tool_call["args"])
            tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
        else:
            tool_messages.append(ToolMessage(content=f"Tool {tool_name!r} not found", tool_call_id=tool_call["id"]))
    return {"messages": tool_messages}


def should_use_tool(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "call_tool"
    return "respond"


async def respond_node(state: AgentState) -> AgentState:
    return state


# --- Graph builder ---
def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    builder.add_node("call_tool", tool_node)
    builder.add_node("respond", respond_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        should_use_tool,
        {"call_tool": "call_tool", "respond": "respond"},
    )
    builder.add_edge("call_tool", "agent")
    builder.add_edge("respond", END)

    return builder.compile()


if __name__ == "__main__":
    import asyncio

    async def main():
        graph = build_graph()
        result = await graph.ainvoke({
            "messages": [HumanMessage(content="What is 42 * 17?")]
        })
        print(result["messages"][-1].content)

    asyncio.run(main())
