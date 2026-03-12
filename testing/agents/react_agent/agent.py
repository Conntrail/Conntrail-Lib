"""
ReAct Agent — adapted from langchain-ai/langgraph examples.

Uses tool-selection conditional edges: agent → (call_tool | respond)
This is the Contrail test harness version.

Source: https://github.com/langchain-ai/langgraph/tree/main/examples/react-agent
"""
import os
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
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
        # Safe eval for arithmetic only
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            return "Error: only arithmetic expressions allowed"
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def _get_tools():
    tools = [calculator]

    # Only add web search if Tavily key is available
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            web_search = TavilySearchResults(max_results=3, api_key=tavily_key)
            tools.append(web_search)
        except ImportError:
            pass  # langchain_community not installed

    return tools


def _get_llm():
    tools = _get_tools()
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=512,
    )
    return llm.bind_tools(tools), tools


# --- Nodes ---
async def agent_node(state: AgentState) -> AgentState:
    """Main agent reasoning node — decides whether to call a tool or respond."""
    llm, _ = _get_llm()
    response = await llm.ainvoke([
        SystemMessage(content="You are a helpful assistant. Use tools when you need current information or calculations. Otherwise answer directly."),
        *state["messages"],
    ])
    return {"messages": [response]}


async def tool_node(state: AgentState) -> AgentState:
    """Execute the tool calls requested by the agent."""
    tools_map = {t.name: t for t in _get_tools()}
    last_message = state["messages"][-1]
    tool_messages = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        if tool_name in tools_map:
            result = await tools_map[tool_name].ainvoke(tool_call["args"])
            tool_messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )
        else:
            tool_messages.append(
                ToolMessage(content=f"Tool {tool_name!r} not found", tool_call_id=tool_call["id"])
            )

    return {"messages": tool_messages}


def should_use_tool(state: AgentState) -> str:
    """Conditional edge: use tool if the agent made tool calls, otherwise end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "call_tool"
    return "respond"


async def respond_node(state: AgentState) -> AgentState:
    """Final response — passes through the last message unchanged."""
    return state


# --- Graph builder ---
def build_graph():
    """Build and compile the ReAct agent graph."""
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
    builder.add_edge("call_tool", "agent")  # loop back after tool use
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
