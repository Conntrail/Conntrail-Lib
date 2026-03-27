"""
Multi-Agent Supervisor — adapted from langchain-ai/langgraph examples.

Supervisor routes to: code_agent | writing_agent | research_agent
Provider-agnostic: works with Groq, Anthropic, or OpenAI.

Source: https://github.com/langchain-ai/langgraph/tree/main/examples/multi_agent_supervisor
"""
from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph


class SupervisorState(TypedDict):
    task: str
    assigned_agent: str | None
    result: str | None


def _get_llm():
    from testing.harness.llm import get_llm
    return get_llm()


# --- Supervisor node ---
async def supervisor_node(state: SupervisorState) -> SupervisorState:
    llm = _get_llm()
    response = await llm.ainvoke([
        SystemMessage(content="""You are a supervisor that routes tasks to specialists.
Choose exactly one agent. Respond with only the agent name, nothing else.

Agents:
- code_agent: writing Python/code, debugging, technical implementation
- writing_agent: prose writing, summarisation, editing, documentation
- research_agent: web research, fact-finding, current events, data lookup"""),
        HumanMessage(content=state["task"]),
    ])
    agent = response.content.strip().lower()
    if agent not in ("code_agent", "writing_agent", "research_agent"):
        agent = "writing_agent"
    return {**state, "assigned_agent": agent}


def route_to_agent(state: SupervisorState) -> Literal["code_agent", "writing_agent", "research_agent"]:
    return state["assigned_agent"]


# --- Specialist nodes ---
async def code_agent(state: SupervisorState) -> SupervisorState:
    response = await _get_llm().ainvoke([
        SystemMessage(content="You are an expert Python developer. Write clean, well-documented code."),
        HumanMessage(content=state["task"]),
    ])
    return {**state, "result": response.content}


async def writing_agent(state: SupervisorState) -> SupervisorState:
    response = await _get_llm().ainvoke([
        SystemMessage(content="You are an expert writer. Write clear, engaging prose."),
        HumanMessage(content=state["task"]),
    ])
    return {**state, "result": response.content}


async def research_agent(state: SupervisorState) -> SupervisorState:
    response = await _get_llm().ainvoke([
        SystemMessage(content="You are a research specialist. Provide accurate, well-sourced information."),
        HumanMessage(content=state["task"]),
    ])
    return {**state, "result": response.content}


# --- Graph builder ---
def build_graph():
    builder = StateGraph(SupervisorState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("code_agent", code_agent)
    builder.add_node("writing_agent", writing_agent)
    builder.add_node("research_agent", research_agent)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "code_agent": "code_agent",
            "writing_agent": "writing_agent",
            "research_agent": "research_agent",
        },
    )
    builder.add_edge("code_agent", END)
    builder.add_edge("writing_agent", END)
    builder.add_edge("research_agent", END)

    return builder.compile()


if __name__ == "__main__":
    import asyncio

    async def main():
        graph = build_graph()
        result = await graph.ainvoke({
            "task": "Write a Python function to reverse a linked list.",
            "assigned_agent": None,
            "result": None,
        })
        print(f"Assigned to: {result['assigned_agent']}")
        print(f"Result: {result['result'][:200]}...")

    asyncio.run(main())
