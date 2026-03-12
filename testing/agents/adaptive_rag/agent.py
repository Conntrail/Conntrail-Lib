"""
Adaptive RAG Agent — adapted from langchain-ai/langgraph examples.

Routes to: direct_answer | vector_search | web_search
This is the Contrail test harness version with an in-memory vector store.

Source: https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_adaptive_rag.ipynb
"""
import os
from typing import Literal, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph


class RAGState(TypedDict):
    query: str
    strategy: str | None
    retrieved_docs: list[str]
    answer: str | None


# --- In-memory document fixture ---
FIXTURE_DOCS = [
    Document(page_content="Contrail Q4 2024 sales increased 23% year-over-year.", metadata={"source": "q4_report"}),
    Document(page_content="Customer churn decreased from 8% to 5% after the loyalty programme launch.", metadata={"source": "churn_analysis"}),
    Document(page_content="The engineering team shipped 12 features in Q3 2024.", metadata={"source": "eng_report"}),
    Document(page_content="EMEA expansion plans are on track for H1 2025.", metadata={"source": "strategy_doc"}),
    Document(page_content="Support ticket volume peaked in November 2024 at 2,400 tickets/week.", metadata={"source": "support_metrics"}),
]


def _simple_search(query: str, docs: list[Document], top_k: int = 2) -> list[str]:
    """Naive keyword overlap search (no embeddings needed for test harness)."""
    query_words = set(query.lower().split())
    scored = []
    for doc in docs:
        doc_words = set(doc.page_content.lower().split())
        score = len(query_words & doc_words)
        scored.append((score, doc.page_content))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [content for _, content in scored[:top_k]]


def _get_llm():
    return ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=512,
    )


# --- Router node ---
async def route_query(state: RAGState) -> RAGState:
    """Determine retrieval strategy for the query."""
    llm = _get_llm()
    response = await llm.ainvoke([
        SystemMessage(content="""Select a retrieval strategy for this query. Respond with only the strategy name.

Strategies:
- direct_answer: general knowledge, definitions, facts from training data
- vector_search: questions about internal company documents, reports, or data
- web_search: current events, recent news, real-time information"""),
        HumanMessage(content=state["query"]),
    ])
    strategy = response.content.strip().lower()
    if strategy not in ("direct_answer", "vector_search", "web_search"):
        strategy = "direct_answer"
    return {**state, "strategy": strategy}


def select_strategy(state: RAGState) -> Literal["direct_answer", "vector_search", "web_search"]:
    return state["strategy"]


# --- Handler nodes ---
async def direct_answer(state: RAGState) -> RAGState:
    """Answer directly from LLM knowledge."""
    llm = _get_llm()
    response = await llm.ainvoke([
        SystemMessage(content="Answer the question directly from your knowledge."),
        HumanMessage(content=state["query"]),
    ])
    return {**state, "retrieved_docs": [], "answer": response.content}


async def vector_search(state: RAGState) -> RAGState:
    """Search internal document fixtures."""
    docs = _simple_search(state["query"], FIXTURE_DOCS)
    llm = _get_llm()
    context = "\n".join(docs) if docs else "No relevant documents found."
    response = await llm.ainvoke([
        SystemMessage(content=f"Answer based on these internal documents:\n\n{context}"),
        HumanMessage(content=state["query"]),
    ])
    return {**state, "retrieved_docs": docs, "answer": response.content}


async def web_search(state: RAGState) -> RAGState:
    """Simulate web search (returns placeholder in test harness)."""
    # In a real agent this would call Tavily or similar
    simulated_result = f"[Web search placeholder for: {state['query']}]"
    llm = _get_llm()
    response = await llm.ainvoke([
        SystemMessage(content="You searched the web. Provide a helpful response based on typical web results."),
        HumanMessage(content=state["query"]),
    ])
    return {**state, "retrieved_docs": [simulated_result], "answer": response.content}


# --- Graph builder ---
def build_graph():
    """Build and compile the adaptive RAG graph."""
    builder = StateGraph(RAGState)

    builder.add_node("route_query", route_query)
    builder.add_node("direct_answer", direct_answer)
    builder.add_node("vector_search", vector_search)
    builder.add_node("web_search", web_search)

    builder.add_edge(START, "route_query")
    builder.add_conditional_edges(
        "route_query",
        select_strategy,
        {
            "direct_answer": "direct_answer",
            "vector_search": "vector_search",
            "web_search": "web_search",
        },
    )
    builder.add_edge("direct_answer", END)
    builder.add_edge("vector_search", END)
    builder.add_edge("web_search", END)

    return builder.compile()


if __name__ == "__main__":
    import asyncio

    async def main():
        graph = build_graph()
        result = await graph.ainvoke({
            "query": "What was Q4 2024 sales performance?",
            "strategy": None,
            "retrieved_docs": [],
            "answer": None,
        })
        print(f"Strategy: {result['strategy']}")
        print(f"Answer: {result['answer'][:200]}...")

    asyncio.run(main())
