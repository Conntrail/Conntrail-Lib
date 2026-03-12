# ReAct Agent

Adapted from: `langchain-ai/langgraph` examples/react-agent/

## Description

A ReAct (Reasoning + Acting) agent with two tools:
- `web_search` — Tavily web search for current information
- `calculator` — simple arithmetic evaluation

The conditional edge routes between `call_tool` and `respond` based on whether the LLM decides to use a tool or answer directly.

## Why it's a good Contrail test subject

Tool-selection routing is one of the most common LangGraph patterns. The boundary between "use web search" and "answer from knowledge" is a genuine routing ambiguity. Attribution dimension is typically "recency" or "specificity".

## Source

```
git clone https://github.com/langchain-ai/langgraph
# See: examples/react-agent/
```

## Environment Variables

- `ANTHROPIC_API_KEY` — required
- `TAVILY_API_KEY` — optional (web_search tool will be disabled if not set)
