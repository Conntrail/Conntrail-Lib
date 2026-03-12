# Multi-Agent Supervisor

Adapted from: `langchain-ai/langgraph` examples/multi_agent_supervisor/

## Description

A supervisor agent that routes tasks to specialist sub-agents:
- `code_agent` — Python code generation and debugging
- `writing_agent` — prose writing, summarisation, documentation
- `research_agent` — web research and fact-finding

The supervisor uses LLM-based routing to select the right specialist.

## Why it's a good Contrail test subject

Hierarchical routing is a pattern with genuinely ambiguous boundaries (e.g. "document this code" — code_agent or writing_agent?). Good for testing multi-dimensional attribution (task_type, domain, formality).

## Source

```
git clone https://github.com/langchain-ai/langgraph
# See: examples/multi_agent_supervisor/
```
