# Contrail Testing Environment

This folder contains real-world LangGraph agent examples used to validate Contrail integration.

## Philosophy

Rather than testing against toy fixtures, Contrail is validated against actual agents that engineers would deploy in production. This catches integration issues early and ensures the entropy/attribution outputs are meaningful.

## Agents

All agents are adapted from `langchain-ai/langgraph` official examples. Each has been trimmed to the routing-relevant core and adapted to be Contrail-wrappable without modifying the agent logic itself.

| Agent | Source | Routing type |
|---|---|---|
| `customer_support` | langchain-ai/langgraph `examples/customer-support/` | Multi-class: escalate / refund / info / general |
| `react_agent` | langchain-ai/langgraph `examples/react-agent/` | Tool-selection conditional edges |
| `multi_agent_supervisor` | langchain-ai/langgraph `examples/multi_agent_supervisor/` | Supervisor → specialist routing |
| `adaptive_rag` | langchain-ai/langgraph `examples/rag/` | Retrieval strategy selection |

## Setup

```bash
# From the repo root
cd testing
bash setup.sh
```

This will:
1. Install testing dependencies (`pip install -e ".[dev]"` from repo root)
2. Install agent-specific dependencies
3. Verify all agents are importable

## Running Tests

```bash
# Baseline tests (no Contrail, just verify agents work)
pytest testing/ -m baseline

# Integration tests (requires API keys)
pytest testing/ -m integration

# Full test matrix
pytest testing/ -v
```

## Required Environment Variables

```bash
ANTHROPIC_API_KEY=...         # required for all agents using Claude
OPENAI_API_KEY=...            # optional, for OpenAI-backed agents
LANGSMITH_API_KEY=...         # optional, only for LangSmith exporter tests
TAVILY_API_KEY=...            # optional, for react_agent web search tool
```

## Harness

`testing/harness/` contains the shared test infrastructure:

- `runner.py` — `BaseTestRunner`: runs an agent and returns its full execution trace
- `assertions.py` — reusable assertion helpers for TraceRecord validation
- `fixtures.py` — standard input sets (confident / boundary / fragile) shared across agents
