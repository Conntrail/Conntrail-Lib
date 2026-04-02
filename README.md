# Conntrail

[![CI](https://github.com/Conntrail/conntrail/actions/workflows/ci.yml/badge.svg)](https://github.com/Conntrail/conntrail/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Decision path tracer for LangGraph agents. Measures routing stability via contrastive analysis — wraps routing nodes, generates semantic variants of each input, reruns them, and reports how consistently the node makes the same decision.

## Why Conntrail?

LangGraph agents route user inputs through conditional nodes. But **routing that seems correct in testing often fails in production** when user phrasing varies.

**Real impacts:**
- **Customer support** — "I need help with my order" vs "Where is my stuff?" route to different handlers
- **Content moderation** — "This is spicy" (food) vs "This is spicy" (NSFW) cause inconsistent actions
- **Safety-critical** — "stop" vs "please stop now" bypass safeguards
- **Debugging nightmare** — "the router did something weird" with no attribution

**Conntrail helps you:**
- **Detect decision boundaries** — know which inputs sit on the edge between routes
- **Measure robustness** — quantify how stable each routing decision is
- **Debug with attribution** — understand *why* a node made a particular choice
- **Validate automatically** — catch fragile routing before it reaches production

## How it works

For each traced node call, Conntrail:

1. Captures the input and runs the node normally (hot path unchanged)
2. Generates 3 semantic variants: **similar** (paraphrase), **neutral** (flattened), **opposite** (inverted)
3. Runs all 4 variants through the node
4. Computes **Shannon entropy** over the 4 routing outcomes — `0.0` = always the same route, `1.0` = every variant went somewhere different
5. Attributes divergence to a semantic dimension (e.g. "urgency/sentiment", "semantic intensity")
6. Exports a `TraceRecord` with a plain-language summary

```
[CONNTRAIL] classify_query | BOUNDARY | entropy=0.41 | attr: semantic intensity
  → Route:   refund
  → Summary: The 'classify_query' node routed to 'refund' with boundary confidence (entropy: 0.41).
             Removing the urgency dimension would likely flip the decision to 'order_info'.
```

Stability labels:

| Label | Entropy | Meaning |
|---|---|---|
| `confident` | 0.0 – 0.25 | All variants agree — routing is robust |
| `boundary` | 0.25 – 0.60 | Some variants flip — input sits near a decision boundary |
| `fragile` | 0.60 – 1.0 | Multiple variants disagree — small wording changes change the route |

## Installation

```bash
pip install conntrail
```

Requires Python 3.11+, LangGraph ≥ 0.2, LangChain Core ≥ 0.3.

Conntrail uses a small LLM for contrast generation (`claude-haiku-4-5-20251001` by default). Set whichever provider key you want to use:

```bash
export ANTHROPIC_API_KEY=...   # default
export GROQ_API_KEY=...
export OPENAI_API_KEY=...
```

## Usage

### Wrap an entire graph

```python
from conntrail import trace_graph, ConntrailConfig

graph = trace_graph(
    compiled_graph,
    config=ConntrailConfig(
        sample_rate=0.2,          # trace 20% of calls
        export_format="jsonl",    # write to ./conntrail_traces/
        async_mode=True,          # never block the hot path
    ),
)

# Use the graph exactly as before
result = await graph.ainvoke({"message": "I want a refund"})
```

### Trace a specific node only

```python
graph = trace_graph(
    compiled_graph,
    config=ConntrailConfig(sample_rate=1.0, export_format="stdout"),
    only_nodes={"classify_query"},   # skip handler/leaf nodes
)
```

### Decorate a single node

```python
from conntrail import trace_node, ConntrailConfig

@trace_node(config=ConntrailConfig(export_format="stdout"))
async def classify_query(state):
    ...
```

### Access traces in-process

Set `async_mode=False` to collect `TraceRecord` objects directly in the result:

```python
graph = trace_graph(compiled_graph, config=ConntrailConfig(async_mode=False))

result = await graph.ainvoke({"message": "..."})
for trace in result.get("__conntrail_traces__", []):
    print(trace.node_id, trace.entropy_score, trace.stability)
```

### Alert on fragile nodes

```python
def alert(trace):
    print(f"FRAGILE: {trace.node_id} entropy={trace.entropy_score:.2f}")

graph = trace_graph(
    compiled_graph,
    config=ConntrailConfig(
        entropy_alert_threshold=0.6,
        on_alert=alert,
    ),
)
```

## Configuration

```python
ConntrailConfig(
    contrast_model="claude-haiku-4-5-20251001",  # LLM for generating contrasts
    sample_rate=1.0,             # fraction of calls to trace [0.0, 1.0]
    async_mode=True,             # True = fire-and-forget; False = await and inject into result
    export_format="jsonl",       # "jsonl" | "stdout" | "langsmith"
    export_path="./conntrail_traces",
    entropy_alert_threshold=0.6, # trigger on_alert when entropy >= this
    on_alert=None,               # callback(trace_record: TraceRecord) -> None
)
```

`sample_rate=0.1–0.2` is recommended for production. `sample_rate=1.0` for development.

## TraceRecord fields

Each trace contains:

| Field | Type | Description |
|---|---|---|
| `trace_id` | `str` | UUID |
| `node_id` | `str` | Node name in the graph |
| `timestamp` | `datetime` | UTC time of the original call |
| `original_input` | `str` | Input text that was traced |
| `original_route` | `str` | Route taken by the actual call |
| `entropy_score` | `float` | Routing instability [0.0, 1.0] |
| `stability` | `str` | `"confident"` / `"boundary"` / `"fragile"` |
| `attribution_dimension` | `str` | Semantic feature driving the divergence |
| `counterfactual_route` | `str \| None` | Where the most-divergent contrast went |
| `plain_language_summary` | `str` | Human-readable explanation |

## Export formats

**JSONL** — one JSON object per line, written to `{export_path}/traces_{date}.jsonl`

**Stdout** — coloured terminal output, useful during development

**LangSmith** — planned (Phase 7)

## Roadmap

- [ ] **Phase 3**: Embedding-based similarity for non-branching nodes
- [ ] **Phase 6**: LLM-based open-ended attribution labels
- [ ] **Phase 7**: LangSmith integration for centralized trace storage

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, testing, and contribution guidelines.

```bash
git clone https://github.com/your-org/conntrail
cd conntrail
pip install -e ".[dev]"
pytest tests/          # unit tests (no API keys needed)
```

Integration tests against real LangGraph agents require a provider key:

```bash
cp testing/.env.example testing/.env
# fill in GROQ_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY
pytest testing/test_conntrail_integration.py -m "integration and not slow"
```

## License

MIT
