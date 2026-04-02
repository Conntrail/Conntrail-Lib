# Conntrail User Guide

A comprehensive guide to understanding and using Conntrail for tracing LangGraph agent routing decisions.

---

## Table of Contents

- [Quick Start (5 Minutes)](#quick-start-5-minutes) - 🔰 Beginner
- [Understanding Conntrail Output](#understanding-conntrail-output) - 🔰 Beginner
- [Core Concepts](#core-concepts) - 🔰 Beginner
- [Configuration Guide](#configuration-guide) - 🚀 Intermediate
- [Integration Patterns](#integration-patterns) - 🚀 Intermediate
- [Production Deployment](#production-deployment) - 🚀 Intermediate
- [Interpreting Results](#interpreting-results) - 🚀 Intermediate
- [Troubleshooting](#troubleshooting) - 🚀 Intermediate
- [API Reference](#api-reference) - 🚀 Intermediate

---

## Quick Start (5 Minutes)

### Installation

```bash
pip install conntrail
```

### Your First Trace

```python
from conntrail import trace_graph, ConntrailConfig
from your_agent import build_graph  # Your LangGraph agent

# Build your graph
graph = build_graph()

# Wrap it with Conntrail
traced_graph = trace_graph(
    graph,
    config=ConntrailConfig(
        export_format="stdout",  # See traces in terminal
        sample_rate=1.0,         # Trace every call (use 0.1 in production)
    )
)

# Use it exactly like before
result = await traced_graph.ainvoke({"message": "Hello!"})
```

**That's it!** You'll see Conntrail output like:

```
[CONNTRAIL] your_node | CONFIDENT | entropy=0.00 | attr: none detected
  → Route:   some_route
  → Summary: The 'your_node' node routed to 'some_route'...
```

---

## Understanding Conntrail Output

### Reading the Console Output

Conntrail produces colored output for each traced decision:

```
[CONNTRAIL] classify_query | CONFIDENT | entropy=0.00 | attr: none detected
  → Route:   refund
  → Summary: The 'classify_query' node routed to 'refund' with confident 
             confidence (entropy: 0.00). The decision appears to have been 
             driven by none detected in the input.
```

**Breaking it down:**

| Part | Meaning |
|------|---------|
| `classify_query` | The name of the node being traced |
| `CONFIDENT` | Stability label (CONFIDENT/BOUNDARY/FRAGILE) |
| `entropy=0.00` | Numerical stability score (0.0 = stable, 1.0 = unstable) |
| `attr: none detected` | What dimension drives decision changes |
| `Route: refund` | The routing decision made |
| `Summary` | Human-readable explanation |

### Understanding Entropy Scores

```
0.00 ═══════════════════════════════════════════════════ 1.00
  │                                                    │
  🟢 CONFIDENT      🟡 BOUNDARY          🔴 FRAGILE   │
  0.0 - 0.25       0.25 - 0.60           0.60 - 1.0   │
  │                                                    │
  All variants     Some variants flip    Most variants│
  agree            (near boundary)       disagree     │
```

**Quick Reference:**

| Score | Label | What It Means |
|-------|-------|---------------|
| 0.00 | Confident | Perfect stability - all 4 input variants produced the same route |
| 0.25 | Confident | Minor variation - 3 of 4 variants agreed |
| 0.50 | Boundary | Moderate uncertainty - 2 of 4 variants differed |
| 0.75 | Fragile | High uncertainty - most variants produced different routes |
| 1.00 | Fragile | Maximum instability - every variant produced a different route |

### Understanding Attribution

Conntrail tells you **why** decisions change:

| Attribution | When It Appears | What It Means |
|-------------|-----------------|---------------|
| `none detected` | All variants agree | Your routing is stable |
| `semantic intensity` | Opposite variant flips | Strong vs weak phrasing changes the route |
| `urgency/sentiment` | Neutral variant flips | Emotional tone or urgency changes the route |
| `surface form` | Similar variant flips | Even slight rephrasing changes the route (very fragile!) |

**Example:**

```
Input: "I want a refund NOW!"
→ Route: urgent_refund

Opposite variant: "I'm satisfied with my purchase"
→ Route: feedback

Attribution: semantic intensity
→ Insight: The intensity/emphasis drives the urgent vs calm routing
```

---

## Core Concepts

### What is Routing Stability?

Routing stability measures how consistently your LangGraph agent makes the same routing decision when the input is slightly varied.

**Why it matters:**
- Users phrase the same request in many ways
- "I want a refund" vs "Can I get my money back?" vs "This product is broken"
- A stable router sends all three to the same handler
- A fragile router might send them to different handlers

### How Conntrail Works

For each traced node call:

1. **Capture** the original input
2. **Generate** 3 semantic variants:
   - **Similar**: Paraphrase (same meaning, different words)
   - **Neutral**: Flattened (urgency/sentiment removed)
   - **Opposite**: Inverted (opposite meaning)
3. **Run** all 4 variants through the node
4. **Compare** routing decisions
5. **Calculate** Shannon entropy over outcomes
6. **Attribute** which variant type caused divergence
7. **Report** the results

### Contrast Variants Explained

Given input: *"I need this fixed urgently!"*

| Variant | Example | Purpose |
|---------|---------|---------|
| **Original** | "I need this fixed urgently!" | Baseline |
| **Similar** | "Can you please fix this as soon as possible?" | Test paraphrase robustness |
| **Neutral** | "I need this fixed." | Test urgency/sentiment impact |
| **Opposite** | "This isn't urgent, take your time." | Test semantic inversion |

**The Logic:**
- If **Similar** flips the route → Your router cares about exact wording (fragile!)
- If **Neutral** flips the route → Your router cares about urgency/tone
- If **Opposite** flips the route → Your router correctly understands meaning

---

## Configuration Guide

### ConntrailConfig Options

```python
from conntrail import ConntrailConfig

config = ConntrailConfig(
    # LLM for generating contrast variants
    # Use a cheap/fast model - never your production model
    contrast_model="groq-llama3-8b-8192",
    
    # Fraction of calls to trace (0.0 - 1.0)
    # 1.0 = trace every call (development)
    # 0.1 = trace 10% of calls (production)
    sample_rate=0.2,
    
    # Export destination
    # "stdout" = terminal output (development)
    # "jsonl" = files for audit trails (production)
    # "langsmith" = LangSmith integration (coming soon)
    export_format="jsonl",
    
    # Directory for jsonl files (when export_format="jsonl")
    export_path="./conntrail_traces",
    
    # Entropy threshold for alerts
    # Calls with entropy >= this trigger on_alert
    entropy_alert_threshold=0.6,
    
    # Callback for fragile decisions
    # Signature: (trace_record: TraceRecord) -> None
    on_alert=your_alert_function,
    
    # Timeout for node analysis (seconds)
    # Prevents hanging on slow nodes
    analysis_timeout=30.0,
)
```

### Choosing Sample Rates

| Environment | Recommended Rate | Reason |
|-------------|------------------|--------|
| **Development** | 1.0 (100%) | Catch issues early, cost doesn't matter |
| **Staging** | 0.5 (50%) | Good coverage without excessive costs |
| **Production (low volume)** | 0.2 (20%) | Monitor without breaking the bank |
| **Production (high volume)** | 0.05 (5%) | Statistical sampling is sufficient |
| **Safety-critical** | 1.0 (100%) | Never miss a fragile decision |

**Cost Calculation:**

```
1000 requests/day × sample_rate 0.1 = 100 traced
100 traced × 4 variants = 400 LLM calls
400 calls × $0.0001/call = $0.04/day
```

### Choosing Alert Thresholds

| Application Type | Recommended Threshold | Reason |
|------------------|----------------------|--------|
| **Content moderation** | 0.5 | Safety-critical, catch uncertainty early |
| **Customer support** | 0.6 | Balance sensitivity with noise |
| **Internal tools** | 0.7 | Only catch clearly fragile decisions |
| **Research/debugging** | 0.3 | Very sensitive, catch all boundary cases |

### Export Format Comparison

| Format | Best For | Pros | Cons |
|--------|----------|------|------|
| **stdout** | Development | Instant visibility, colored output | Not persistent, hard to analyze |
| **jsonl** | Production | Persistent, queryable, audit trails | Requires file management |
| **langsmith** | Enterprise | Centralized, dashboards | Phase 7 (coming soon) |

---

## Integration Patterns

### Pattern 1: Wrap Entire Graph

Trace all nodes in your graph:

```python
from conntrail import trace_graph, ConntrailConfig

graph = build_graph()  # Your LangGraph

traced_graph = trace_graph(
    graph,
    config=ConntrailConfig(
        export_format="stdout",
        sample_rate=0.1,
    )
)

result = await traced_graph.ainvoke({"message": "Hello!"})
```

**When to use:** Quick setup, debugging unknown issues

---

### Pattern 2: Trace Specific Nodes Only

Trace only routing nodes (skip handlers):

```python
traced_graph = trace_graph(
    graph,
    config=ConntrailConfig(
        export_format="jsonl",
        sample_rate=0.2,
    ),
    only_nodes={"classify_query", "route_request"}  # Only these nodes
)
```

**When to use:** Production systems, reduce noise from non-routing nodes

---

### Pattern 3: Decorate Single Function

Trace one specific node:

```python
from conntrail import trace_node, ConntrailConfig

@trace_node(config=ConntrailConfig(export_format="stdout"))
async def classify_query(state):
    # Your routing logic
    return {"category": "refund"}
```

**When to use:** Targeted debugging of a problematic node

---

### Pattern 4: Access Traces Programmatically

Get traces in your code for custom analysis:

```python
config = ConntrailConfig(
    export_format="stdout",
    sample_rate=1.0,
    async_mode=False,  # Wait for analysis to complete
)

traced_graph = trace_graph(graph, config=config)

result = await traced_graph.ainvoke({"message": "Hello!"})

# Access traces from result
traces = result.get("__conntrail_traces__", [])

for trace in traces:
    print(f"Node: {trace.node_id}")
    print(f"Entropy: {trace.entropy_score}")
    print(f"Stability: {trace.stability}")
```

**When to use:** Building custom dashboards, automated testing

---

## Production Deployment

### Setting Up Alert Callbacks

```python
import requests
from conntrail import ConntrailConfig

def send_to_slack(record):
    """Send fragile decision alerts to Slack."""
    if record.entropy_score < 0.6:
        return
    
    message = {
        "text": f"🚨 Fragile routing detected!\n"
                f"Node: {record.node_id}\n"
                f"Entropy: {record.entropy_score:.2f}\n"
                f"Input: {record.original_input[:100]}..."
    }
    
    requests.post(
        "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
        json=message
    )

config = ConntrailConfig(
    export_format="jsonl",
    sample_rate=0.1,
    entropy_alert_threshold=0.6,
    on_alert=send_to_slack,
)
```

### JSONL Export for Audit Trails

```python
config = ConntrailConfig(
    export_format="jsonl",
    export_path="/var/log/conntrail",  # Persistent storage
    sample_rate=0.2,
)
```

**Output files:**
```
/var/log/conntrail/
  conntrail_traces_2026-04-03.jsonl
  conntrail_traces_2026-04-04.jsonl
```

**Analyzing logs:**

```python
import json

# Read traces
with open("conntrail_traces_2026-04-03.jsonl") as f:
    traces = [json.loads(line) for line in f]

# Find fragile decisions
fragile = [t for t in traces if t["stability"] == "fragile"]

print(f"Found {len(fragile)} fragile decisions out of {len(traces)}")
```

### Performance Considerations

**Hot Path Overhead:**
- Conntrail adds <20ms overhead to traced node calls
- Original node call completes before analysis starts
- No blocking of user-facing operations

**Analysis Cost:**
- Each trace generates 3 LLM calls (for variants)
- Use `sample_rate` to control costs
- Analysis runs asynchronously (unless `async_mode=False`)

**Resource Usage:**
- Shared exporter instances reduce file handle usage
- Async file I/O prevents event loop blocking
- LRU cache for prompt templates reduces disk I/O

### Security Considerations

**Data Privacy:**
- Conntrail stores input text in trace records
- Be careful with PII/sensitive data
- Consider sampling or filtering for sensitive inputs

**API Keys:**
- Conntrail uses your existing LLM provider keys
- No additional keys required
- Keys never logged or exposed in traces

---

## Interpreting Results

### Common Patterns and What They Mean

#### Pattern 1: Always Confident (entropy=0.00)

```
[CONNTRAIL] node_name | CONFIDENT | entropy=0.00 | attr: none detected
```

**Interpretation:** Your routing is rock-solid. All input variants produce the same route.

**Action:** None needed - this is the goal!

---

#### Pattern 2: Boundary with Similar Variant Flip

```
[CONNTRAIL] node_name | BOUNDARY | entropy=0.50 | attr: surface form
  → Route:   route_a
  → Alt:     route_b
```

**Interpretation:** Even slight rephrasing changes the route. Your router is too sensitive to exact wording.

**Action:** 
- Review training data for inconsistent labeling
- Consider using embeddings instead of keywords
- Add more examples of similar phrasings

---

#### Pattern 3: Fragile with High Entropy

```
[CONNTRAIL] node_name | FRAGILE | entropy=0.85 | attr: semantic intensity
  → Route:   urgent
  → Alt:     routine
```

**Interpretation:** The input is right on the decision boundary. Small changes flip the route.

**Action:**
- Flag for human review in production
- Collect these examples for model retraining
- Consider adding a "uncertain" category

---

#### Pattern 4: Handler Node Fragility

```
[CONNTRAIL] handle_request | FRAGILE | entropy=1.00
```

**Interpretation:** Handler nodes (that generate responses) often show high entropy because they embed input text into output.

**Action:** This is usually expected behavior. Focus monitoring on classifier nodes instead:

```python
trace_graph(graph, config=config, only_nodes={"classifier", "router"})
```

---

### Interpreting Attribution Dimensions

| Attribution | Root Cause | Solution |
|-------------|------------|----------|
| **surface form** | Router relies on exact keywords | Use semantic similarity instead |
| **urgency/sentiment** | Router overweights tone | Add neutral examples to training |
| **semantic intensity** | Router conflates strength with category | Separate intensity from intent |
| **none detected** | Stable routing | No action needed |

---

## Troubleshooting

### Common Issues

#### Issue 1: "No traces appearing"

**Symptoms:** No Conntrail output in console or files

**Possible causes:**
1. `sample_rate=0` - Set to a value > 0
2. `export_format` not set - Defaults to "jsonl", check the export_path
3. Nodes wrapped but not called - Ensure the traced graph is being used

**Solution:**
```python
config = ConntrailConfig(
    sample_rate=1.0,  # Make sure > 0
    export_format="stdout",  # Use stdout for visibility
)
```

---

#### Issue 2: "All entropy scores are 1.00"

**Symptoms:** Every trace shows FRAGILE with entropy=1.00

**Possible causes:**
1. Handler nodes are being traced (they generate variable text)
2. Router returns full response instead of just route key

**Solution:**
```python
# Only trace routing nodes, not handlers
trace_graph(graph, config=config, only_nodes={"classifier", "router"})
```

---

#### Issue 3: "Rate limit errors"

**Symptoms:** Tests fail with 429 rate limit errors

**Possible causes:**
1. Too many concurrent traces
2. Sample rate too high for your tier

**Solution:**
```python
config = ConntrailConfig(
    sample_rate=0.1,  # Reduce sampling
    contrast_model="groq-llama3-8b-8192",  # Use cheaper model
)
```

Conntrail has built-in retry logic with exponential backoff for rate limits.

---

#### Issue 4: "High latency"

**Symptoms:** Node calls taking too long

**Possible causes:**
1. `async_mode=False` - Analysis blocks the hot path
2. Analysis timeout too high

**Solution:**
```python
config = ConntrailConfig(
    async_mode=True,  # Fire-and-forget analysis
    analysis_timeout=10.0,  # Lower timeout
)
```

---

#### Issue 5: "Empty trace list in result"

**Symptoms:** `result.get("__conntrail_traces__")` returns empty list

**Possible causes:**
1. `async_mode=True` (default) - Traces not injected into result
2. Sampling skipped this call
3. Analysis not yet complete

**Solution:**
```python
config = ConntrailConfig(
    async_mode=False,  # Required for programmatic access
    sample_rate=1.0,   # Ensure tracing
)
```

---

## API Reference

### Core Classes

#### ConntrailConfig

```python
@dataclass
class ConntrailConfig:
    contrast_model: str = "claude-haiku-4-5-20251001"
    sample_rate: float = 1.0
    async_mode: bool = True
    export_format: Literal["jsonl", "stdout"] = "jsonl"
    export_path: str = "./conntrail_traces"
    entropy_alert_threshold: float = 0.6
    on_alert: Callable | None = None
    analysis_timeout: float = 30.0
```

#### trace_graph

```python
def trace_graph(
    compiled_graph: Any,
    config: ConntrailConfig | None = None,
    *,
    input_key: str = "message",
    route_key: str | None = None,
    only_nodes: set[str] | None = None,
) -> Any
```

**Parameters:**
- `compiled_graph`: Your compiled LangGraph
- `config`: ConntrailConfig instance
- `input_key`: State key containing text input (default: "message")
- `route_key`: State key for routing decision (auto-detected if None)
- `only_nodes`: Set of node names to trace (None = all nodes)

**Returns:** The same graph with tracing wrapped around nodes

---

#### trace_node

```python
def trace_node(
    config: ConntrailConfig | None = None,
    *,
    input_key: str = "message",
    route_key: str | None = None,
) -> Callable
```

**Usage:**
```python
@trace_node(config=ConntrailConfig())
async def my_node(state):
    return state
```

---

### TraceRecord Fields

```python
@dataclass
class TraceRecord:
    trace_id: str                    # Unique UUID
    node_id: str                     # Node name
    timestamp: datetime              # When trace occurred
    original_input: str              # Input text
    original_route: str              # Route taken
    entropy_score: float             # 0.0 to 1.0
    stability: str                   # "confident" / "boundary" / "fragile"
    attribution_dimension: str       # What drives changes
    plain_language_summary: str      # Human-readable explanation
    raw_contrasts: ContrastSet       # The 3 variants generated
    raw_outputs: dict                # All 4 routing outcomes
    counterfactual_route: str | None # Where opposite variant went
```

---

## Examples

See the `examples/` directory for complete working examples:

1. **customer_support.py** - Basic tracing with alert callbacks
2. **content_moderation.py** - JSONL export for audit trails
3. **research_agent.py** - Selective tracing with programmatic access

Run them with:

```bash
cd examples
python customer_support.py
python content_moderation.py
python research_agent.py
```

---

## Support

- **GitHub Issues:** https://github.com/Conntrail/conntrail/issues
- **Documentation:** https://github.com/Conntrail/conntrail#readme
- **Showcase:** See [CONNTAIL_SHOWCASE.md](../testing/results/CONNTAIL_SHOWCASE.md) for real test results

---

**Happy tracing! 🚀**
