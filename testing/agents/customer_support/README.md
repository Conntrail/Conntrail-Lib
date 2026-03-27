# Customer Support Agent

Adapted from: `langchain-ai/langgraph` examples/customer-support/

## Description

A multi-route customer support agent that classifies incoming queries and routes them to:
- `handle_refund` — refund or return requests
- `handle_escalation` — urgent complaints or manager requests
- `handle_order_info` — order status or tracking
- `handle_general` — all other queries

## Why it's a good Conntrail test subject

Clear routing with 4 distinct branches. "Urgency" and "sentiment" are the primary attribution dimensions. Easy to construct confident / boundary / fragile inputs.

## Source

```
git clone https://github.com/langchain-ai/langgraph
# See: examples/customer-support/
```

## Running standalone

```bash
cd testing/agents/customer_support
python agent.py
```
