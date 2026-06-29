"""
Conntrail demo — run against the customer support agent.

Shows:
  1. A "confident" input  → low entropy, routing is stable
  2. A "boundary" input   → mid entropy, one contrast flips the route
  3. A "fragile" input    → high entropy, multiple contrasts flip

Usage:
    python demo.py

Requires GROQ_API_KEY (or ANTHROPIC_API_KEY / OPENAI_API_KEY) in the environment,
or in testing/.env if you have one from the integration test suite.
"""
import asyncio
from dotenv import load_dotenv

load_dotenv("testing/.env")

from conntrail import ConntrailConfig, trace_graph
from testing.agents.customer_support.agent import build_graph


INPUTS = [
    ("confident", "What are your business hours?"),
    ("boundary",  "My order hasn't arrived and it's been 3 weeks, I want a refund immediately!"),
    ("fragile",   "Maybe I should return this, not sure yet."),
]


async def main():
    graph = trace_graph(
        build_graph(),
        config=ConntrailConfig(
            contrast_model="llama-3.1-8b-instant",  # fast + cheap
            sample_rate=1.0,
            async_mode=False,        # await analysis so traces appear in result
            export_format="stdout",  # print to terminal
        ),
        only_nodes={"classify_query"},  # only trace the routing node
    )

    for label, message in INPUTS:
        print(f"\n{'='*60}")
        print(f"  [{label.upper()}] {message}")
        print('='*60)
        result = await graph.ainvoke(
            {"message": message, "category": None, "response": None}
        )
        traces = result.get("__conntrail_traces__", [])
        if traces:
            t = traces[0]
            print(f"\n  entropy={t.entropy_score:.2f}  stability={t.stability}")
            print(f"  route → {t.original_route}")
            if t.counterfactual_route:
                print(f"  counterfactual → {t.counterfactual_route}")
            print(f"  {t.plain_language_summary}")


if __name__ == "__main__":
    asyncio.run(main())
