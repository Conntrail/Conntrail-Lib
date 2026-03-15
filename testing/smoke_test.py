"""
Phase 0 smoke test — runs all 4 agents and records results.

Usage:
    GROQ_API_KEY=... python testing/smoke_test.py

Output is printed to stdout and saved to testing/smoke_test_results.txt
"""
from __future__ import annotations

import asyncio
import sys
import traceback
from datetime import datetime
from io import StringIO

from langchain_core.messages import HumanMessage


DIVIDER = "=" * 70

lines: list[str] = []


def log(text: str = "") -> None:
    print(text)
    lines.append(text)


async def run_customer_support() -> bool:
    log(f"\n{DIVIDER}")
    log("AGENT 1: Customer Support (4-way routing)")
    log(DIVIDER)

    from testing.agents.customer_support.agent import build_graph

    graph = build_graph()
    test_cases = [
        ("I want a full refund, my order arrived broken!", "refund"),
        ("I need to speak to a manager RIGHT NOW, this is outrageous!", "escalation"),
        ("Where is my order #98765? It's been 5 days.", "order_info"),
        ("What are your store opening hours?", "general"),
    ]

    all_passed = True
    for message, expected_category in test_cases:
        try:
            result = await graph.ainvoke({
                "message": message,
                "category": None,
                "response": None,
            })
            got = result["category"]
            status = "PASS" if got == expected_category else "WARN"
            if got != expected_category:
                all_passed = False
            log(f"  [{status}] Input: {message[:55]!r}")
            log(f"         Category: {got} (expected: {expected_category})")
            log(f"         Response: {result['response'][:100]}...")
        except Exception as e:
            log(f"  [FAIL] {message[:55]!r}")
            log(f"         Error: {e}")
            all_passed = False

    return all_passed


async def run_react_agent() -> bool:
    log(f"\n{DIVIDER}")
    log("AGENT 2: ReAct Agent (tool-selection routing)")
    log(DIVIDER)

    from testing.agents.react_agent.agent import build_graph

    graph = build_graph()
    test_cases = [
        ("What is 144 / 12?", "calculator"),
        ("What is the capital of France?", "direct"),
    ]

    all_passed = True
    for question, expected_path in test_cases:
        try:
            result = await graph.ainvoke({
                "messages": [HumanMessage(content=question)]
            })
            last_msg = result["messages"][-1].content
            # Detect if tool was used by checking message history length
            used_tool = len(result["messages"]) > 2
            path = "calculator" if used_tool else "direct"
            status = "PASS" if path == expected_path else "WARN"
            if path != expected_path:
                all_passed = False
            log(f"  [{status}] Input: {question!r}")
            log(f"         Path taken: {path} (expected: {expected_path})")
            log(f"         Answer: {last_msg[:100]}")
        except Exception as e:
            log(f"  [FAIL] {question!r}")
            log(f"         Error: {e}")
            all_passed = False

    return all_passed


async def run_supervisor() -> bool:
    log(f"\n{DIVIDER}")
    log("AGENT 3: Multi-Agent Supervisor (specialist routing)")
    log(DIVIDER)

    from testing.agents.multi_agent_supervisor.agent import build_graph

    graph = build_graph()
    test_cases = [
        ("Write a Python function to sort a list of dictionaries by a key.", "code_agent"),
        ("Write a professional email to decline a job offer politely.", "writing_agent"),
    ]

    all_passed = True
    for task, expected_agent in test_cases:
        try:
            result = await graph.ainvoke({
                "task": task,
                "assigned_agent": None,
                "result": None,
            })
            got = result["assigned_agent"]
            status = "PASS" if got == expected_agent else "WARN"
            if got != expected_agent:
                all_passed = False
            log(f"  [{status}] Task: {task[:55]!r}")
            log(f"         Assigned to: {got} (expected: {expected_agent})")
            log(f"         Result: {result['result'][:100]}...")
        except Exception as e:
            log(f"  [FAIL] {task[:55]!r}")
            log(f"         Error: {e}")
            all_passed = False

    return all_passed


async def run_adaptive_rag() -> bool:
    log(f"\n{DIVIDER}")
    log("AGENT 4: Adaptive RAG (retrieval strategy routing)")
    log(DIVIDER)

    from testing.agents.adaptive_rag.agent import build_graph

    graph = build_graph()
    test_cases = [
        ("What was Q4 2024 sales performance?", "vector_search"),
        ("What is the boiling point of water?", "direct_answer"),
    ]

    all_passed = True
    for query, expected_strategy in test_cases:
        try:
            result = await graph.ainvoke({
                "query": query,
                "strategy": None,
                "retrieved_docs": [],
                "answer": None,
            })
            got = result["strategy"]
            status = "PASS" if got == expected_strategy else "WARN"
            if got != expected_strategy:
                all_passed = False
            log(f"  [{status}] Query: {query!r}")
            log(f"         Strategy: {got} (expected: {expected_strategy})")
            log(f"         Answer: {result['answer'][:100]}...")
        except Exception as e:
            log(f"  [FAIL] {query!r}")
            log(f"         Error: {e}")
            all_passed = False

    return all_passed


async def main() -> None:
    from testing.harness.llm import detect_provider

    log(DIVIDER)
    log("CONTRAIL — Phase 0 Smoke Test")
    log(f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        provider = detect_provider()
        log(f"Provider  : {provider.upper()}")
    except EnvironmentError as e:
        log(f"ERROR: {e}")
        log("Set GROQ_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY and retry.")
        sys.exit(1)
    log(DIVIDER)

    results: dict[str, bool] = {}
    runners = [
        ("customer_support", run_customer_support),
        ("react_agent", run_react_agent),
        ("multi_agent_supervisor", run_supervisor),
        ("adaptive_rag", run_adaptive_rag),
    ]

    for name, runner in runners:
        try:
            passed = await runner()
            results[name] = passed
        except Exception:
            log(f"\n[FATAL] {name} crashed:")
            log(traceback.format_exc())
            results[name] = False

    # --- Summary ---
    log(f"\n{DIVIDER}")
    log("SUMMARY")
    log(DIVIDER)
    for name, passed in results.items():
        status = "PASS" if passed else "WARN"
        log(f"  [{status}] {name}")
    total = len(results)
    passed_count = sum(results.values())
    log(f"\n  {passed_count}/{total} agents fully matched expected routes")
    log(f"  (WARN = agent ran but routed differently than expected — LLM variance is normal)")
    log(DIVIDER)

    # --- Write to file ---
    out_path = "testing/smoke_test_results.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
