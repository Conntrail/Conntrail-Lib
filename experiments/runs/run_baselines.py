"""
Baseline optimization experiments.

Runs two baselines for comparison against CPE-GEPA:
  - scalar:   accuracy-only feedback (no routing-stability signal)
  - critique: LLM self-critique on wrong predictions (no CPE)

Usage:
    python -m experiments.runs.run_baselines --agents all --iterations 10 \\
        --out experiments/results/baselines.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / "testing" / ".env")

from langchain_core.messages import HumanMessage  # noqa: E402

logging.basicConfig(level=logging.WARNING)

# ── Prompt evolution helpers ───────────────────────────────────────────────────

_SCALAR_TEMPLATE = """\
You are optimizing a routing system prompt for an LLM agent.

=== Current prompt ===
{current_prompt}

=== Evaluation results (iteration {iteration}) ===
Routing accuracy: {accuracy:.0%} ({correct}/{total} correct)

Generate an improved version of the routing system prompt to increase accuracy.
Return ONLY the improved prompt — no explanation, no markdown fences.
"""

_CRITIQUE_TEMPLATE = """\
You are optimizing a routing system prompt for an LLM agent.

=== Current prompt ===
{current_prompt}

=== Evaluation results (iteration {iteration}) ===
Routing accuracy: {accuracy:.0%} ({correct}/{total} correct)

=== Wrong predictions ===
{wrong_predictions}

Generate an improved version of the routing system prompt that fixes these errors.
Return ONLY the improved prompt — no explanation, no markdown fences.
"""


async def _next_prompt_scalar(llm, current_prompt, accuracy, correct, total, iteration):
    msg = _SCALAR_TEMPLATE.format(
        current_prompt=current_prompt,
        accuracy=accuracy,
        correct=correct,
        total=total,
        iteration=iteration,
    )
    r = await llm.ainvoke([HumanMessage(content=msg)])
    return r.content.strip()


async def _next_prompt_critique(llm, current_prompt, accuracy, correct, total, wrong, iteration):
    wrong_str = "\n".join(
        f"- Input: {w['input']!r}\n  Got: {w['got']!r}  Expected: {w['expected']!r}"
        for w in wrong
    ) or "None"
    msg = _CRITIQUE_TEMPLATE.format(
        current_prompt=current_prompt,
        accuracy=accuracy,
        correct=correct,
        total=total,
        wrong_predictions=wrong_str,
        iteration=iteration,
    )
    r = await llm.ainvoke([HumanMessage(content=msg)])
    return r.content.strip()


# ── Core loop ─────────────────────────────────────────────────────────────────

def _get_llm():
    from langchain_ollama import ChatOllama
    return ChatOllama(model="qwen2.5:7b", num_predict=1024, temperature=0.0)


async def run_baseline(name: str, adapter, method: str, num_iterations: int) -> dict:

    print(f"\n{'='*60}")
    print(f"{method.upper()} baseline: {name}  ({num_iterations} iterations)")
    print(f"{'='*60}")

    trainset = adapter.TRAINSET
    prompt_holder = adapter.PromptHolder()
    raw_graph = adapter.build_graph(prompt_holder)

    # Baselines do not use CPE traces — run the graph unmodified.
    # (Conntrail instrumentation is omitted to avoid rate-limit interference
    #  with the CPE-GEPA run when both experiments share the same Groq quota.)
    instrumented = raw_graph

    llm = _get_llm()
    current_prompt = adapter.DEFAULT_SYSTEM_PROMPT
    iteration_results = []

    for i in range(num_iterations):
        iter_start = time.time()
        print(f"\n  Iteration {i+1}/{num_iterations}")

        prompt_holder.system_prompt = current_prompt

        correct = 0
        wrong = []
        for ex in trainset:
            state = adapter.make_initial_state(ex["input"])
            try:
                result = await instrumented.ainvoke(state)
                got = adapter.get_result_route(result)
                if got == ex["expected_route"]:
                    correct += 1
                else:
                    wrong.append({"input": ex["input"], "got": got, "expected": ex["expected_route"]})
                print(f"    [{ex['entropy_category']:9s}] got={got:15s} expected={ex['expected_route']}")
            except Exception as exc:
                print(f"    ERROR: {exc}")
                wrong.append({"input": ex["input"], "got": "error", "expected": ex["expected_route"]})

        accuracy = correct / len(trainset) if trainset else 0.0
        print(f"  → accuracy={accuracy:.0%}")

        iter_result = {
            "iteration": i + 1,
            "accuracy": round(accuracy, 4),
            "correct": correct,
            "total": len(trainset),
            "elapsed_s": round(time.time() - iter_start, 2),
        }
        iteration_results.append(iter_result)

        if i < num_iterations - 1:
            try:
                if method == "scalar":
                    current_prompt = await _next_prompt_scalar(
                        llm, current_prompt, accuracy, correct, len(trainset), i + 1
                    )
                else:  # critique
                    current_prompt = await _next_prompt_critique(
                        llm, current_prompt, accuracy, correct, len(trainset), wrong, i + 1
                    )
            except Exception as exc:
                print(f"  Prompt generation failed: {exc} — reusing current prompt")

    return {
        "agent": name,
        "method": method,
        "num_iterations": num_iterations,
        "trainset_size": len(trainset),
        "iterations": iteration_results,
        "final_score": iteration_results[-1]["accuracy"] if iteration_results else None,
    }


# ── Entry point ────────────────────────────────────────────────────────────────

AGENT_MODULES = {
    "customer_support":       "experiments.agents.customer_support.adapter",
    "react_agent":            "experiments.agents.react_agent.adapter",
    "multi_agent_supervisor": "experiments.agents.multi_agent_supervisor.adapter",
    "adaptive_rag":           "experiments.agents.adaptive_rag.adapter",
}


async def main(agents: list[str], methods: list[str], iterations: int, out: str) -> None:
    import importlib

    results = []
    for name in agents:
        if name not in AGENT_MODULES:
            print(f"Unknown agent {name!r}, skipping.")
            continue
        try:
            adapter = importlib.import_module(AGENT_MODULES[name])
        except Exception:
            print(f"Could not import adapter for {name}: {traceback.format_exc()}")
            continue

        for method in methods:
            try:
                result = await run_baseline(name, adapter, method, iterations)
                results.append(result)
            except Exception:
                print(f"\nERROR running {name}/{method}:")
                traceback.print_exc()
                results.append({
                    "agent": name,
                    "method": method,
                    "error": traceback.format_exc(),
                })

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", default="all")
    parser.add_argument("--baseline", default="all", help="scalar | critique | all")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--out", default="experiments/results/baselines.json")
    args = parser.parse_args()

    targets = (
        list(AGENT_MODULES.keys())
        if args.agents == "all"
        else [a.strip() for a in args.agents.split(",")]
    )
    methods = (
        ["scalar", "critique"]
        if args.baseline == "all"
        else [args.baseline]
    )
    asyncio.run(main(targets, methods, args.iterations, args.out))
