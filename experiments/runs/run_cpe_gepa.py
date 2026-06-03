"""
CPE-GEPA optimization experiment.

Runs a GEPA-style prompt optimization loop for each of the four Conntrail
test agents. CPE entropy scores from TraceCollector become the feedback
signal used by an LLM to improve the routing system prompt each iteration.

No dspy dependency — the optimization loop is implemented directly.

Usage:
    python -m experiments.runs.run_cpe_gepa --agents all --iterations 10 \\
        --out experiments/results/cpe_gepa.json
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

# ── Patch Conntrail's provider factory to support Ollama ──────────────────────
import conntrail.utils.providers as _conntrail_providers  # noqa: E402
from langchain_ollama import ChatOllama as _ChatOllama      # noqa: E402

_orig_get_chat_model = _conntrail_providers.get_chat_model
def _patched_get_chat_model(model, *, max_tokens=512, temperature=0.0):
    if model.startswith("ollama:"):
        return _ChatOllama(model=model[7:], num_predict=max_tokens, temperature=temperature)
    return _orig_get_chat_model(model, max_tokens=max_tokens, temperature=temperature)
_conntrail_providers.get_chat_model = _patched_get_chat_model
# ─────────────────────────────────────────────────────────────────────────────

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from conntrail import ConntrailConfig, trace_graph  # noqa: E402
from conntrail.gepa.bridge import TraceCollector  # noqa: E402
from conntrail.gepa.feedback import cpe_feedback  # noqa: E402

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("cpe_gepa")

# ── Prompt evolution ───────────────────────────────────────────────────────────

_PROMPT_GEN_TEMPLATE = """\
You are optimizing a routing system prompt for an LLM agent.

=== Current prompt ===
{current_prompt}

=== Evaluation results (iteration {iteration}) ===
Routing accuracy: {accuracy:.0%} ({correct}/{total} correct)

=== CPE routing-stability feedback ===
{cpe_feedback}

=== Task ===
Generate an improved version of the routing system prompt.
The new prompt must fix the instability described above while preserving
the routing categories exactly.
Return ONLY the improved prompt — no explanation, no markdown fences.
"""


async def generate_next_prompt(
    llm,
    current_prompt: str,
    accuracy: float,
    correct: int,
    total: int,
    feedback_str: str,
    iteration: int,
) -> str:
    prompt = _PROMPT_GEN_TEMPLATE.format(
        current_prompt=current_prompt,
        accuracy=accuracy,
        correct=correct,
        total=total,
        cpe_feedback=feedback_str,
        iteration=iteration,
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return response.content.strip()


# ── Core loop ─────────────────────────────────────────────────────────────────

def _get_llm():
    return _ChatOllama(model="qwen2.5:7b", num_predict=1024, temperature=0.0)


async def run_agent_cpe_gepa(
    name: str,
    adapter,
    num_iterations: int,
) -> dict:
    print(f"\n{'='*60}")
    print(f"CPE-GEPA: {name}  ({num_iterations} iterations)")
    print(f"{'='*60}")

    trainset = adapter.TRAINSET
    prompt_holder = adapter.PromptHolder()

    # Build graph once; swap prompt via holder each iteration
    raw_graph = adapter.build_graph(prompt_holder)

    # Conntrail config: collect every trace synchronously
    collector = TraceCollector()
    conntrail_cfg = collector.make_config(ConntrailConfig(
        contrast_model="ollama:qwen2.5:7b",   # local Ollama — no rate limits
        sample_rate=1.0,
        entropy_alert_threshold=0.0,
        async_mode=False,
        export_format="stdout",
    ))

    instrumented = trace_graph(
        raw_graph,
        config=conntrail_cfg,
        input_key=adapter.INPUT_KEY,
        route_key=adapter.ROUTE_KEY,
        only_nodes=adapter.ONLY_NODES,
    )

    llm = _get_llm()
    current_prompt = adapter.DEFAULT_SYSTEM_PROMPT

    iteration_results = []

    for i in range(num_iterations):
        iter_start = time.time()
        print(f"\n  Iteration {i+1}/{num_iterations} — prompt hash {hash(current_prompt) % 9999:04d}")

        prompt_holder.system_prompt = current_prompt
        collector.begin_attempt(current_prompt)

        correct = 0
        for ex in trainset:
            state = adapter.make_initial_state(ex["input"])
            try:
                result = await instrumented.ainvoke(state)
                got = adapter.get_result_route(result)
                if got == ex["expected_route"]:
                    correct += 1
                print(f"    [{ex['entropy_category']:9s}] got={got:15s} expected={ex['expected_route']}")
            except Exception as exc:
                print(f"    ERROR on example: {exc}")

        accuracy = correct / len(trainset) if trainset else 0.0
        record = collector.end_attempt(scalar_score=accuracy)

        mean_e = record.mean_entropy
        feedback_str = cpe_feedback(record)

        entropy_str = f"{mean_e:.3f}" if mean_e is not None else "N/A"
        print(f"  → accuracy={accuracy:.0%}  entropy={entropy_str}  "
              f"fragile={record.fragile_count}  boundary={record.boundary_count}")

        iter_result = {
            "iteration": i + 1,
            "prompt_candidate": current_prompt[:120] + ("..." if len(current_prompt) > 120 else ""),
            "accuracy": round(accuracy, 4),
            "correct": correct,
            "total": len(trainset),
            "mean_entropy": round(mean_e, 4) if mean_e is not None else None,
            "fragile_count": record.fragile_count,
            "boundary_count": record.boundary_count,
            "confident_count": len(record.traces) - record.fragile_count - record.boundary_count,
            "num_traces": len(record.traces),
            "dominant_attribution": record.dominant_attribution,
            "elapsed_s": round(time.time() - iter_start, 2),
        }
        iteration_results.append(iter_result)

        # Generate improved prompt for next iteration (skip on last)
        if i < num_iterations - 1:
            try:
                current_prompt = await generate_next_prompt(
                    llm, current_prompt, accuracy, correct, len(trainset),
                    feedback_str, i + 1,
                )
            except Exception as exc:
                print(f"  Prompt generation failed: {exc} — reusing current prompt")

    return {
        "agent": name,
        "method": "cpe_gepa",
        "num_iterations": num_iterations,
        "trainset_size": len(trainset),
        "iterations": iteration_results,
        "final_score": iteration_results[-1]["accuracy"] if iteration_results else None,
        "final_entropy": iteration_results[-1]["mean_entropy"] if iteration_results else None,
    }


# ── Entry point ────────────────────────────────────────────────────────────────

AGENT_MODULES = {
    "customer_support":       "experiments.agents.customer_support.adapter",
    "react_agent":            "experiments.agents.react_agent.adapter",
    "multi_agent_supervisor": "experiments.agents.multi_agent_supervisor.adapter",
    "adaptive_rag":           "experiments.agents.adaptive_rag.adapter",
}


async def main(agents: list[str], iterations: int, out: str) -> None:
    import importlib

    results = []
    for name in agents:
        if name not in AGENT_MODULES:
            print(f"Unknown agent {name!r}, skipping.")
            continue
        try:
            adapter = importlib.import_module(AGENT_MODULES[name])
            result = await run_agent_cpe_gepa(name, adapter, iterations)
            results.append(result)
        except Exception:
            print(f"\nERROR running {name}:")
            traceback.print_exc()
            results.append({"agent": name, "method": "cpe_gepa", "error": traceback.format_exc()})

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", default="all")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--out", default="experiments/results/cpe_gepa.json")
    args = parser.parse_args()

    targets = (
        list(AGENT_MODULES.keys())
        if args.agents == "all"
        else [a.strip() for a in args.agents.split(",")]
    )
    asyncio.run(main(targets, args.iterations, args.out))
