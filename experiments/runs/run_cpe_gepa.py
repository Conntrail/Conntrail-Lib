"""
CPE-GEPA optimization experiment.

Runs a GEPA-style prompt optimization loop for each agent. CPE entropy scores
from TraceCollector become the feedback signal used by Sonnet to improve the
routing system prompt each iteration.

Usage:
    python -m experiments.runs.run_cpe_gepa --agents all --iterations 10 --runs 3 \\
        --out experiments/results/cpe_gepa.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # picks up ANTHROPIC_API_KEY from root .env
load_dotenv(Path(__file__).parent.parent.parent / "testing" / ".env")

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from tqdm import tqdm  # noqa: E402

from conntrail import ConntrailConfig, trace_graph  # noqa: E402
from conntrail.gepa.bridge import TraceCollector  # noqa: E402
from conntrail.gepa.feedback import cpe_feedback  # noqa: E402

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("cpe_gepa")

ERROR_LOG = Path("experiments/results/errors.log")

EXPERIMENT_CONFIG = {
    "optimization_loop": "custom_gepa_implementation",
    "router_model": "claude-haiku-4-5-20251001",
    "contrast_model": "claude-haiku-4-5-20251001",
    "prompt_gen_model": "claude-sonnet-4-6",
    "router_temperature": 0.0,
    "prompt_gen_temperature": 0.3,
    "note_on_variance": (
        "temperature=0.3 on prompt generator produces genuine trajectory variance "
        "across runs; router temperature=0.0 ensures deterministic routing decisions"
    ),
}

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


def _get_llm(model: str = "claude-sonnet-4-6"):
    from conntrail.utils.providers import get_chat_model
    # temperature=0.3 so multiple runs produce different optimization trajectories
    return get_chat_model(model, max_tokens=1024, temperature=0.3)


def _log_error(msg: str) -> None:
    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ERROR_LOG, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}\n")


# ── Single optimization pass ───────────────────────────────────────────────────

async def _run_single(
    name: str,
    adapter,
    num_iterations: int,
    router_model: str = "claude-haiku-4-5-20251001",
    contrast_model: str = "claude-haiku-4-5-20251001",
    prompt_gen_model: str = "claude-sonnet-4-6",
) -> tuple[list[dict], int]:
    """One optimization pass from DEFAULT_SYSTEM_PROMPT. Returns (iteration_results, trainset_size)."""
    trainset = getattr(adapter, "TRAINSET_LARGE", adapter.TRAINSET)

    prompt_holder = adapter.PromptHolder()
    raw_graph = adapter.build_graph(prompt_holder, model=router_model)

    collector = TraceCollector()
    conntrail_cfg = collector.make_config(ConntrailConfig(
        contrast_model=contrast_model,
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

    current_prompt = adapter.DEFAULT_SYSTEM_PROMPT
    iteration_results = []

    iter_bar = tqdm(range(num_iterations), desc=f"  {name}", unit="iter", leave=True)
    for i in iter_bar:
        iter_start = time.time()

        prompt_holder.system_prompt = current_prompt
        collector.begin_attempt(current_prompt)

        correct = 0
        ex_bar = tqdm(trainset, desc=f"    iter {i+1}", unit="ex", leave=False)
        for ex in ex_bar:
            state = adapter.make_initial_state(ex["input"])
            try:
                result = await instrumented.ainvoke(state)
                got = adapter.get_result_route(result)
                if got == ex["expected_route"]:
                    correct += 1
                ex_bar.set_postfix(got=got[:12], ok="✓" if got == ex["expected_route"] else "✗")
            except Exception as exc:
                tqdm.write(f"      ERROR on example: {exc}")
                _log_error(f"{name} iter {i+1} example error: {exc}")

        accuracy = correct / len(trainset) if trainset else 0.0
        record = collector.end_attempt(scalar_score=accuracy)

        mean_e = record.mean_entropy
        feedback_str = cpe_feedback(record)

        entropy_str = f"{mean_e:.3f}" if mean_e is not None else "N/A"
        iter_bar.set_postfix(acc=f"{accuracy:.0%}", entropy=entropy_str,
                             fragile=record.fragile_count, boundary=record.boundary_count)
        tqdm.write(f"  iter {i+1:2d}/{num_iterations} — acc={accuracy:.0%}  "
                   f"entropy={entropy_str}  fragile={record.fragile_count}  "
                   f"boundary={record.boundary_count}  "
                   f"elapsed={time.time()-iter_start:.0f}s")

        iteration_results.append({
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
        })

        if i < num_iterations - 1:
            try:
                llm = _get_llm(prompt_gen_model)  # fresh token right before use
                current_prompt = await generate_next_prompt(
                    llm, current_prompt, accuracy, correct, len(trainset),
                    feedback_str, i + 1,
                )
            except Exception as exc:
                tqdm.write(f"    Prompt generation failed: {exc} — reusing current prompt")
                _log_error(f"{name} iter {i+1} prompt gen error: {exc}")

    return iteration_results, len(trainset)


# ── Multi-run wrapper ──────────────────────────────────────────────────────────

async def run_agent_cpe_gepa(
    name: str,
    adapter,
    num_iterations: int,
    num_runs: int,
    router_model: str = "claude-haiku-4-5-20251001",
    contrast_model: str = "claude-haiku-4-5-20251001",
    prompt_gen_model: str = "claude-sonnet-4-6",
) -> dict:
    print(f"\n{'='*60}")
    print(f"CPE-GEPA: {name}  ({num_iterations} iterations × {num_runs} runs)")
    print(f"  router={router_model}  contrast={contrast_model}  prompt_gen={prompt_gen_model}")
    print(f"{'='*60}")

    all_run_scores: list[list[float]] = []
    last_iteration_results: list[dict] = []
    trainset_size = 0

    for run_idx in range(num_runs):
        tqdm.write(f"\n  ── Run {run_idx + 1}/{num_runs} ──")
        try:
            iteration_results, trainset_size = await _run_single(
                name, adapter, num_iterations,
                router_model=router_model,
                contrast_model=contrast_model,
                prompt_gen_model=prompt_gen_model,
            )
            scores = [it["accuracy"] for it in iteration_results]
            all_run_scores.append(scores)
            last_iteration_results = iteration_results
        except Exception as exc:
            msg = f"CPE-GEPA {name} run {run_idx + 1}: {traceback.format_exc()}"
            tqdm.write(f"\n  ERROR: {exc}")
            _log_error(msg)

    if not all_run_scores:
        return {"agent": name, "method": "cpe_gepa", "error": "all runs failed"}

    final_scores = [scores[-1] for scores in all_run_scores]
    best_scores = [max(scores) for scores in all_run_scores]

    mean_final = statistics.mean(final_scores)
    std_final = statistics.stdev(final_scores) if len(final_scores) > 1 else 0.0
    mean_best = statistics.mean(best_scores)

    config = {
        **EXPERIMENT_CONFIG,
        "router_model": router_model,
        "contrast_model": contrast_model,
        "prompt_gen_model": prompt_gen_model,
        "num_runs": num_runs,
        "random_seed": 42,
    }

    return {
        "agent": name,
        "method": "cpe_gepa",
        "num_runs": num_runs,
        "num_iterations": num_iterations,
        "trainset_size": trainset_size,
        "iterations": last_iteration_results,
        "scores_per_run": all_run_scores,
        "final_score": round(mean_final, 4),
        "final_score_mean": round(mean_final, 4),
        "final_score_std": round(std_final, 4),
        "best_score_mean": round(mean_best, 4),
        "experiment_config": config,
    }


# ── Entry point ────────────────────────────────────────────────────────────────

AGENT_MODULES = {
    "customer_support":       "experiments.agents.customer_support.adapter",
    "react_agent":            "experiments.agents.react_agent.adapter",
    "multi_agent_supervisor": "experiments.agents.multi_agent_supervisor.adapter",
    "adaptive_rag":           "experiments.agents.adaptive_rag.adapter",
    "document_triage":        "experiments.agents.document_triage.adapter",
    "code_review":            "experiments.agents.code_review.adapter",
    "medical_triage":         "experiments.agents.medical_triage.adapter",
    "financial_query":        "experiments.agents.financial_query.adapter",
    "mental_health_triage":   "experiments.agents.mental_health_triage.adapter",
    "content_moderation":     "experiments.agents.content_moderation.adapter",
}


_LOCAL_MODEL = "local"  # resolves to LOCAL_MODEL_NAME env var = full model ID

async def main(
    agents: list[str],
    iterations: int,
    runs: int,
    out: str,
    resume: bool = False,
    local: bool = False,
    router_model: str | None = None,
    contrast_model: str | None = None,
    prompt_gen_model: str | None = None,
) -> None:
    import importlib

    # Model resolution: explicit flags > --local shortcut > Anthropic defaults
    r_model  = router_model      or (_LOCAL_MODEL if local else "claude-haiku-4-5-20251001")
    c_model  = contrast_model    or (_LOCAL_MODEL if local else "claude-haiku-4-5-20251001")
    pg_model = prompt_gen_model  or (_LOCAL_MODEL if local else "claude-sonnet-4-6")

    if local:
        print(f"[LOCAL MODE] router={r_model}  contrast={c_model}  prompt_gen={pg_model}\n")

    out_path = Path(out)
    results = []
    if resume and out_path.exists():
        existing = json.loads(out_path.read_text())
        results = [r for r in existing if "error" not in r]
        completed = {r["agent"] for r in results}
        agents = [a for a in agents if a not in completed]
        print(f"Resuming — already done: {sorted(completed)}")
        print(f"Still to run: {agents}\n")

    for name in agents:
        if name not in AGENT_MODULES:
            print(f"Unknown agent {name!r}, skipping.")
            continue
        try:
            adapter = importlib.import_module(AGENT_MODULES[name])
            result = await run_agent_cpe_gepa(
                name, adapter, iterations, runs,
                router_model=r_model,
                contrast_model=c_model,
                prompt_gen_model=pg_model,
            )
            results.append(result)
        except Exception:
            msg = f"CPE-GEPA outer error for {name}: {traceback.format_exc()}"
            print(f"\nERROR running {name}:")
            traceback.print_exc()
            _log_error(msg)
            results.append({"agent": name, "method": "cpe_gepa", "error": traceback.format_exc()})

        # Incremental save after each agent
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", default="all")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of independent optimization runs for variance estimation")
    parser.add_argument("--out", default="experiments/results/cpe_gepa.json")
    parser.add_argument("--resume", action="store_true",
                        help="Skip agents that already have a successful result in --out")
    parser.add_argument("--local", action="store_true",
                        help="Use local Qwen3 model (http://127.0.0.1:8888) for all LLM calls")
    parser.add_argument("--router-model",  default=None, help="Override router model")
    parser.add_argument("--contrast-model", default=None, help="Override contrast model")
    parser.add_argument("--prompt-gen-model", default=None, help="Override prompt generation model")
    args = parser.parse_args()

    targets = (
        list(AGENT_MODULES.keys())
        if args.agents == "all"
        else [a.strip() for a in args.agents.split(",")]
    )
    asyncio.run(main(
        targets, args.iterations, args.runs, args.out,
        resume=args.resume,
        local=args.local,
        router_model=args.router_model,
        contrast_model=args.contrast_model,
        prompt_gen_model=args.prompt_gen_model,
    ))
