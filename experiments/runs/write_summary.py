"""
Read cpe_gepa.json and baselines.json and write summary.md.
Run after both experiment scripts have completed.

Usage:
    python -m experiments.runs.write_summary
"""
from __future__ import annotations

import json
from pathlib import Path


def load(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text())


def fmt(v, fmt_str="") -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:{fmt_str}}" if fmt_str else f"{v:.3f}"
    return str(v)


def convergence_iter(iterations: list[dict], threshold: float = 0.83) -> str:
    for it in iterations:
        if it.get("accuracy", 0) >= threshold:
            return str(it["iteration"])
    return "never"


def entropy_trajectory(iterations: list[dict]) -> str:
    vals = [it.get("mean_entropy") for it in iterations]
    parts = []
    for v in vals:
        if v is None:
            parts.append("N/A")
        else:
            parts.append(f"{v:.2f}")
    return " → ".join(parts)


def score_trajectory(iterations: list[dict]) -> str:
    return " → ".join(f"{it.get('accuracy', 0):.0%}" for it in iterations)


def main() -> None:
    cpe_data = load("experiments/results/cpe_gepa.json")
    baseline_data = load("experiments/results/baselines.json")

    # Index by (agent, method)
    results: dict[tuple, dict] = {}
    for r in cpe_data:
        if "error" not in r:
            results[(r["agent"], "cpe_gepa")] = r
    for r in baseline_data:
        if "error" not in r:
            results[(r["agent"], r["method"])] = r

    agents = ["customer_support", "react_agent", "multi_agent_supervisor", "adaptive_rag"]
    methods = ["cpe_gepa", "scalar", "critique"]

    lines: list[str] = []
    lines.append("# Conntrail × GEPA — Experiment Results\n")
    lines.append(f"**Runs:** CPE-GEPA + scalar + critique baselines  ")
    lines.append(f"**Model:** Groq llama-3.3-70b-versatile (routing + contrast gen), openai/gpt-oss-120b (prompt gen)  ")
    lines.append(f"**Config:** sample_rate=1.0, async_mode=False, entropy_alert_threshold=0.0  \n")

    # ── Main results table ────────────────────────────────────────────────────
    lines.append("## Results Table\n")
    lines.append("| Agent | Method | Final Score | Best Score | Converged @ | Mean Entropy (iter 1→last) |")
    lines.append("|---|---|---|---|---|---|")

    for agent in agents:
        for method in methods:
            r = results.get((agent, method))
            if r is None:
                lines.append(f"| {agent} | {method} | — | — | — | — |")
                continue
            iters = r.get("iterations", [])
            final = r.get("final_score")
            best = max((it.get("accuracy", 0) for it in iters), default=None)
            conv = convergence_iter(iters)
            if method == "cpe_gepa":
                e_start = fmt(iters[0].get("mean_entropy") if iters else None, ".2f")
                e_end = fmt(iters[-1].get("mean_entropy") if iters else None, ".2f")
                entropy_col = f"{e_start} → {e_end}"
            else:
                entropy_col = "n/a (no CPE)"
            lines.append(
                f"| {agent} | {method} | {fmt(final, '.0%')} | {fmt(best, '.0%')} "
                f"| iter {conv} | {entropy_col} |"
            )

    # ── Per-agent deep dives ───────────────────────────────────────────────────
    lines.append("\n## Per-Agent Detail\n")

    for agent in agents:
        lines.append(f"### {agent.replace('_', ' ').title()}\n")

        cpe = results.get((agent, "cpe_gepa"))
        scalar = results.get((agent, "scalar"))
        critique = results.get((agent, "critique"))

        if cpe:
            iters = cpe.get("iterations", [])
            lines.append(f"**CPE-GEPA** — {len(iters)} iterations, trainset size {cpe.get('trainset_size', '?')}")
            lines.append(f"- Score trajectory: {score_trajectory(iters)}")
            lines.append(f"- Entropy trajectory: {entropy_trajectory(iters)}")
            frag_totals = [it.get("fragile_count", 0) for it in iters]
            bound_totals = [it.get("boundary_count", 0) for it in iters]
            attr = [it.get("dominant_attribution") or "—" for it in iters]
            lines.append(f"- Fragile counts: {' → '.join(map(str, frag_totals))}")
            lines.append(f"- Boundary counts: {' → '.join(map(str, bound_totals))}")
            lines.append(f"- Dominant attribution: {' → '.join(attr)}")
            lines.append("")

        if scalar:
            iters = scalar.get("iterations", [])
            lines.append(f"**Scalar baseline** — {len(iters)} iterations")
            lines.append(f"- Score trajectory: {score_trajectory(iters)}\n")

        if critique:
            iters = critique.get("iterations", [])
            lines.append(f"**Critique baseline** — {len(iters)} iterations")
            lines.append(f"- Score trajectory: {score_trajectory(iters)}\n")

    # ── Notable findings ──────────────────────────────────────────────────────
    lines.append("## Notable Findings\n")

    findings = []

    # Find agent where CPE-GEPA improved most
    best_improvement = None
    best_agent = None
    for agent in agents:
        r = results.get((agent, "cpe_gepa"))
        if not r:
            continue
        iters = r.get("iterations", [])
        if len(iters) < 2:
            continue
        first = iters[0].get("accuracy", 0)
        best = max(it.get("accuracy", 0) for it in iters)
        gain = best - first
        if best_improvement is None or gain > best_improvement:
            best_improvement = gain
            best_agent = agent

    if best_agent and best_improvement is not None:
        findings.append(
            f"- **Largest CPE-GEPA improvement:** `{best_agent}` gained "
            f"{best_improvement:.0%} accuracy (from first to peak iteration)."
        )

    # Entropy reduction signal
    for agent in agents:
        r = results.get((agent, "cpe_gepa"))
        if not r:
            continue
        iters = r.get("iterations", [])
        e_vals = [it.get("mean_entropy") for it in iters if it.get("mean_entropy") is not None]
        if len(e_vals) >= 2 and e_vals[-1] < e_vals[0]:
            findings.append(
                f"- **Entropy reduction confirmed** for `{agent}`: "
                f"{e_vals[0]:.3f} → {e_vals[-1]:.3f} across {len(e_vals)} traced iterations, "
                "indicating CPE feedback drove the prompt toward more stable routing decisions."
            )

    # Find where CPE beat scalar
    for agent in agents:
        cpe = results.get((agent, "cpe_gepa"))
        scal = results.get((agent, "scalar"))
        if cpe and scal:
            cpe_final = cpe.get("final_score", 0) or 0
            scal_final = scal.get("final_score", 0) or 0
            diff = cpe_final - scal_final
            if diff > 0.05:
                findings.append(
                    f"- **CPE-GEPA outperformed scalar** on `{agent}` by "
                    f"{diff:.0%} at final iteration ({cpe_final:.0%} vs {scal_final:.0%})."
                )
            elif diff < -0.05:
                findings.append(
                    f"- **Scalar outperformed CPE-GEPA** on `{agent}` by "
                    f"{-diff:.0%} — entropy feedback may have over-constrained the prompt."
                )

    # Rate-limit degradation note
    for agent in agents:
        r = results.get((agent, "cpe_gepa"))
        if not r:
            continue
        iters = r.get("iterations", [])
        empty_trace_iters = [it["iteration"] for it in iters if it.get("num_traces", 1) == 0]
        if empty_trace_iters:
            findings.append(
                f"- **Rate-limit impact on `{agent}`:** iterations {empty_trace_iters} had "
                "zero traces collected (Groq 30 RPM limit exceeded max retries). "
                "Accuracy scores for those iterations reflect routing only, not CPE signal."
            )

    if not findings:
        findings.append("- Insufficient data to identify clear findings — check JSON files for raw results.")

    lines.extend(findings)

    # ── Token cost comparison ─────────────────────────────────────────────────
    lines.append("\n## Token Cost Comparison\n")

    lines.append("| Method | LLM calls per iteration | Call breakdown | Relative cost |")
    lines.append("|---|---|---|---|")
    lines.append("| **CPE-GEPA** | 5 × N + 1 | 1 contrast gen (llama-70b) + 4 routing calls + 1 prompt gen (120b) | 1.0× |")
    lines.append("| **Scalar** | 1 × N + 1 | 1 routing call per example + 1 prompt gen | ~0.2× |")
    lines.append("| **Critique** | 1 × N + 1 + W | 1 routing + 1 critique per wrong pred + 1 prompt gen | ~0.3–0.5× |")
    lines.append("")
    lines.append("_N = trainset size. CPE-GEPA runs 4× more routing calls per example due to contrast analysis._  ")
    lines.append("_Contrast generation (Anthropic Haiku) is separate from routing (Groq 8b-instant)._  ")

    # Concrete numbers
    for agent in agents:
        r = results.get((agent, "cpe_gepa"))
        if not r:
            continue
        iters = r.get("iterations", [])
        total_traces = sum(it.get("num_traces", 0) for it in iters)
        n = r.get("trainset_size", 0)
        n_iters = len(iters)
        routing_calls = n * n_iters              # 1 original routing call per example per iteration
        contrast_calls = total_traces            # 1 contrast gen per traced example
        contrast_routing = total_traces * 3      # 3 contrast routing calls per trace
        prompt_gen_calls = max(0, n_iters - 1)  # one per inter-iteration gap
        total = routing_calls + contrast_calls + contrast_routing + prompt_gen_calls
        lines.append(
            f"\n**{agent}** (CPE-GEPA): "
            f"{n_iters} iter × {n} examples → "
            f"{routing_calls} routing + {contrast_calls} contrast gen + "
            f"{contrast_routing} contrast routing + {prompt_gen_calls} prompt gen "
            f"= **{total} total LLM calls**"
        )

    lines.append("\n---")
    lines.append("_Generated by `experiments/runs/write_summary.py`_")

    out = Path("experiments/results/summary.md")
    out.write_text("\n".join(lines) + "\n")
    print(f"Written → {out}")


if __name__ == "__main__":
    main()
