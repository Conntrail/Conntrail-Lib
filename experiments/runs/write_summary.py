"""
Read cpe_gepa.json and baselines.json and write summary.md + extended_summary.md.
Run after both experiment scripts have completed.

Usage:
    # Single-run summary (defaults to run_1):
    python -m experiments.runs.write_summary --dir experiments/results/run_1

    # Cross-run comparison (CPE-GEPA from run_1+run_2, baselines from run_1):
    python -m experiments.runs.write_summary --cross-run
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def score_trajectory(iterations: list[dict]) -> str:
    return " → ".join(f"{it.get('accuracy', 0):.0%}" for it in iterations)


def entropy_trajectory(iterations: list[dict]) -> str:
    vals = [it.get("mean_entropy") for it in iterations]
    return " → ".join("N/A" if v is None else f"{v:.2f}" for v in vals)


ALL_AGENTS = [
    "customer_support", "react_agent", "multi_agent_supervisor", "adaptive_rag",
    "document_triage", "code_review", "medical_triage", "financial_query",
]

ROUTING_GROUPS = {
    "Intent classification": ["customer_support", "financial_query", "medical_triage"],
    "Document/content classification": ["document_triage", "adaptive_rag"],
    "Tool/resource selection": ["react_agent", "code_review"],
    "Agent dispatch": ["multi_agent_supervisor"],
}

METHODS = ["cpe_gepa", "scalar", "critique"]


# ── Single-run summary ────────────────────────────────────────────────────────

def main(results_dir: str = "experiments/results/run_1") -> None:
    d = Path(results_dir)
    cpe_data = load(str(d / "cpe_gepa.json"))
    baseline_data = load(str(d / "baselines.json"))

    results: dict[tuple, dict] = {}
    for r in cpe_data:
        if "error" not in r:
            results[(r["agent"], "cpe_gepa")] = r
    for r in baseline_data:
        if "error" not in r:
            results[(r["agent"], r["method"])] = r

    agents = [a for a in ALL_AGENTS if any((a, m) in results for m in METHODS)]

    lines: list[str] = []
    lines.append("# Conntrail × GEPA — Experiment Results\n")
    lines.append(f"**Source:** `{results_dir}`  ")
    lines.append("**Models:** Haiku (router + contrast) · Sonnet 4.6 (prompt optimizer)  ")
    lines.append("**Config:** sample_rate=1.0, async_mode=False, entropy_alert_threshold=0.0  \n")

    lines.append("## Results Table\n")
    lines.append("| Agent | Method | Runs | Final Score (mean±std) | Best Score | Converged @ | Mean Entropy (iter 1→last) |")
    lines.append("|---|---|---|---|---|---|---|")

    for agent in agents:
        for method in METHODS:
            r = results.get((agent, method))
            if r is None:
                lines.append(f"| {agent} | {method} | — | — | — | — | — |")
                continue
            iters = r.get("iterations", [])
            mean_f = r.get("final_score_mean", r.get("final_score"))
            std_f = r.get("final_score_std")
            best = r.get("best_score_mean", max((it.get("accuracy", 0) for it in iters), default=None))
            runs_n = r.get("num_runs", 1)
            conv = convergence_iter(iters)
            score_str = fmt(mean_f, ".0%") if mean_f is not None else "—"
            if std_f is not None:
                score_str += f" ±{std_f:.2f}"
            if method == "cpe_gepa":
                e_start = fmt(iters[0].get("mean_entropy") if iters else None, ".2f")
                e_end = fmt(iters[-1].get("mean_entropy") if iters else None, ".2f")
                entropy_col = f"{e_start} → {e_end}"
            else:
                entropy_col = "n/a"
            lines.append(
                f"| {agent} | {method} | {runs_n} | {score_str} | {fmt(best, '.0%')} "
                f"| iter {conv} | {entropy_col} |"
            )

    lines.append("\n## Per-Agent Detail\n")
    for agent in agents:
        lines.append(f"### {agent.replace('_', ' ').title()}\n")
        for method in METHODS:
            r = results.get((agent, method))
            if not r:
                continue
            iters = r.get("iterations", [])
            label = {"cpe_gepa": "CPE-GEPA", "scalar": "Scalar baseline", "critique": "Critique baseline"}[method]
            lines.append(f"**{label}** — {len(iters)} iterations, trainset={r.get('trainset_size', '?')}, runs={r.get('num_runs', 1)}")
            lines.append(f"- Score trajectory: {score_trajectory(iters)}")
            if method == "cpe_gepa":
                lines.append(f"- Entropy trajectory: {entropy_trajectory(iters)}")
                frag = [it.get("fragile_count", 0) for it in iters]
                attr = [it.get("dominant_attribution") or "—" for it in iters]
                lines.append(f"- Fragile counts: {' → '.join(map(str, frag))}")
                lines.append(f"- Dominant attribution: {' → '.join(attr)}")
            lines.append("")

    out = d / "summary.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"Written → {out}")


# ── Cross-run summary (run_1 + run_2 CPE-GEPA, run_1 baselines) ──────────────

def cross_run_main(
    run1_dir: str = "experiments/results/run_1",
    run2_dir: str = "experiments/results/run_2",
    out_dir: str = "experiments/results",
) -> None:
    r1d = Path(run1_dir)
    r2d = Path(run2_dir)

    cpe1 = {r["agent"]: r for r in load(str(r1d / "cpe_gepa.json")) if "error" not in r}
    cpe2 = {r["agent"]: r for r in load(str(r2d / "cpe_gepa.json")) if "error" not in r}
    baselines = {}
    for r in load(str(r1d / "baselines.json")):
        if "error" not in r:
            baselines[(r["agent"], r["method"])] = r

    lines: list[str] = []
    lines.append("# Conntrail × GEPA — Cross-Run Comparison\n")
    lines.append("**CPE-GEPA:** run_1 + run_2 (variance across two independent seeds)  ")
    lines.append("**Baselines:** run_1 only (complete set)  ")
    lines.append("**Models:** Haiku (router + contrast) · Sonnet 4.6 (prompt gen, temp=0.3)  \n")

    # ── Section 1: CPE-GEPA reproducibility table ──────────────────────────────
    lines.append("## Section 1 — CPE-GEPA Reproducibility (run_1 vs run_2)\n")
    lines.append("| Agent | Run 1 final | Run 2 final | Mean | Δ (abs) | Best run_1 | Best run_2 |")
    lines.append("|---|---|---|---|---|---|---|")

    cpe_combined: dict[str, dict] = {}
    for agent in ALL_AGENTS:
        r1 = cpe1.get(agent)
        r2 = cpe2.get(agent)
        f1 = r1.get("final_score_mean", r1.get("final_score")) if r1 else None
        f2 = r2.get("final_score_mean", r2.get("final_score")) if r2 else None
        b1 = r1.get("best_score_mean") if r1 else None
        b2 = r2.get("best_score_mean") if r2 else None

        both = [v for v in [f1, f2] if v is not None]
        mean_f = statistics.mean(both) if both else None
        delta = abs(f1 - f2) if (f1 is not None and f2 is not None) else None

        cpe_combined[agent] = {
            "run1_final": f1, "run2_final": f2, "mean_final": mean_f,
            "run1_best": b1, "run2_best": b2,
            "delta": delta,
            "iterations_r1": r1.get("iterations", []) if r1 else [],
        }

        lines.append(
            f"| {agent} | {fmt(f1, '.0%')} | {fmt(f2, '.0%')} | {fmt(mean_f, '.0%')} "
            f"| {fmt(delta, '.0%') if delta is not None else '—'} "
            f"| {fmt(b1, '.0%')} | {fmt(b2, '.0%')} |"
        )

    # ── Section 2: CPE-GEPA vs Baselines ─────────────────────────────────────
    lines.append("\n## Section 2 — CPE-GEPA vs Baselines\n")
    lines.append("CPE-GEPA mean across both runs vs scalar and critique baselines (run_1).\n")
    lines.append("| Agent | CPE-GEPA (mean) | Scalar | Critique | CPE vs Scalar | CPE vs Critique |")
    lines.append("|---|---|---|---|---|---|")

    for agent in ALL_AGENTS:
        cpe_mean = cpe_combined[agent]["mean_final"]
        r_sc = baselines.get((agent, "scalar"))
        r_cr = baselines.get((agent, "critique"))
        sc = r_sc.get("final_score_mean", r_sc.get("final_score")) if r_sc else None
        cr = r_cr.get("final_score_mean", r_cr.get("final_score")) if r_cr else None

        def lift(a, b):
            if a is None or b is None:
                return "—"
            v = a - b
            return (f"+{v:.0%}" if v >= 0 else f"{v:.0%}")

        lines.append(
            f"| {agent} | {fmt(cpe_mean, '.0%')} | {fmt(sc, '.0%')} | {fmt(cr, '.0%')} "
            f"| {lift(cpe_mean, sc)} | {lift(cpe_mean, cr)} |"
        )

    # ── Section 3: Routing group analysis ─────────────────────────────────────
    lines.append("\n## Section 3 — Routing Group Analysis\n")
    lines.append("| Routing Group | Agents | CPE-GEPA mean | Scalar mean | CPE lift |")
    lines.append("|---|---|---|---|---|")

    for group_name, group_agents in ROUTING_GROUPS.items():
        cpe_scores, scalar_scores = [], []
        for agent in group_agents:
            v = cpe_combined[agent]["mean_final"]
            if v is not None:
                cpe_scores.append(v)
            r_sc = baselines.get((agent, "scalar"))
            if r_sc:
                sv = r_sc.get("final_score_mean", r_sc.get("final_score"))
                if sv is not None:
                    scalar_scores.append(sv)

        if cpe_scores and scalar_scores:
            mean_cpe = statistics.mean(cpe_scores)
            mean_sc = statistics.mean(scalar_scores)
            lift_v = mean_cpe - mean_sc
            lift_str = f"+{lift_v:.0%}" if lift_v >= 0 else f"{lift_v:.0%}"
        else:
            mean_cpe = mean_sc = None
            lift_str = "—"

        lines.append(
            f"| {group_name} | {', '.join(group_agents)} "
            f"| {fmt(mean_cpe, '.0%')} | {fmt(mean_sc, '.0%')} | {lift_str} |"
        )

    # ── Section 4: Attribution dimension frequency (run_1) ────────────────────
    lines.append("\n## Section 4 — Attribution Dimension Frequency (run_1 CPE-GEPA)\n")
    lines.append("| Agent | semantic intensity | urgency/sentiment | surface form | none/unknown |")
    lines.append("|---|---|---|---|---|")

    global_counter: Counter = Counter()
    for agent in ALL_AGENTS:
        iters = cpe_combined[agent]["iterations_r1"]
        counter: Counter = Counter()
        for it in iters:
            attr = it.get("dominant_attribution") or "none/unknown"
            counter[attr] += 1
            global_counter[attr] += 1
        lines.append(
            f"| {agent} "
            f"| {counter.get('semantic intensity', 0)} "
            f"| {counter.get('urgency/sentiment', 0)} "
            f"| {counter.get('surface form', 0)} "
            f"| {counter.get('none/unknown', 0)} |"
        )
    lines.append(
        f"| **TOTAL** "
        f"| **{global_counter.get('semantic intensity', 0)}** "
        f"| **{global_counter.get('urgency/sentiment', 0)}** "
        f"| **{global_counter.get('surface form', 0)}** "
        f"| **{global_counter.get('none/unknown', 0)}** |"
    )

    dominant = global_counter.most_common(1)
    if dominant:
        dim, count = dominant[0]
        total_attrs = sum(global_counter.values())
        pct = count / total_attrs if total_attrs else 0
        lines.append(f"\n**Finding**: `{dim}` is the dominant attribution dimension ({count}/{total_attrs} iterations, {pct:.0%}).")

    # ── Section 5: Token cost at scale ────────────────────────────────────────
    lines.append("\n## Section 5 — Token Cost at Scale\n")
    lines.append("Extrapolated to 50 iterations × 100 examples (realistic production run).\n")
    lines.append("Pricing (June 2025): Haiku input $0.80/MTok · Haiku output $4.00/MTok · Sonnet input $3.00/MTok · Sonnet output $15.00/MTok\n")

    haiku_in, haiku_out = 0.80 / 1e6, 4.00 / 1e6
    sonnet_in, sonnet_out = 3.00 / 1e6, 15.00 / 1e6
    N_EXAMPLES, N_ITER = 100, 50

    cpe_haiku_calls = N_EXAMPLES * N_ITER * 4  # 1 router + 3 contrast
    cpe_sonnet_calls = N_ITER - 1
    cpe_cost = (
        cpe_haiku_calls * 350 * haiku_in + cpe_haiku_calls * 20 * haiku_out +
        cpe_sonnet_calls * 900 * sonnet_in + cpe_sonnet_calls * 350 * sonnet_out
    )

    sc_haiku_calls = N_EXAMPLES * N_ITER
    sc_sonnet_calls = N_ITER - 1
    sc_cost = (
        sc_haiku_calls * 350 * haiku_in + sc_haiku_calls * 10 * haiku_out +
        sc_sonnet_calls * 700 * sonnet_in + sc_sonnet_calls * 300 * sonnet_out
    )

    cr_cost = sc_cost * 1.15  # critique adds ~15% for wrong-prediction context

    lines.append("| Method | Haiku calls | Sonnet calls | Est. cost | vs scalar |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| CPE-GEPA | {cpe_haiku_calls:,} | {cpe_sonnet_calls} | ${cpe_cost:.2f} | {cpe_cost/sc_cost:.1f}× |")
    lines.append(f"| Scalar   | {sc_haiku_calls:,} | {sc_sonnet_calls} | ${sc_cost:.2f} | 1.0× |")
    lines.append(f"| Critique | {sc_haiku_calls:,} | {sc_sonnet_calls} | ${cr_cost:.2f} | {cr_cost/sc_cost:.1f}× |")

    lines.append(
        f"\nCPE-GEPA costs ~{cpe_cost/sc_cost:.1f}× more than scalar at scale, "
        f"driven by the 4× contrast routing overhead. The premium provides entropy-aware "
        f"feedback unavailable to scalar or critique baselines."
    )

    lines.append("\n---")
    lines.append("_Generated by `experiments/runs/write_summary.py --cross-run`_")

    out = Path(out_dir) / "cross_run_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"Written → {out}")


# ── Extended summary (run_1 only, full detail) ────────────────────────────────

def extended_main(results_dir: str = "experiments/results/run_1") -> None:
    d = Path(results_dir)
    cpe_data = load(str(d / "cpe_gepa.json"))
    baseline_data = load(str(d / "baselines.json"))

    results: dict[tuple, dict] = {}
    for r in cpe_data:
        if "error" not in r:
            results[(r["agent"], "cpe_gepa")] = r
    for r in baseline_data:
        if "error" not in r:
            results[(r["agent"], r["method"])] = r

    agents = [a for a in ALL_AGENTS if any((a, m) in results for m in METHODS)]

    lines: list[str] = []
    lines.append("# Conntrail × GEPA — Extended Experiment Results\n")

    cfg = {}
    for m in METHODS:
        for a in agents:
            r = results.get((a, m))
            if r and "experiment_config" in r:
                cfg = r["experiment_config"]
                break
        if cfg:
            break

    if cfg:
        lines.append("**Experiment configuration:**\n")
        lines.append("```json")
        lines.append(json.dumps(cfg, indent=2))
        lines.append("```\n")

    lines.append("## Section 1 — Extended Results Table\n")
    lines.append("⚠ = std > 0.10 (high variance across runs)\n")
    lines.append("| Agent | Method | Runs | Final (mean±std) | Best (mean) | Converged @ |")
    lines.append("|---|---|---|---|---|---|")

    for agent in ALL_AGENTS:
        for method in METHODS:
            r = results.get((agent, method))
            if r is None:
                lines.append(f"| {agent} | {method} | — | — | — | — |")
                continue
            iters = r.get("iterations", [])
            mean_f = r.get("final_score_mean", r.get("final_score"))
            std_f = r.get("final_score_std", 0.0)
            best = r.get("best_score_mean")
            runs_n = r.get("num_runs", 1)
            conv = convergence_iter(iters)
            warn = " ⚠" if (std_f or 0) > 0.10 else ""
            score_str = (fmt(mean_f, ".0%") + f" ±{std_f:.2f}" + warn) if mean_f is not None else "—"
            lines.append(
                f"| {agent} | {method} | {runs_n} | {score_str} | {fmt(best, '.0%')} | iter {conv} |"
            )

    lines.append("\n## Section 2 — Routing Pattern Analysis\n")
    lines.append("| Routing Group | Agents | CPE-GEPA final | Scalar final | CPE lift |")
    lines.append("|---|---|---|---|---|")

    for group_name, group_agents in ROUTING_GROUPS.items():
        cpe_scores, scalar_scores = [], []
        for agent in group_agents:
            for m, lst in [("cpe_gepa", cpe_scores), ("scalar", scalar_scores)]:
                r = results.get((agent, m))
                if r:
                    v = r.get("final_score_mean", r.get("final_score"))
                    if v is not None:
                        lst.append(v)
        if cpe_scores and scalar_scores:
            mean_cpe = statistics.mean(cpe_scores)
            mean_sc = statistics.mean(scalar_scores)
            lift = mean_cpe - mean_sc
            lift_str = f"+{lift:.0%}" if lift >= 0 else f"{lift:.0%}"
        else:
            mean_cpe = mean_sc = None
            lift_str = "—"
        lines.append(
            f"| {group_name} | {', '.join(group_agents)} "
            f"| {fmt(mean_cpe, '.0%')} | {fmt(mean_sc, '.0%')} | {lift_str} |"
        )

    lines.append("\n## Section 3 — Cloud vs Local Model Comparison\n")
    lines.append(
        "_Not run. Project uses Anthropic models exclusively (Haiku routing/contrast, "
        "Sonnet 4.6 prompt generation). Local model comparison is out of scope._"
    )

    lines.append("\n## Section 4 — Attribution Dimension Frequency\n")
    lines.append("| Agent | semantic intensity | urgency/sentiment | surface form | none/unknown |")
    lines.append("|---|---|---|---|---|")

    global_counter: Counter = Counter()
    for agent in ALL_AGENTS:
        r = results.get((agent, "cpe_gepa"))
        if not r:
            lines.append(f"| {agent} | — | — | — | — |")
            continue
        counter: Counter = Counter()
        for it in r.get("iterations", []):
            attr = it.get("dominant_attribution") or "none/unknown"
            counter[attr] += 1
            global_counter[attr] += 1
        lines.append(
            f"| {agent} "
            f"| {counter.get('semantic intensity', 0)} "
            f"| {counter.get('urgency/sentiment', 0)} "
            f"| {counter.get('surface form', 0)} "
            f"| {counter.get('none/unknown', 0)} |"
        )
    lines.append(
        f"| **TOTAL** "
        f"| **{global_counter.get('semantic intensity', 0)}** "
        f"| **{global_counter.get('urgency/sentiment', 0)}** "
        f"| **{global_counter.get('surface form', 0)}** "
        f"| **{global_counter.get('none/unknown', 0)}** |"
    )

    dominant = global_counter.most_common(1)
    if dominant:
        dim, count = dominant[0]
        total_attrs = sum(global_counter.values())
        pct = count / total_attrs if total_attrs else 0
        lines.append(f"\n**Finding**: `{dim}` is the dominant attribution dimension ({pct:.0%} of all iterations).")
        if pct > 0.5:
            lines.append("This dominance across all agent types suggests it is a general property of LLM routing.")

    lines.append("\n## Section 5 — Token Cost at Scale\n")
    lines.append("Extrapolated to 50 iterations × 100 examples.\n")
    lines.append("Pricing (June 2025): Haiku input $0.80/MTok · Haiku output $4.00/MTok · Sonnet input $3.00/MTok · Sonnet output $15.00/MTok\n")

    haiku_in, haiku_out = 0.80 / 1e6, 4.00 / 1e6
    sonnet_in, sonnet_out = 3.00 / 1e6, 15.00 / 1e6
    N_EXAMPLES, N_ITER = 100, 50

    cpe_haiku_calls = N_EXAMPLES * N_ITER * 4
    cpe_cost = (
        cpe_haiku_calls * 350 * haiku_in + cpe_haiku_calls * 20 * haiku_out +
        (N_ITER - 1) * 900 * sonnet_in + (N_ITER - 1) * 350 * sonnet_out
    )
    sc_haiku_calls = N_EXAMPLES * N_ITER
    sc_cost = (
        sc_haiku_calls * 350 * haiku_in + sc_haiku_calls * 10 * haiku_out +
        (N_ITER - 1) * 700 * sonnet_in + (N_ITER - 1) * 300 * sonnet_out
    )
    cr_cost = sc_cost * 1.15

    lines.append("| Method | Haiku calls | Sonnet calls | Est. cost | vs scalar |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| CPE-GEPA | {cpe_haiku_calls:,} | {N_ITER - 1} | ${cpe_cost:.2f} | {cpe_cost/sc_cost:.1f}× |")
    lines.append(f"| Scalar   | {sc_haiku_calls:,} | {N_ITER - 1} | ${sc_cost:.2f} | 1.0× |")
    lines.append(f"| Critique | {sc_haiku_calls:,} | {N_ITER - 1} | ${cr_cost:.2f} | {cr_cost/sc_cost:.1f}× |")
    lines.append(f"\nCPE-GEPA costs ~{cpe_cost/sc_cost:.1f}× more than scalar at scale.")

    lines.append("\n---")
    lines.append("_Generated by `experiments/runs/write_summary.py`_")

    out = d / "extended_summary.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"Written → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", default="experiments/results/run_1",
        help="Results directory for single-run summaries (default: experiments/results/run_1)",
    )
    parser.add_argument(
        "--cross-run", action="store_true",
        help="Generate cross-run comparison using run_1 + run_2 CPE-GEPA and run_1 baselines",
    )
    args = parser.parse_args()

    if args.cross_run:
        cross_run_main()
    else:
        main(args.dir)
        extended_main(args.dir)
