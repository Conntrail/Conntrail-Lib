# Conntrail × GEPA — Experiment Results

**Runs:** CPE-GEPA + scalar + critique baselines  
**Model:** Groq llama-3.3-70b-versatile (routing + contrast gen), openai/gpt-oss-120b (prompt gen)  
**Config:** sample_rate=1.0, async_mode=False, entropy_alert_threshold=0.0  

## Results Table

| Agent | Method | Final Score | Best Score | Converged @ | Mean Entropy (iter 1→last) |
|---|---|---|---|---|---|
| customer_support | cpe_gepa | 100% | 100% | iter 1 | 0.19 → 0.20 |
| customer_support | scalar | 83% | 100% | iter 1 | n/a (no CPE) |
| customer_support | critique | 100% | 100% | iter 1 | n/a (no CPE) |
| react_agent | cpe_gepa | 25% | 25% | iter never | 0.39 → 0.30 |
| react_agent | scalar | 25% | 25% | iter never | n/a (no CPE) |
| react_agent | critique | 25% | 25% | iter never | n/a (no CPE) |
| multi_agent_supervisor | cpe_gepa | 75% | 100% | iter 2 | 0.10 → 0.10 |
| multi_agent_supervisor | scalar | 25% | 75% | iter never | n/a (no CPE) |
| multi_agent_supervisor | critique | 75% | 100% | iter 2 | n/a (no CPE) |
| adaptive_rag | cpe_gepa | 75% | 75% | iter never | 0.10 → 0.23 |
| adaptive_rag | scalar | 25% | 75% | iter never | n/a (no CPE) |
| adaptive_rag | critique | 25% | 75% | iter never | n/a (no CPE) |

## Per-Agent Detail

### Customer Support

**CPE-GEPA** — 10 iterations, trainset size 6
- Score trajectory: 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100%
- Entropy trajectory: 0.19 → 0.19 → 0.26 → 0.26 → 0.26 → 0.33 → 0.19 → 0.14 → 0.19 → 0.20
- Fragile counts: 1 → 1 → 1 → 1 → 1 → 1 → 1 → 0 → 1 → 0
- Boundary counts: 1 → 1 → 2 → 2 → 2 → 3 → 1 → 2 → 1 → 3
- Dominant attribution: none detected → none detected → none detected → none detected → none detected → semantic intensity → none detected → none detected → none detected → none detected

**Scalar baseline** — 10 iterations
- Score trajectory: 100% → 100% → 100% → 83% → 83% → 83% → 83% → 83% → 83% → 83%

**Critique baseline** — 10 iterations
- Score trajectory: 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100%

### React Agent

**CPE-GEPA** — 10 iterations, trainset size 4
- Score trajectory: 25% → 25% → 25% → 25% → 25% → 25% → 25% → 25% → 25% → 25%
- Entropy trajectory: 0.39 → 0.32 → 0.33 → 0.32 → 0.32 → 0.32 → 0.39 → 0.37 → 0.37 → 0.30
- Fragile counts: 0 → 0 → 0 → 0 → 0 → 0 → 0 → 0 → 0 → 0
- Boundary counts: 6 → 5 → 5 → 5 → 5 → 5 → 6 → 5 → 5 → 4
- Dominant attribution: semantic intensity → semantic intensity → none detected → semantic intensity → semantic intensity → semantic intensity → semantic intensity → semantic intensity → semantic intensity → none detected

**Scalar baseline** — 10 iterations
- Score trajectory: 25% → 25% → 25% → 25% → 25% → 25% → 25% → 25% → 25% → 25%

**Critique baseline** — 10 iterations
- Score trajectory: 25% → 25% → 25% → 25% → 25% → 25% → 25% → 25% → 25% → 25%

### Multi Agent Supervisor

**CPE-GEPA** — 10 iterations, trainset size 4
- Score trajectory: 75% → 100% → 100% → 100% → 100% → 75% → 100% → 75% → 100% → 75%
- Entropy trajectory: 0.10 → 0.12 → 0.23 → 0.23 → 0.12 → 0.10 → 0.12 → 0.29 → 0.10 → 0.10
- Fragile counts: 0 → 0 → 0 → 0 → 0 → 0 → 0 → 1 → 0 → 0
- Boundary counts: 1 → 1 → 2 → 2 → 1 → 1 → 1 → 1 → 1 → 1
- Dominant attribution: none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected

**Scalar baseline** — 10 iterations
- Score trajectory: 75% → 75% → 75% → 75% → 75% → 75% → 75% → 75% → 75% → 25%

**Critique baseline** — 10 iterations
- Score trajectory: 75% → 100% → 75% → 75% → 100% → 100% → 75% → 75% → 75% → 75%

### Adaptive Rag

**CPE-GEPA** — 10 iterations, trainset size 4
- Score trajectory: 50% → 50% → 50% → 25% → 50% → 75% → 75% → 75% → 75% → 75%
- Entropy trajectory: 0.10 → 0.20 → 0.45 → 0.12 → 0.20 → 0.23 → 0.10 → 0.20 → 0.23 → 0.23
- Fragile counts: 0 → 0 → 0 → 0 → 0 → 0 → 0 → 0 → 0 → 0
- Boundary counts: 1 → 2 → 4 → 1 → 2 → 2 → 1 → 2 → 2 → 2
- Dominant attribution: none detected → semantic intensity → semantic intensity → none detected → semantic intensity → none detected → none detected → none detected → none detected → none detected

**Scalar baseline** — 10 iterations
- Score trajectory: 50% → 75% → 50% → 50% → 50% → 25% → 25% → 25% → 50% → 25%

**Critique baseline** — 10 iterations
- Score trajectory: 50% → 25% → 25% → 75% → 75% → 75% → 75% → 75% → 75% → 25%

## Notable Findings

- **Largest CPE-GEPA improvement:** `multi_agent_supervisor` gained 25% accuracy (from first to peak iteration).
- **Entropy reduction confirmed** for `react_agent`: 0.388 → 0.302 across 10 traced iterations, indicating CPE feedback drove the prompt toward more stable routing decisions.
- **CPE-GEPA outperformed scalar** on `customer_support` by 17% at final iteration (100% vs 83%).
- **CPE-GEPA outperformed scalar** on `multi_agent_supervisor` by 50% at final iteration (75% vs 25%).
- **CPE-GEPA outperformed scalar** on `adaptive_rag` by 50% at final iteration (75% vs 25%).

## Token Cost Comparison

| Method | LLM calls per iteration | Call breakdown | Relative cost |
|---|---|---|---|
| **CPE-GEPA** | 5 × N + 1 | 1 contrast gen (llama-70b) + 4 routing calls + 1 prompt gen (120b) | 1.0× |
| **Scalar** | 1 × N + 1 | 1 routing call per example + 1 prompt gen | ~0.2× |
| **Critique** | 1 × N + 1 + W | 1 routing + 1 critique per wrong pred + 1 prompt gen | ~0.3–0.5× |

_N = trainset size. CPE-GEPA runs 4× more routing calls per example due to contrast analysis._  
_Contrast generation (Anthropic Haiku) is separate from routing (Groq 8b-instant)._  

**customer_support** (CPE-GEPA): 10 iter × 6 examples → 60 routing + 60 contrast gen + 180 contrast routing + 9 prompt gen = **309 total LLM calls**

**react_agent** (CPE-GEPA): 10 iter × 4 examples → 40 routing + 67 contrast gen + 201 contrast routing + 9 prompt gen = **317 total LLM calls**

**multi_agent_supervisor** (CPE-GEPA): 10 iter × 4 examples → 40 routing + 40 contrast gen + 120 contrast routing + 9 prompt gen = **209 total LLM calls**

**adaptive_rag** (CPE-GEPA): 10 iter × 4 examples → 40 routing + 40 contrast gen + 120 contrast routing + 9 prompt gen = **209 total LLM calls**

---
_Generated by `experiments/runs/write_summary.py`_
