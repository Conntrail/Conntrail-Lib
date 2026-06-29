# Conntrail × GEPA — Experiment Results

**Source:** `experiments/results/run_local`  
**Models:** Haiku (router + contrast) · Sonnet 4.6 (prompt optimizer)  
**Config:** sample_rate=1.0, async_mode=False, entropy_alert_threshold=0.0  

## Results Table

| Agent | Method | Runs | Final Score (mean±std) | Best Score | Converged @ | Mean Entropy (iter 1→last) |
|---|---|---|---|---|---|---|
| customer_support | cpe_gepa | 3 | 75% ±0.05 | 85% | iter 1 | 0.40 → 0.28 |
| customer_support | scalar | 3 | 80% ±0.05 | 85% | iter 1 | n/a |
| customer_support | critique | 3 | 100% ±0.00 | 100% | iter 1 | n/a |
| multi_agent_supervisor | cpe_gepa | 3 | 87% ±0.06 | 88% | iter 2 | 0.08 → 0.04 |
| multi_agent_supervisor | scalar | 3 | 88% ±0.06 | 88% | iter 2 | n/a |
| multi_agent_supervisor | critique | 3 | 100% ±0.00 | 100% | iter 2 | n/a |
| adaptive_rag | cpe_gepa | 3 | 92% ±0.03 | 93% | iter 1 | 0.16 → 0.16 |
| adaptive_rag | scalar | 3 | 92% ±0.03 | 92% | iter 1 | n/a |
| adaptive_rag | critique | 3 | 100% ±0.00 | 100% | iter 1 | n/a |
| code_review | cpe_gepa | 3 | 100% ±0.00 | 100% | iter 1 | 0.00 → 0.00 |
| code_review | scalar | 3 | 100% ±0.00 | 100% | iter 1 | n/a |
| code_review | critique | 3 | 100% ±0.00 | 100% | iter 1 | n/a |
| medical_triage | cpe_gepa | 3 | 90% ±0.05 | 93% | iter 1 | 0.32 → 0.32 |
| medical_triage | scalar | 3 | 89% ±0.06 | 96% | iter 1 | n/a |
| medical_triage | critique | 3 | 100% ±0.00 | 100% | iter 1 | n/a |
| financial_query | cpe_gepa | 3 | 95% ±0.00 | 95% | iter 1 | 0.19 → 0.16 |
| financial_query | scalar | 3 | 95% ±0.00 | 95% | iter 1 | n/a |
| financial_query | critique | 3 | 100% ±0.00 | 100% | iter 1 | n/a |

## Per-Agent Detail

### Customer Support

**CPE-GEPA** — 10 iterations, trainset=20, runs=3
- Score trajectory: 85% → 80% → 85% → 85% → 80% → 80% → 80% → 80% → 80% → 80%
- Entropy trajectory: 0.40 → 0.25 → 0.23 → 0.27 → 0.26 → 0.30 → 0.26 → 0.28 → 0.28 → 0.28
- Fragile counts: 1 → 0 → 0 → 0 → 0 → 0 → 0 → 0 → 0 → 0
- Dominant attribution: semantic intensity → semantic intensity → semantic intensity → semantic intensity → semantic intensity → semantic intensity → semantic intensity → semantic intensity → semantic intensity → semantic intensity

**Scalar baseline** — 10 iterations, trainset=20, runs=3
- Score trajectory: 85% → 75% → 70% → 65% → 65% → 70% → 70% → 80% → 80% → 80%

**Critique baseline** — 10 iterations, trainset=20, runs=3
- Score trajectory: 85% → 70% → 95% → 100% → 100% → 100% → 100% → 100% → 100% → 100%

### Multi Agent Supervisor

**CPE-GEPA** — 10 iterations, trainset=20, runs=3
- Score trajectory: 80% → 95% → 90% → 90% → 90% → 90% → 90% → 90% → 90% → 90%
- Entropy trajectory: 0.08 → 0.06 → 0.04 → 0.04 → 0.04 → 0.04 → 0.04 → 0.04 → 0.04 → 0.04
- Fragile counts: 0 → 1 → 0 → 0 → 0 → 0 → 0 → 0 → 0 → 0
- Dominant attribution: none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected

**Scalar baseline** — 10 iterations, trainset=20, runs=3
- Score trajectory: 80% → 95% → 95% → 95% → 95% → 95% → 95% → 95% → 95% → 95%

**Critique baseline** — 10 iterations, trainset=20, runs=3
- Score trajectory: 80% → 95% → 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100%

### Adaptive Rag

**CPE-GEPA** — 10 iterations, trainset=20, runs=3
- Score trajectory: 90% → 90% → 90% → 90% → 90% → 90% → 90% → 90% → 90% → 90%
- Entropy trajectory: 0.16 → 0.16 → 0.16 → 0.16 → 0.16 → 0.16 → 0.16 → 0.16 → 0.16 → 0.16
- Fragile counts: 0 → 0 → 0 → 0 → 0 → 0 → 0 → 0 → 0 → 0
- Dominant attribution: none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected

**Scalar baseline** — 10 iterations, trainset=20, runs=3
- Score trajectory: 90% → 90% → 90% → 90% → 90% → 90% → 90% → 90% → 90% → 90%

**Critique baseline** — 10 iterations, trainset=20, runs=3
- Score trajectory: 90% → 95% → 95% → 95% → 100% → 100% → 100% → 100% → 100% → 100%

### Code Review

**CPE-GEPA** — 10 iterations, trainset=20, runs=3
- Score trajectory: 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100%
- Entropy trajectory: 0.00 → 0.00 → 0.00 → 0.00 → 0.00 → 0.00 → 0.00 → 0.00 → 0.00 → 0.00
- Fragile counts: 0 → 0 → 0 → 0 → 0 → 0 → 0 → 0 → 0 → 0
- Dominant attribution: none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected

**Scalar baseline** — 10 iterations, trainset=20, runs=3
- Score trajectory: 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100%

**Critique baseline** — 10 iterations, trainset=20, runs=3
- Score trajectory: 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100%

### Medical Triage

**CPE-GEPA** — 10 iterations, trainset=24, runs=3
- Score trajectory: 92% → 92% → 88% → 88% → 88% → 88% → 88% → 88% → 88% → 88%
- Entropy trajectory: 0.32 → 0.34 → 0.34 → 0.32 → 0.32 → 0.32 → 0.32 → 0.32 → 0.32 → 0.32
- Fragile counts: 1 → 1 → 1 → 1 → 1 → 1 → 1 → 1 → 1 → 1
- Dominant attribution: semantic intensity → semantic intensity → semantic intensity → semantic intensity → semantic intensity → semantic intensity → semantic intensity → semantic intensity → semantic intensity → semantic intensity

**Scalar baseline** — 10 iterations, trainset=24, runs=3
- Score trajectory: 92% → 96% → 96% → 96% → 96% → 92% → 88% → 88% → 88% → 83%

**Critique baseline** — 10 iterations, trainset=24, runs=3
- Score trajectory: 92% → 96% → 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100%

### Financial Query

**CPE-GEPA** — 10 iterations, trainset=20, runs=3
- Score trajectory: 95% → 95% → 95% → 95% → 95% → 95% → 95% → 95% → 95% → 95%
- Entropy trajectory: 0.19 → 0.16 → 0.16 → 0.16 → 0.16 → 0.16 → 0.16 → 0.16 → 0.16 → 0.16
- Fragile counts: 0 → 0 → 0 → 0 → 0 → 0 → 0 → 0 → 0 → 0
- Dominant attribution: none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected

**Scalar baseline** — 10 iterations, trainset=20, runs=3
- Score trajectory: 95% → 95% → 95% → 95% → 95% → 95% → 95% → 95% → 95% → 95%

**Critique baseline** — 10 iterations, trainset=20, runs=3
- Score trajectory: 95% → 95% → 95% → 95% → 100% → 100% → 100% → 100% → 100% → 100%

