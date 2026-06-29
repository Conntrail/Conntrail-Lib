# Conntrail × GEPA — Experiment Results

**Source:** `experiments/results/run_1`  
**Models:** Haiku (router + contrast) · Sonnet 4.6 (prompt optimizer)  
**Config:** sample_rate=1.0, async_mode=False, entropy_alert_threshold=0.0  

## Results Table

| Agent | Method | Runs | Final Score (mean±std) | Best Score | Converged @ | Mean Entropy (iter 1→last) |
|---|---|---|---|---|---|---|
| customer_support | cpe_gepa | 3 | 95% ±0.00 | 95% | iter 1 | 0.25 → 0.26 |
| customer_support | scalar | 1 | 85% ±0.00 | 95% | iter 1 | n/a |
| customer_support | critique | 1 | 100% ±0.00 | 100% | iter 1 | n/a |
| react_agent | cpe_gepa | 1 | 45% ±0.00 | 45% | iter never | 0.28 → 0.27 |
| react_agent | scalar | 1 | 45% ±0.00 | 45% | iter never | n/a |
| react_agent | critique | 1 | 45% ±0.00 | 45% | iter never | n/a |
| multi_agent_supervisor | cpe_gepa | 1 | 100% ±0.00 | 100% | iter 1 | 0.14 → 0.16 |
| multi_agent_supervisor | scalar | 1 | 100% ±0.00 | 100% | iter 1 | n/a |
| multi_agent_supervisor | critique | 1 | 100% ±0.00 | 100% | iter 1 | n/a |
| adaptive_rag | cpe_gepa | 1 | 90% ±0.00 | 90% | iter 1 | 0.12 → 0.12 |
| adaptive_rag | scalar | 1 | 90% ±0.00 | 90% | iter 1 | n/a |
| adaptive_rag | critique | 1 | 100% ±0.00 | 100% | iter 1 | n/a |
| document_triage | cpe_gepa | 1 | 100% ±0.00 | 100% | iter 1 | 0.13 → 0.13 |
| document_triage | scalar | 1 | 100% ±0.00 | 100% | iter 1 | n/a |
| document_triage | critique | 1 | 100% ±0.00 | 100% | iter 1 | n/a |
| code_review | cpe_gepa | 1 | 100% ±0.00 | 100% | iter 1 | 0.02 → 0.02 |
| code_review | scalar | 1 | 100% ±0.00 | 100% | iter 1 | n/a |
| code_review | critique | 1 | 100% ±0.00 | 100% | iter 1 | n/a |
| medical_triage | cpe_gepa | 1 | 92% ±0.00 | 92% | iter 1 | 0.25 → 0.28 |
| medical_triage | scalar | 1 | 62% ±0.00 | 96% | iter 1 | n/a |
| medical_triage | critique | 1 | 100% ±0.00 | 100% | iter 1 | n/a |
| financial_query | cpe_gepa | 1 | 95% ±0.00 | 95% | iter 1 | 0.11 → 0.13 |
| financial_query | scalar | 1 | 95% ±0.00 | 95% | iter 1 | n/a |
| financial_query | critique | 1 | 100% ±0.00 | 100% | iter 1 | n/a |

## Per-Agent Detail

### Customer Support

**CPE-GEPA** — 10 iterations, trainset=20, runs=3
- Score trajectory: 95% → 95% → 95% → 95% → 95% → 95% → 95% → 95% → 95% → 95%
- Entropy trajectory: 0.25 → 0.26 → 0.27 → 0.26 → 0.25 → 0.26 → 0.25 → 0.25 → 0.26 → 0.26
- Fragile counts: 1 → 2 → 2 → 2 → 1 → 2 → 1 → 1 → 2 → 1
- Dominant attribution: none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected → semantic intensity

**Scalar baseline** — 10 iterations, trainset=20, runs=1
- Score trajectory: 95% → 80% → 75% → 70% → 75% → 80% → 75% → 80% → 85% → 85%

**Critique baseline** — 10 iterations, trainset=20, runs=1
- Score trajectory: 95% → 75% → 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100%

### React Agent

**CPE-GEPA** — 10 iterations, trainset=20, runs=1
- Score trajectory: 45% → 45% → 45% → 45% → 45% → 45% → 45% → 45% → 45% → 45%
- Entropy trajectory: 0.28 → 0.27 → 0.27 → 0.26 → 0.25 → 0.24 → 0.24 → 0.21 → 0.24 → 0.27
- Fragile counts: 0 → 0 → 0 → 0 → 0 → 0 → 0 → 0 → 0 → 0
- Dominant attribution: semantic intensity → semantic intensity → none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected

**Scalar baseline** — 10 iterations, trainset=20, runs=1
- Score trajectory: 45% → 45% → 45% → 45% → 45% → 45% → 40% → 45% → 45% → 45%

**Critique baseline** — 10 iterations, trainset=20, runs=1
- Score trajectory: 45% → 45% → 45% → 45% → 45% → 45% → 45% → 45% → 45% → 45%

### Multi Agent Supervisor

**CPE-GEPA** — 10 iterations, trainset=20, runs=1
- Score trajectory: 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100%
- Entropy trajectory: 0.14 → 0.14 → 0.16 → 0.14 → 0.16 → 0.16 → 0.16 → 0.16 → 0.16 → 0.16
- Fragile counts: 1 → 1 → 1 → 0 → 1 → 1 → 1 → 1 → 1 → 1
- Dominant attribution: none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected

**Scalar baseline** — 10 iterations, trainset=20, runs=1
- Score trajectory: 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100%

**Critique baseline** — 10 iterations, trainset=20, runs=1
- Score trajectory: 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100%

### Adaptive Rag

**CPE-GEPA** — 10 iterations, trainset=20, runs=1
- Score trajectory: 90% → 90% → 90% → 90% → 90% → 90% → 90% → 90% → 90% → 90%
- Entropy trajectory: 0.12 → 0.12 → 0.12 → 0.12 → 0.12 → 0.12 → 0.16 → 0.14 → 0.14 → 0.12
- Fragile counts: 0 → 0 → 0 → 0 → 0 → 0 → 0 → 0 → 0 → 0
- Dominant attribution: none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected

**Scalar baseline** — 10 iterations, trainset=20, runs=1
- Score trajectory: 90% → 90% → 90% → 90% → 90% → 90% → 90% → 90% → 90% → 90%

**Critique baseline** — 10 iterations, trainset=20, runs=1
- Score trajectory: 90% → 95% → 95% → 95% → 100% → 100% → 100% → 100% → 100% → 100%

### Document Triage

**CPE-GEPA** — 10 iterations, trainset=20, runs=1
- Score trajectory: 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100%
- Entropy trajectory: 0.13 → 0.15 → 0.15 → 0.13 → 0.14 → 0.13 → 0.13 → 0.15 → 0.13 → 0.13
- Fragile counts: 0 → 0 → 0 → 0 → 1 → 0 → 0 → 0 → 0 → 0
- Dominant attribution: none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected

**Scalar baseline** — 10 iterations, trainset=20, runs=1
- Score trajectory: 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100%

**Critique baseline** — 10 iterations, trainset=20, runs=1
- Score trajectory: 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100%

### Code Review

**CPE-GEPA** — 10 iterations, trainset=20, runs=1
- Score trajectory: 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100%
- Entropy trajectory: 0.02 → 0.02 → 0.02 → 0.02 → 0.02 → 0.04 → 0.02 → 0.02 → 0.02 → 0.02
- Fragile counts: 0 → 0 → 0 → 0 → 0 → 0 → 0 → 0 → 0 → 0
- Dominant attribution: none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected

**Scalar baseline** — 10 iterations, trainset=20, runs=1
- Score trajectory: 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100%

**Critique baseline** — 10 iterations, trainset=20, runs=1
- Score trajectory: 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100% → 100%

### Medical Triage

**CPE-GEPA** — 10 iterations, trainset=24, runs=1
- Score trajectory: 88% → 88% → 88% → 88% → 88% → 92% → 92% → 92% → 88% → 92%
- Entropy trajectory: 0.25 → 0.25 → 0.29 → 0.27 → 0.22 → 0.28 → 0.25 → 0.27 → 0.30 → 0.28
- Fragile counts: 1 → 1 → 2 → 2 → 0 → 1 → 1 → 2 → 2 → 1
- Dominant attribution: semantic intensity → semantic intensity → semantic intensity → semantic intensity → semantic intensity → semantic intensity → semantic intensity → semantic intensity → semantic intensity → semantic intensity

**Scalar baseline** — 10 iterations, trainset=24, runs=1
- Score trajectory: 88% → 96% → 83% → 92% → 83% → 83% → 58% → 62% → 62% → 62%

**Critique baseline** — 10 iterations, trainset=24, runs=1
- Score trajectory: 88% → 96% → 96% → 96% → 100% → 100% → 100% → 100% → 100% → 100%

### Financial Query

**CPE-GEPA** — 10 iterations, trainset=20, runs=1
- Score trajectory: 85% → 95% → 95% → 95% → 95% → 95% → 95% → 95% → 95% → 95%
- Entropy trajectory: 0.11 → 0.13 → 0.13 → 0.13 → 0.15 → 0.13 → 0.13 → 0.13 → 0.13 → 0.13
- Fragile counts: 0 → 0 → 0 → 0 → 0 → 0 → 0 → 0 → 0 → 0
- Dominant attribution: none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected → none detected

**Scalar baseline** — 10 iterations, trainset=20, runs=1
- Score trajectory: 85% → 95% → 90% → 95% → 95% → 95% → 95% → 90% → 95% → 95%

**Critique baseline** — 10 iterations, trainset=20, runs=1
- Score trajectory: 85% → 95% → 90% → 100% → 100% → 100% → 100% → 100% → 100% → 100%

