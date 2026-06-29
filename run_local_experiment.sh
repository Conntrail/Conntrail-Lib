#!/usr/bin/env bash
set -euo pipefail

export LOCAL_PASSWORD="rochie2145"
export LOCAL_MODEL_NAME="unsloth/Qwen3.6-27B-MTP-GGUF"
export LOCAL_LLM_URL="http://127.0.0.1:8888/v1"

OUT_DIR="experiments/results/run_local"
mkdir -p "$OUT_DIR"

echo "========================================"
echo " Conntrail Local Experiment (Qwen3 27B)"
echo " Output: $OUT_DIR"
echo "========================================"
echo ""

# Excluded: react_agent (slow tool-calling, 2-route trivial), document_triage (redundant with customer_support)
# Added: mental_health_triage, content_moderation (high-volatility routing)
AGENTS="customer_support,adaptive_rag,code_review,financial_query,medical_triage,multi_agent_supervisor,mental_health_triage,content_moderation"

echo "[1/3] Running CPE-GEPA (3 runs x 10 iterations)..."
uv run python -m experiments.runs.run_cpe_gepa \
  --agents "$AGENTS" \
  --iterations 10 \
  --runs 3 \
  --out "$OUT_DIR/cpe_gepa.json" \
  --local \
  --resume

echo ""
echo "[2/3] Running baselines (3 runs x 10 iterations)..."
uv run python -m experiments.runs.run_baselines \
  --agents "$AGENTS" \
  --baseline all \
  --iterations 10 \
  --runs 3 \
  --out "$OUT_DIR/baselines.json" \
  --local \
  --resume

echo ""
echo "[3/3] Writing summaries..."
uv run python -m experiments.runs.write_summary --dir "$OUT_DIR"

echo ""
echo "========================================"
echo " Done. Results in $OUT_DIR/"
echo "========================================"
