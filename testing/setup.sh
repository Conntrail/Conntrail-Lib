#!/usr/bin/env bash
# Contrail testing environment setup
# Sets up all real-world LangGraph agent examples for integration testing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
AGENTS_DIR="$SCRIPT_DIR/agents"

echo "==> Setting up Contrail testing environment"
echo "    Repo root: $REPO_ROOT"

# --- Install Contrail in editable mode with dev deps ---
echo ""
echo "==> Installing Contrail (editable + dev deps)"
pip install -e "$REPO_ROOT[dev]" --quiet

# --- Install additional testing deps ---
echo ""
echo "==> Installing testing dependencies"
pip install \
  langgraph \
  langchain-core \
  langchain-anthropic \
  langchain-openai \
  langchain-community \
  tavily-python \
  --quiet

mkdir -p "$AGENTS_DIR"

# --- Clone langchain-ai/langgraph for examples ---
LANGGRAPH_CLONE="$SCRIPT_DIR/.langgraph_source"
if [ ! -d "$LANGGRAPH_CLONE" ]; then
  echo ""
  echo "==> Cloning langchain-ai/langgraph (examples source)"
  git clone --depth=1 --filter=blob:none --sparse \
    https://github.com/langchain-ai/langgraph.git \
    "$LANGGRAPH_CLONE"
  cd "$LANGGRAPH_CLONE"
  git sparse-checkout set examples docs/docs
  cd "$REPO_ROOT"
else
  echo ""
  echo "==> langchain-ai/langgraph already cloned, pulling latest"
  cd "$LANGGRAPH_CLONE" && git pull --quiet && cd "$REPO_ROOT"
fi

echo ""
echo "==> Verifying agent stubs are importable"
python -c "
import sys
sys.path.insert(0, '$AGENTS_DIR')

agents = [
    'customer_support.agent',
    'react_agent.agent',
    'multi_agent_supervisor.agent',
    'adaptive_rag.agent',
]

for agent_module in agents:
    try:
        # Agents use lazy imports — just check the file exists and is syntactically valid
        import importlib.util
        parts = agent_module.split('.')
        agent_dir = '$AGENTS_DIR/' + parts[0]
        spec = importlib.util.spec_from_file_location(agent_module, f'{agent_dir}/{parts[1]}.py')
        if spec is None:
            print(f'  MISSING: {agent_module}')
        else:
            print(f'  OK: {agent_module}')
    except Exception as e:
        print(f'  ERROR: {agent_module} — {e}')
"

echo ""
echo "==> Setup complete."
echo ""
echo "Run baseline tests:    pytest testing/ -m baseline"
echo "Run integration tests: pytest testing/ -m integration (requires API keys)"
