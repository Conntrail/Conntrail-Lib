# Conntrail — Agent Development Guide

Decision path tracer for LangGraph agents. Wraps routing nodes and measures decision stability via contrastive analysis.

## Project Overview

- **Language**: Python 3.11+
- **Build System**: Hatchling
- **Package Manager**: `uv` (preferred) or `pip`
- **Test Runner**: pytest
- **Linter/Formatter**: Ruff

## Quick Commands

### Setup
```bash
# Install with uv (recommended)
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all unit tests (fast, no API keys needed)
pytest tests/ -v

# Run a single test file
pytest tests/test_record.py -v

# Run a single test class
pytest tests/test_record.py::TestStabilityLabel -v

# Run a single test method
pytest tests/test_record.py::TestStabilityLabel::test_zero_entropy_is_confident -v

# Run integration tests (requires API keys)
pytest testing/test_conntrail_integration.py -v -m "integration"

# Skip slow tests
pytest tests/ -v -m "not slow"
```

### Linting and Formatting

```bash
# Check all files
ruff check .

# Fix auto-fixable issues
ruff check . --fix

# Format all files
ruff format .

# Check specific file
ruff check conntrail/wrap.py
```

### Type Checking (if using mypy)
```bash
mypy conntrail/
```

## Code Style Guidelines

### Imports
- Use `from __future__ import annotations` at the top of every file
- Group imports: stdlib → third-party → local
- Use absolute imports for project modules: `from conntrail.config import ConntrailConfig`

```python
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Literal

from langchain_core.messages import HumanMessage

from conntrail.config import ConntrailConfig
```

### Formatting
- **Line length**: 100 characters (configured in `pyproject.toml`)
- Use double quotes for strings
- Trailing commas in multi-line collections

### Type Hints
- Use Python 3.11+ union syntax: `str | None`, `dict[str, Any]`
- Prefer `dict`, `list` over `Dict`, `List` (from `__future__`)
- Use `Literal` for string enums: `Literal["jsonl", "stdout", "langsmith"]`
- Annotate all function parameters and return types
- Use `Callable` for callback types

### Naming Conventions
- **Classes**: PascalCase (e.g., `NodeInterceptor`, `TraceRecord`)
- **Functions/Methods**: snake_case (e.g., `trace_graph`, `_should_sample`)
- **Constants**: UPPER_SNAKE_CASE for module-level constants (e.g., `TRACE_KEY`)
- **Private members**: leading underscore (e.g., `_run_contrast_analysis`)
- **Type variables**: PascalCase, descriptive (e.g., `T`, `State`)

### Docstrings
- Use Google-style docstrings
- First line is a concise summary
- Include Args/Returns sections for public APIs

```python
def trace_graph(
    compiled_graph: Any,
    config: ConntrailConfig | None = None,
) -> Any:
    """Wrap all nodes in a compiled LangGraph with Conntrail tracing.

    Patches each node's underlying callable in-place. Does not recompile
    the graph. Returns the same compiled_graph object.

    Args:
        compiled_graph: A LangGraph CompiledStateGraph instance.
        config: ConntrailConfig. Defaults to ConntrailConfig() if not provided.

    Returns:
        The same compiled_graph with all nodes wrapped.
    """
```

### Error Handling
- Use specific exceptions (`ValueError` for config validation)
- Include descriptive error messages
- Use `logger.warning()` for non-fatal errors in async operations
- Never raise from fire-and-forget tasks

```python
def __post_init__(self) -> None:
    if not 0.0 <= self.sample_rate <= 1.0:
        raise ValueError(f"sample_rate must be in [0.0, 1.0], got {self.sample_rate}")
```

### Async Patterns
- Prefer `async`/`await` over threading
- Fire-and-forget tasks: `asyncio.create_task()`
- Use `asyncio.iscoroutinefunction()` to detect async functions
- ContextVars for per-invocation state: `_active_trace_list: ContextVar[list | None]`

### Data Classes
- Use `@dataclass` for configuration and records
- Use `field(default=None, repr=False)` for sensitive/non-printable fields
- Implement `to_dict()` / `from_dict()` for serialization

```python
@dataclass
class ConntrailConfig:
    contrast_model: str = "claude-haiku-4-5-20251001"
    on_alert: Callable | None = field(default=None, repr=False)
```

## Test Patterns

### Test Structure
- Use classes to group related tests: `class TestStabilityLabel:`
- Descriptive test names: `test_zero_entropy_is_confident`
- Use pytest fixtures in `conftest.py` for shared setup

### Test Markers
- `@pytest.mark.asyncio` for async tests
- `@pytest.mark.integration` — requires live API keys
- `@pytest.mark.baseline` — agent works without Conntrail
- `@pytest.mark.slow` — takes > 10 seconds

### Test Helpers
```python
def _make_record(**overrides) -> TraceRecord:
    """Build a minimal but complete TraceRecord for testing."""
    defaults = dict(...)
    defaults.update(overrides)
    return TraceRecord(**defaults)
```

## Project Structure

```
conntrail/           # Main package
  __init__.py        # Public API exports
  config.py          # ConntrailConfig dataclass
  wrap.py            # trace_node, trace_graph decorators
  interceptor.py     # NodeInterceptor class
  record.py          # TraceRecord dataclass
  contrast.py        # ContrastGenerator, ContrastSet
  analyser.py        # DivergenceAnalyser
  exporters/         # Output format handlers
    stdout.py
    jsonl.py
  utils/             # Provider utilities
  prompts/           # LLM prompt templates

tests/               # Unit tests (fast, no API keys)
  test_record.py
  test_interceptor.py
  conftest.py        # pytest configuration

testing/             # Integration tests (needs API keys)
  test_conntrail_integration.py
  agents/            # Sample LangGraph agents for testing
  .env               # API keys (gitignored)
```

## Configuration

Key settings in `pyproject.toml`:
- `[tool.ruff]`: line-length = 100, target-version = "py311"
- `[tool.pytest.ini_options]`: asyncio_mode = "auto", markers for integration/slow tests

## Dependencies

Core: `langgraph`, `langchain-core`, `anthropic`, `pydantic`
Dev: `pytest`, `pytest-asyncio`, `pytest-mock`, `ruff`
Optional: `langsmith`, provider packages (`langchain-groq`, etc.)

uv run python -m experiments.runs.run_cpe_gepa --agents all --iterations 10 --runs 3 --out experiments/results/cpe_gepa.json && uv run python -m experiments.runs.run_baselines --agents all --baseline all --iterations 10 --runs 3 --out experiments/results/baselines.json && uv run python -m experiments.runs.write_summary