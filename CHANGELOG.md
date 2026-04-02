# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Deprecated

### Removed

### Fixed

### Security

## [0.1.0] - 2026-04-03

### Added

- Initial release of Conntrail
- Decision path tracing for LangGraph agents
- `trace_node()` decorator for wrapping individual nodes
- `trace_graph()` function for wrapping entire compiled graphs
- `ConntrailConfig` dataclass for configuration
- `NodeInterceptor` class for intercepting node invocations
- `TraceRecord` dataclass for storing trace information
- Contrastive analysis via `ContrastGenerator` and `ContrastSet`
- `DivergenceAnalyser` for analyzing routing stability
- Entropy-based stability scoring
- Alert callbacks via `on_alert` configuration option
- Multiple export destinations: stdout, JSONL, LangSmith
- Sampling support with configurable sample rate
- Async/await support throughout the codebase
- ContextVar-based trace state management
- Comprehensive test suite with pytest
- Ruff linting and formatting configuration
- Support for Python 3.11 and 3.12

### Features

- **Node Wrapping**: Wrap individual LangGraph nodes to capture inputs/outputs
- **Graph Wrapping**: Wrap entire compiled LangGraph graphs in-place
- **Contrastive Analysis**: Generate contrast sets and measure decision stability
- **Entropy Scoring**: Calculate routing entropy to detect uncertain decisions
- **Export Options**: Output traces to stdout, JSONL files, or LangSmith
- **Sampling**: Configurable sampling rate for production use
- **Alerts**: Customizable alert callbacks for unstable routing decisions
- **Async Support**: Full async/await compatibility with proper context handling
