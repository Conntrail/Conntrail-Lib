# Contributing to Conntrail

Thank you for your interest in contributing to Conntrail! This document provides guidelines and instructions for contributing to the project.

## Welcome

We welcome contributions of all kinds, including bug reports, feature requests, documentation improvements, and code contributions. Whether you're fixing a typo or implementing a major feature, your contribution is appreciated.

## Development Setup

To set up your development environment:

```bash
# Clone the repository
git clone https://github.com/ibzie/conntrail.git
cd conntrail

# Install in editable mode with all development dependencies
pip install -e ".[dev,testing]"
```

This installs:
- The `conntrail` package in editable mode
- Development dependencies: pytest, pytest-asyncio, pytest-mock, ruff
- Testing dependencies: langchain providers, python-dotenv

## Running Tests

Run the test suite with pytest:

```bash
# Run all unit tests (fast, no API keys needed)
pytest tests/ -v

# Run a specific test file
pytest tests/test_record.py -v

# Run integration tests (requires API keys)
pytest testing/test_conntrail_integration.py -v -m "integration"

# Skip slow tests
pytest tests/ -v -m "not slow"
```

## Code Style

Please follow the code style guidelines detailed in [`AGENTS.md`](AGENTS.md). Key points:

- Use `from __future__ import annotations` at the top of every file
- Line length: 100 characters
- Use Python 3.11+ union syntax: `str | None`, `dict[str, Any]`
- Use Google-style docstrings
- Run `ruff check .` and `ruff format .` before committing

## Pull Request Process

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following the code style guidelines
3. **Add tests** for any new functionality
4. **Run the test suite** to ensure all tests pass
5. **Run linting** with `ruff check .` and fix any issues
6. **Update documentation** if needed (README, docstrings, etc.)
7. **Submit a pull request** with a clear description of the changes

### PR Guidelines

- Keep changes focused and atomic
- Write clear commit messages
- Reference any related issues in your PR description
- Ensure CI checks pass before requesting review

## Reporting Issues

When reporting issues, please include:

- **Description**: A clear description of the issue
- **Steps to reproduce**: How to recreate the problem
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Environment**: Python version, OS, and package versions
- **Code sample**: A minimal reproducible example if possible

### Security Issues

If you discover a security vulnerability, please do not open a public issue. Instead, email the maintainers directly.

## Code of Conduct

Be respectful and constructive in all interactions. We aim to maintain a welcoming and inclusive community for everyone.

## Questions?

If you have questions about contributing, feel free to:
- Open an issue with the "question" label
- Contact the maintainers

Thank you for contributing to Conntrail!
