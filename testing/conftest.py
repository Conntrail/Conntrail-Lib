"""
Pytest configuration for the Contrail testing environment.
Provides shared fixtures for all agent integration tests.
"""
import os
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "baseline: agent works without Contrail")
    config.addinivalue_line("markers", "integration: requires live API keys")
    config.addinivalue_line("markers", "slow: takes > 10 seconds")


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless API keys are present."""
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))

    for item in items:
        if "integration" in item.keywords:
            if not (has_anthropic or has_openai):
                item.add_marker(
                    pytest.mark.skip(
                        reason="No API keys found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY."
                    )
                )


@pytest.fixture(scope="session")
def anthropic_api_key():
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return key


@pytest.fixture(scope="session")
def openai_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key
