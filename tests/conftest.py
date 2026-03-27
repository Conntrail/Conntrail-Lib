"""
Pytest configuration for conntrail unit tests.
"""
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load testing/.env so integration tests have API keys available
load_dotenv(Path(__file__).parent.parent / "testing" / ".env")


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires live API keys")
    config.addinivalue_line("markers", "baseline: agent works without Conntrail")
    config.addinivalue_line("markers", "slow: takes > 10 seconds")
