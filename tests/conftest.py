"""
Pytest configuration for contrail unit tests.
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires live API keys")
    config.addinivalue_line("markers", "baseline: agent works without Contrail")
    config.addinivalue_line("markers", "slow: takes > 10 seconds")
