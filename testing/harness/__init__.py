"""
Contrail testing harness.
Provides BaseTestRunner, assertion helpers, and shared input fixtures.
"""
from .assertions import assert_entropy_range, assert_non_intrusive, assert_trace_record
from .llm import get_contrast_llm, get_llm
from .runner import BaseTestRunner

__all__ = [
    "BaseTestRunner",
    "assert_trace_record",
    "assert_entropy_range",
    "assert_non_intrusive",
    "get_llm",
    "get_contrast_llm",
]
