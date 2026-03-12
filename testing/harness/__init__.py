"""
Contrail testing harness.
Provides BaseTestRunner, assertion helpers, and shared input fixtures.
"""
from .runner import BaseTestRunner
from .assertions import assert_trace_record, assert_entropy_range, assert_non_intrusive

__all__ = ["BaseTestRunner", "assert_trace_record", "assert_entropy_range", "assert_non_intrusive"]
