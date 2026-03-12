"""
TraceRecord — holds the full analysis output for one node call.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from contrail.contrast import ContrastSet


@dataclass
class TraceRecord:
    """
    Complete trace output for a single node invocation.

    Produced by DivergenceAnalyser and consumed by exporters.
    Attached to agent state under the key ``__contrail_traces__``.
    """

    trace_id: str
    node_id: str
    timestamp: datetime
    original_input: str
    original_route: str
    entropy_score: float
    stability: Literal["confident", "boundary", "fragile"]
    attribution_dimension: str
    plain_language_summary: str
    raw_contrasts: ContrastSet
    raw_outputs: dict[str, Any]
    counterfactual_route: str | None = None

    @classmethod
    def make_id(cls) -> str:
        return str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        raise NotImplementedError("Phase 4")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceRecord:
        """Reconstruct a TraceRecord from a serialised dictionary."""
        raise NotImplementedError("Phase 4")

    @staticmethod
    def stability_label(entropy: float) -> Literal["confident", "boundary", "fragile"]:
        """Map an entropy score to a stability label."""
        if entropy < 0.25:
            return "confident"
        elif entropy <= 0.60:
            return "boundary"
        return "fragile"

    @staticmethod
    def build_summary(
        node_id: str,
        route: str,
        stability: str,
        entropy: float,
        attribution: str,
        counterfactual: str | None,
    ) -> str:
        """Generate the plain-language summary string."""
        cf_part = (
            f" If that dimension were removed, the agent would likely have taken "
            f"the '{counterfactual}' path instead."
            if counterfactual
            else ""
        )
        return (
            f"The '{node_id}' node routed to '{route}' with {stability} confidence "
            f"(entropy: {entropy:.2f}). The decision appears to have been driven by "
            f"{attribution} in the input.{cf_part}"
        )
