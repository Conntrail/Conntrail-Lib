"""
ContrailConfig — single configuration object passed at setup time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal


@dataclass
class ContrailConfig:
    """
    Configuration for Contrail tracing.

    Attributes:
        contrast_model: LLM model ID used for contrast generation.
            Should be a cheap/fast model — never the model being traced.
        sample_rate: Fraction of node calls to analyse [0.0, 1.0].
            1.0 = trace every call (dev default). 0.1–0.2 recommended for prod.
        async_mode: When True, contrast analysis never blocks the hot path.
        export_format: Where to write TraceRecords.
        export_path: Directory path for jsonl exports.
        entropy_alert_threshold: entropy_score >= this value triggers on_alert.
        on_alert: Optional callback fired when a fragile node is detected.
            Signature: (trace_record: TraceRecord) -> None
    """

    contrast_model: str = "claude-haiku-4-5-20251001"
    sample_rate: float = 1.0
    async_mode: bool = True
    export_format: Literal["jsonl", "langsmith", "stdout"] = "jsonl"
    export_path: str = "./contrail_traces"
    entropy_alert_threshold: float = 0.6
    on_alert: Callable | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be in [0.0, 1.0], got {self.sample_rate}")
        if self.export_format not in ("jsonl", "langsmith", "stdout"):
            raise ValueError(f"Invalid export_format: {self.export_format!r}")
        if not 0.0 <= self.entropy_alert_threshold <= 1.0:
            raise ValueError(
                f"entropy_alert_threshold must be in [0.0, 1.0], got {self.entropy_alert_threshold}"
            )
