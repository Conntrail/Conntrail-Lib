from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bridge import TraceCollector

from .schema import PromptAttemptRecord

_STABILITY_ADVICE = {
    "fragile": (
        "Multiple routing decisions changed under semantic paraphrase. "
        "The prompt is underspecified at this decision point — small wording "
        "differences redirect the agent. Consider adding explicit routing criteria "
        "or examples for this case type."
    ),
    "boundary": (
        "Some routing decisions were inconsistent under paraphrase. "
        "The prompt is near a decision boundary. Clarifying language or a concrete "
        "example may stabilise routing."
    ),
    "confident": (
        "Routing was stable under paraphrase. "
        "The prompt is clear at this decision point."
    ),
}


def cpe_feedback(attempt: PromptAttemptRecord) -> str:
    """
    Converts a PromptAttemptRecord into a natural-language feedback string
    suitable for GEPA's textual feedback slot.
    """
    if not attempt.traces:
        return (
            "No Conntrail traces were collected for this prompt attempt. "
            "Cannot provide entropy-based feedback. Check that the agent ran "
            "with Conntrail instrumentation and entropy_alert_threshold=0.0."
        )

    mean_e = attempt.mean_entropy
    fragile = attempt.fragile_count
    boundary = attempt.boundary_count
    confident = len(attempt.traces) - fragile - boundary
    attribution = attempt.dominant_attribution or "unknown dimension"

    if fragile > boundary and fragile > confident:
        dominant = "fragile"
    elif boundary >= fragile and boundary > confident:
        dominant = "boundary"
    else:
        dominant = "confident"

    advice = _STABILITY_ADVICE[dominant]

    lines = [
        f"CPE Analysis ({len(attempt.traces)} routing nodes sampled):",
        f"  Mean entropy: {mean_e:.3f} | "
        f"Fragile: {fragile} | Boundary: {boundary} | Confident: {confident}",
        f"  Primary instability dimension: {attribution}",
        "",
        advice,
    ]

    if dominant != "confident" and attempt.scalar_score is not None:
        lines.append(
            f"\nTask score: {attempt.scalar_score:.3f}. "
            "Routing instability likely contributes to score variance across inputs."
        )

    return "\n".join(lines)


class CPEFeedbackFunction:
    """
    Callable wrapper around cpe_feedback for use as GEPA's metric/feedback function.

    GEPA expects: feedback_fn(gold, pred, trace=None) -> (score: float, feedback: str)

    Must be used alongside a TraceCollector. The collector captures traces during
    the rollout; this function retrieves the most recently completed attempt.

    Args:
        collector: TraceCollector instance shared with the agent run.
        task_metric_fn: Optional callable(gold, pred) -> float for task accuracy.
                        Falls back to (1 - mean_entropy) as a stability proxy.
    """

    def __init__(self, collector: "TraceCollector", task_metric_fn=None) -> None:
        self._collector = collector
        self._task_metric = task_metric_fn

    def __call__(self, gold, pred, trace=None):
        attempts = self._collector.all_attempts
        if not attempts:
            return 0.0, "No attempts recorded yet."

        latest = attempts[-1]

        if self._task_metric is not None:
            score = float(self._task_metric(gold, pred))
        elif latest.scalar_score is not None:
            score = latest.scalar_score
        elif latest.mean_entropy is not None:
            score = 1.0 - latest.mean_entropy
        else:
            score = 0.0

        feedback = cpe_feedback(latest)
        return score, feedback
