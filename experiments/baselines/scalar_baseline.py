"""
GEPA with scalar-only feedback — no CPE, no textual critique.
Metric returns (score, "") so GEPA receives no routing-stability signal.
Primary comparison for the CPE-GEPA experiment.
"""
import dspy


def scalar_metric(gold, pred, trace=None):
    score = float(gold.get("expected_route") == pred.get("route"))
    return score, ""


def run_scalar_baseline(student, trainset: list, iterations: int = 10):
    gepa = dspy.GEPA(metric=scalar_metric, num_iterations=iterations)
    return gepa.compile(student, trainset=trainset)
