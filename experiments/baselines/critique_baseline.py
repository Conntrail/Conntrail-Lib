"""
GEPA with LLM self-critique feedback.
Compares cost and convergence speed against CPE-guided feedback.
"""
import dspy


def _make_critique_metric():
    def critique_metric(gold, pred, trace=None):
        score = float(gold.get("expected_route") == pred.get("route"))
        prompt = (
            f"The agent routed to '{pred.get('route')}'. "
            f"Expected: '{gold.get('expected_route')}'. "
            f"Input was: {gold.get('input', '')}. "
            "In one sentence, explain why the routing was correct or incorrect "
            "and what the prompt should clarify."
        )
        lm = dspy.settings.lm
        feedback = lm(prompt).completions[0].text.strip()
        return score, feedback

    return critique_metric


def run_critique_baseline(student, trainset: list, iterations: int = 10):
    gepa = dspy.GEPA(metric=_make_critique_metric(), num_iterations=iterations)
    return gepa.compile(student, trainset=trainset)
