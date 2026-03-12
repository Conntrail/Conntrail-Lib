"""
DivergenceAnalyser — runs original + 3 contrasts through a node and measures routing divergence.

Computes Shannon entropy over the 4 routing outcomes and infers which semantic
dimension drove the decision (attribution).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from contrail.contrast import ContrastSet


@dataclass
class AnalysisResult:
    """
    Output of DivergenceAnalyser.analyse() for one node call.

    Consumed by NodeInterceptor to build a TraceRecord.
    """

    original_route: str
    contrast_routes: dict[str, str]  # {"similar": route, "neutral": route, "opposite": route}
    entropy_score: float
    attribution_dimension: str
    counterfactual_route: str | None
    raw_outputs: dict[str, Any]  # all 4 node outputs keyed by variant name


class DivergenceAnalyser:
    """
    Runs the traced node with original + 3 contrast inputs concurrently.
    Measures routing divergence and computes attribution.

    Routing comparison strategy:
      - Branching nodes: compare the route key in the output dict.
      - Non-branching nodes: embed outputs, measure cosine distance (threshold: 0.3).
    """

    COSINE_THRESHOLD: float = 0.3  # tuned in Phase 6

    async def analyse(
        self,
        node_fn: Callable,
        original_input: Any,
        contrast_set: ContrastSet,
        input_key: str = "message",
    ) -> AnalysisResult:
        """
        Run all 4 inputs through node_fn concurrently and analyse divergence.

        Args:
            node_fn: The LangGraph node function being traced.
            original_input: The original state dict passed to the node.
            contrast_set: The 3 contrast variants to run.
            input_key: Which state key holds the text input.

        Returns:
            AnalysisResult with entropy, attribution, and counterfactual route.
        """
        raise NotImplementedError("Phase 3")

    def _extract_route(self, output: Any) -> str:
        """Extract routing label from node output."""
        raise NotImplementedError("Phase 3")

    def _infer_attribution(
        self,
        original_route: str,
        contrast_routes: dict[str, str],
    ) -> tuple[str, str | None]:
        """
        Infer attribution dimension and counterfactual route.

        Returns:
            (attribution_dimension, counterfactual_route)
        """
        raise NotImplementedError("Phase 3")
