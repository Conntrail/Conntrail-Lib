"""
DivergenceAnalyser — runs original + 3 contrasts through a node and measures routing divergence.

Computes Shannon entropy over the 4 routing outcomes and infers which semantic
dimension drove the decision (attribution).
"""
from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Callable

from contrail.contrast import ContrastSet
from contrail.utils.entropy import routing_entropy

# Fixed attribution labels by contrast dimension.
# The dimension that first flips the route names the attribution.
# LLM-based open-ended labelling is a Phase 6 enhancement.
_ATTRIBUTION_LABELS = {
    "opposite": "semantic intensity",   # core dimension fully inverted → strongest signal
    "neutral":  "urgency/sentiment",    # stripping emphasis changed the route
    "similar":  "surface form",         # even a paraphrase flipped the route → very fragile
}


@dataclass
class AnalysisResult:
    """
    Output of DivergenceAnalyser.analyse() for one node call.
    Consumed by NodeInterceptor to build a TraceRecord.
    """

    original_route: str
    contrast_routes: dict[str, str]   # {"similar": route, "neutral": route, "opposite": route}
    entropy_score: float
    attribution_dimension: str
    counterfactual_route: str | None
    raw_outputs: dict[str, Any]       # all 4 node outputs keyed by variant name


class DivergenceAnalyser:
    """
    Runs the traced node with original + 3 contrast inputs concurrently.
    Measures routing divergence and computes attribution.

    Routing comparison strategy:
      - Branching nodes: auto-detect the state key that changed, use its value as the route.
        Pass ``route_key`` explicitly to skip auto-detection.
      - Non-branching nodes: embed outputs, measure cosine distance (threshold: COSINE_THRESHOLD).
    """

    COSINE_THRESHOLD: float = 0.3   # calibrated in Phase 6

    async def analyse(
        self,
        node_fn: Callable,
        original_input: dict[str, Any],
        contrast_set: ContrastSet,
        input_key: str = "message",
        route_key: str | None = None,
    ) -> AnalysisResult:
        """
        Run all 4 inputs through node_fn concurrently and analyse divergence.

        Args:
            node_fn:        The LangGraph node function being traced.
            original_input: The original state dict passed to the node.
            contrast_set:   The 3 contrast variants to run.
            input_key:      Which state key holds the text input (default: "message").
            route_key:      Which output key holds the routing decision.
                            Auto-detected if None.

        Returns:
            AnalysisResult with entropy_score, attribution_dimension, and
            counterfactual_route.
        """
        variants: dict[str, dict[str, Any]] = {
            "original": original_input,
            "similar":  {**original_input, input_key: contrast_set.similar},
            "neutral":  {**original_input, input_key: contrast_set.neutral},
            "opposite": {**original_input, input_key: contrast_set.opposite},
        }

        # Run all 4 concurrently
        outputs_list = await asyncio.gather(
            *[self._call_node(node_fn, state) for state in variants.values()]
        )
        raw_outputs: dict[str, Any] = dict(zip(variants.keys(), outputs_list))

        # Extract route label from each output
        routes: dict[str, str] = {
            name: self._extract_route(variants[name], output, route_key=route_key)
            for name, output in raw_outputs.items()
        }

        entropy = routing_entropy(list(routes.values()))
        contrast_routes = {k: v for k, v in routes.items() if k != "original"}
        attribution, counterfactual = self._infer_attribution(
            original_route=routes["original"],
            contrast_routes=contrast_routes,
        )

        return AnalysisResult(
            original_route=routes["original"],
            contrast_routes=contrast_routes,
            entropy_score=entropy,
            attribution_dimension=attribution,
            counterfactual_route=counterfactual,
            raw_outputs=raw_outputs,
        )

    async def _call_node(self, node_fn: Callable, state: dict[str, Any]) -> Any:
        """Call node_fn, handling both sync and async callables."""
        if inspect.iscoroutinefunction(node_fn):
            return await node_fn(state)
        return await asyncio.to_thread(node_fn, state)

    def _extract_route(
        self,
        input_state: dict[str, Any],
        output_state: dict[str, Any],
        route_key: str | None = None,
    ) -> str:
        """
        Extract the routing label from a node's output state.

        Strategy (in order):
          1. Use ``route_key`` directly if provided.
          2. Find a key that was None in input and is now a non-empty string.
          3. Find any string key whose value changed from input to output.
          4. Fall back to "unknown".
        """
        if route_key:
            value = output_state.get(route_key)
            return str(value) if value is not None else "unknown"

        # Strategy 2: None → string
        for key, value in output_state.items():
            if isinstance(value, str) and value and input_state.get(key) is None:
                return value

        # Strategy 3: any string key that changed
        for key, value in output_state.items():
            if isinstance(value, str) and value and value != input_state.get(key):
                return value

        return "unknown"

    def _infer_attribution(
        self,
        original_route: str,
        contrast_routes: dict[str, str],
    ) -> tuple[str, str | None]:
        """
        Infer attribution dimension and counterfactual route.

        Priority: opposite > neutral > similar.
        The first dimension whose route differs from the original names the driver.

        Returns:
            (attribution_dimension, counterfactual_route)
        """
        for dim in ("opposite", "neutral", "similar"):
            if contrast_routes.get(dim) != original_route:
                return _ATTRIBUTION_LABELS[dim], contrast_routes[dim]

        return "none detected", None
