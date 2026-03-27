"""
Embedding utilities — cosine similarity for non-branching node comparison.

Used by DivergenceAnalyser when a node produces free-form text output
rather than an explicit routing label.
"""
from __future__ import annotations

import math


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Returns a value in [-1.0, 1.0] where 1.0 means identical direction.

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector. Must be same length as vec_a.

    Returns:
        Cosine similarity score.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector length mismatch: {len(vec_a)} vs {len(vec_b)}")

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


async def embed_text(text: str, model: str = "claude-haiku-4-5-20251001") -> list[float]:
    """
    Embed a text string for similarity comparison.

    Used as fallback for non-branching nodes where routing labels
    are not available.

    Args:
        text: Text to embed.
        model: Model to use (uses Anthropic embeddings endpoint).

    Returns:
        Embedding vector as list of floats.
    """
    raise NotImplementedError("Phase 3")


def output_distance(embedding_a: list[float], embedding_b: list[float]) -> float:
    """
    Compute distance between two output embeddings.

    Returns a value in [0.0, 2.0] where 0.0 = identical, 2.0 = maximally different.
    Distance threshold for "different route": 0.3 (tuned in Phase 6).
    """
    return 1.0 - cosine_similarity(embedding_a, embedding_b)
