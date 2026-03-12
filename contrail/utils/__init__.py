"""
Contrail utility functions — entropy calculation and embedding similarity.
"""
from contrail.utils.embedding import cosine_similarity, output_distance
from contrail.utils.entropy import routing_entropy

__all__ = ["routing_entropy", "cosine_similarity", "output_distance"]
