"""
Conntrail utility functions — entropy calculation and embedding similarity.
"""
from conntrail.utils.embedding import cosine_similarity, output_distance
from conntrail.utils.entropy import routing_entropy
from conntrail.utils.providers import get_chat_model

__all__ = ["routing_entropy", "cosine_similarity", "output_distance", "get_chat_model"]
