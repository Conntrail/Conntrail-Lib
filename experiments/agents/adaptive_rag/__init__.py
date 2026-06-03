from .agent import build_graph

# Each item: {"input": str, "expected_route": str}
# Routes: direct_answer | vector_search | web_search
TRAINSET = [
    {"input": "What is the capital of Germany?", "expected_route": "direct_answer"},
    {"input": "Explain the theory of relativity.", "expected_route": "direct_answer"},
    {"input": "What does HTTP stand for?", "expected_route": "direct_answer"},
    {"input": "What were Conntrail's Q4 2024 sales figures?", "expected_route": "vector_search"},
    {"input": "What is the current customer churn rate?", "expected_route": "vector_search"},
    {"input": "How many features did engineering ship last quarter?", "expected_route": "vector_search"},
    {"input": "What is the latest news about AI regulation?", "expected_route": "web_search"},
    {"input": "Who won the most recent US election?", "expected_route": "web_search"},
    {"input": "What is the current Bitcoin price?", "expected_route": "web_search"},
    {"input": "What happened in the news today?", "expected_route": "web_search"},
]

__all__ = ["build_graph", "TRAINSET"]
