from .agent import build_graph

# Each item: {"input": str, "expected_route": str}
# Routes: call_tool | respond
TRAINSET = [
    {"input": "What is 1234 * 5678?", "expected_route": "call_tool"},
    {"input": "Calculate 99 divided by 7.", "expected_route": "call_tool"},
    {"input": "What is 2 to the power of 16?", "expected_route": "call_tool"},
    {"input": "What is the square root of 144?", "expected_route": "call_tool"},
    {"input": "What is the capital of France?", "expected_route": "respond"},
    {"input": "Explain what photosynthesis is.", "expected_route": "respond"},
    {"input": "What is the speed of light?", "expected_route": "respond"},
    {"input": "Write me a haiku about autumn.", "expected_route": "respond"},
    {"input": "What is 15% of 240?", "expected_route": "call_tool"},
    {"input": "Tell me a joke.", "expected_route": "respond"},
]

__all__ = ["build_graph", "TRAINSET"]
