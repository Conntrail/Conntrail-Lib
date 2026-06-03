from .agent import build_graph

# Each item: {"input": str, "expected_route": str}
# Used by run_cpe_gepa.routing_accuracy(gold, pred)
TRAINSET = [
    {"input": "I want a refund for my order, it never arrived.", "expected_route": "refund"},
    {"input": "I need to return this item, it's broken.", "expected_route": "refund"},
    {"input": "Can I get my money back? The product doesn't work.", "expected_route": "refund"},
    {"input": "This is completely unacceptable! I want to speak to a manager right now!", "expected_route": "escalation"},
    {"input": "I'm furious. Fix this immediately or I'm cancelling everything.", "expected_route": "escalation"},
    {"input": "Where is my order? It was supposed to arrive last week.", "expected_route": "order_info"},
    {"input": "Can you give me the tracking number for my shipment?", "expected_route": "order_info"},
    {"input": "What are your store hours?", "expected_route": "general"},
    {"input": "Do you offer gift wrapping?", "expected_route": "general"},
    {"input": "How do I create an account on your website?", "expected_route": "general"},
]

__all__ = ["build_graph", "TRAINSET"]
