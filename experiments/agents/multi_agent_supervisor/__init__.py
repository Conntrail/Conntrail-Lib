from .agent import build_graph

# Each item: {"input": str, "expected_route": str}
# Routes: code_agent | writing_agent | research_agent
TRAINSET = [
    {"input": "Write a Python function to sort a list of dictionaries by a key.", "expected_route": "code_agent"},
    {"input": "Debug this Python script: it raises a KeyError on line 12.", "expected_route": "code_agent"},
    {"input": "Implement a binary search tree in Python.", "expected_route": "code_agent"},
    {"input": "Write a blog post about the benefits of remote work.", "expected_route": "writing_agent"},
    {"input": "Summarise this article in three bullet points.", "expected_route": "writing_agent"},
    {"input": "Edit this paragraph to sound more professional.", "expected_route": "writing_agent"},
    {"input": "What are the latest AI research papers on transformers?", "expected_route": "research_agent"},
    {"input": "Find information about the current inflation rate in the EU.", "expected_route": "research_agent"},
    {"input": "Look up the top 5 programming languages in 2025.", "expected_route": "research_agent"},
    {"input": "What is the population of Tokyo?", "expected_route": "research_agent"},
]

__all__ = ["build_graph", "TRAINSET"]
