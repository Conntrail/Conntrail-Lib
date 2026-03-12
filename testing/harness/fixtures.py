"""
Standard input sets shared across all agent tests.

Each input is tagged with its expected entropy category:
  - confident: clearly maps to one routing bucket (expected entropy < 0.25)
  - boundary:  ambiguous, could go either way (expected entropy 0.25–0.6)
  - fragile:   constructed to sit exactly between routes (expected entropy > 0.6)
"""
from dataclasses import dataclass
from typing import Literal


@dataclass
class TestInput:
    text: str
    category: Literal["confident", "boundary", "fragile"]
    note: str = ""


# --- Customer support inputs ---
CUSTOMER_SUPPORT_INPUTS: list[TestInput] = [
    # Confident
    TestInput(
        text="My order hasn't arrived and it's been 3 weeks, I want a refund immediately!",
        category="confident",
        note="Clear refund + escalation signal",
    ),
    TestInput(
        text="Can you tell me what your business hours are?",
        category="confident",
        note="Clear general info request",
    ),
    # Boundary
    TestInput(
        text="I'm not happy with my purchase, can something be done?",
        category="boundary",
        note="Ambiguous: refund or general complaint?",
    ),
    TestInput(
        text="There's an issue with my account I'd like help with.",
        category="boundary",
        note="Vague — could be billing, access, or info",
    ),
    # Fragile
    TestInput(
        text="Maybe I should return this, not sure yet.",
        category="fragile",
        note="Weak signal in both directions",
    ),
]

# --- ReAct agent inputs (tool-use routing) ---
REACT_AGENT_INPUTS: list[TestInput] = [
    # Confident
    TestInput(
        text="Search the web for the latest Python 3.13 release notes.",
        category="confident",
        note="Unambiguous web search required",
    ),
    TestInput(
        text="What is 2 + 2?",
        category="confident",
        note="No tool needed — direct answer",
    ),
    # Boundary
    TestInput(
        text="Tell me about recent developments in AI.",
        category="boundary",
        note="Could use search or answer from knowledge",
    ),
    # Fragile
    TestInput(
        text="What's happening in the news?",
        category="fragile",
        note="Extremely vague — search vs. decline",
    ),
]

# --- Multi-agent supervisor inputs ---
SUPERVISOR_INPUTS: list[TestInput] = [
    # Confident
    TestInput(
        text="Write a Python function to parse JSON.",
        category="confident",
        note="Clearly routes to code specialist",
    ),
    TestInput(
        text="Summarise this article about climate change.",
        category="confident",
        note="Clearly routes to writing specialist",
    ),
    # Boundary
    TestInput(
        text="Help me document this code.",
        category="boundary",
        note="Code + writing overlap",
    ),
    # Fragile
    TestInput(
        text="Can you help me with this?",
        category="fragile",
        note="Maximally ambiguous routing signal",
    ),
]

# --- Adaptive RAG inputs ---
ADAPTIVE_RAG_INPUTS: list[TestInput] = [
    # Confident
    TestInput(
        text="What is the boiling point of water at sea level?",
        category="confident",
        note="Factual — no retrieval needed",
    ),
    TestInput(
        text="Find documents in our knowledge base about Q4 2024 sales.",
        category="confident",
        note="Clear retrieval request",
    ),
    # Boundary
    TestInput(
        text="What do we know about customer churn?",
        category="boundary",
        note="Ambiguous: internal docs vs. general knowledge",
    ),
    # Fragile
    TestInput(
        text="Explain the situation.",
        category="fragile",
        note="No context — retrieval strategy undefined",
    ),
]

# Combined map for convenience
ALL_INPUTS: dict[str, list[TestInput]] = {
    "customer_support": CUSTOMER_SUPPORT_INPUTS,
    "react_agent": REACT_AGENT_INPUTS,
    "multi_agent_supervisor": SUPERVISOR_INPUTS,
    "adaptive_rag": ADAPTIVE_RAG_INPUTS,
}

# Expected entropy ranges per category (used in assertions)
ENTROPY_RANGES = {
    "confident": (0.0, 0.25),
    "boundary": (0.25, 0.60),
    "fragile": (0.60, 1.0),
}
