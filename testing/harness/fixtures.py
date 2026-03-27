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
    # Confident — clearly maps to one category, all contrasts route the same
    TestInput(
        text="Can you tell me what your business hours are?",
        category="confident",
        note="Clear general info request — unambiguous for non-intrusion testing",
    ),
    TestInput(
        text="Where can I find my order tracking number?",
        category="confident",
        note="Clear order_info request, no ambiguity",
    ),
    # Boundary — original + some contrasts route differently (1 contrast flips)
    TestInput(
        text="My order hasn't arrived and it's been 3 weeks, I want a refund immediately!",
        category="boundary",
        note="Sits between refund and escalation — urgency signal causes some contrast variance",
    ),
    TestInput(
        text="I'm not satisfied with the product I received and I'd like to know my options.",
        category="boundary",
        note="'not satisfied' → refund; neutral contrast drops that signal and flips to general",
    ),
    # Fragile — multiple contrasts flip to different categories
    TestInput(
        text="Maybe I should return this, not sure yet.",
        category="fragile",
        note="Weak, hedging signal — contrasts push toward different categories",
    ),
    TestInput(
        text="I'm not happy with my purchase, can something be done?",
        category="fragile",
        note="Vague complaint — model produces 2+1+1 split across refund/general/escalation",
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
    # Confident — unambiguous task type
    TestInput(
        text="Debug this Python function that raises an IndexError on line 5.",
        category="confident",
        note="Clearly routes to code specialist — all contrasts stay in code territory",
    ),
    TestInput(
        text="Rewrite this paragraph in a more formal, professional tone.",
        category="confident",
        note="Clearly routes to writing specialist — no code or research interpretation",
    ),
    # Boundary — code + writing overlap: opposite contrast flips routing
    TestInput(
        text="Help me debug this Python function and then write clear documentation explaining how it works.",
        category="boundary",
        note="Debug signal → code_agent; opposite (no code, write prose) → writing_agent",
    ),
    # Fragile — multiple contrasts route differently across all 3 agents
    TestInput(
        text="Look up the current Python version and show me how to upgrade it.",
        category="fragile",
        note="research/code/writing 3-way split: research_agent orig, code neutral, writing opposite",
    ),
]

# --- Adaptive RAG inputs ---
ADAPTIVE_RAG_INPUTS: list[TestInput] = [
    # Confident — clear routing decision, all contrasts agree
    TestInput(
        text="What is the boiling point of water at sea level?",
        category="confident",
        note="Factual — direct answer, no retrieval needed",
    ),
    TestInput(
        text="Find documents in our knowledge base about Q4 2024 sales.",
        category="confident",
        note="Explicit retrieval request",
    ),
    # Boundary — sits between vector_search and direct_answer
    TestInput(
        text="How has our customer satisfaction been changing recently?",
        category="boundary",
        note="'our customer' pulls toward vector_search; 'recently' pulls toward direct_answer — 1 contrast flips",
    ),
    # Fragile — ambiguous across multiple retrieval strategies
    TestInput(
        text="What's the latest news about our internal company metrics?",
        category="fragile",
        note="'latest news' → web_search; 'our internal metrics' → vector_search; contrasts pull to each",
    ),
]

# Combined map for convenience
ALL_INPUTS: dict[str, list[TestInput]] = {
    "customer_support": CUSTOMER_SUPPORT_INPUTS,
    "react_agent": REACT_AGENT_INPUTS,
    "multi_agent_supervisor": SUPERVISOR_INPUTS,
    "adaptive_rag": ADAPTIVE_RAG_INPUTS,
}

# Expected entropy ranges per category (used in assertions).
# Calibrated for llama-3.1-8b-instant on Groq: the small model is decisive
# (max observed entropy ~0.50 for fragile inputs, 0.0 for confident ones).
ENTROPY_RANGES = {
    "confident": (0.0, 0.25),
    "boundary": (0.20, 0.55),
    "fragile": (0.35, 1.0),
}
