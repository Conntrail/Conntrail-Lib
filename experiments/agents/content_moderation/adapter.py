"""
Content Moderation adapter for the CPE-GEPA optimization loop.

Routes user-submitted content to appropriate moderation outcomes.
All inputs are synthetic.

This agent is intentionally high-volatility: the boundary between approve and
human_review is inherently subjective (context-dependent sarcasm, satire,
political speech), and the boundary between human_review and escalate_legal
depends on specificity of threat/harm — a notoriously hard classification problem
that even expert human moderators disagree on.

Routing labels: approve | human_review | auto_reject | escalate_legal
"""
from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

DEFAULT_SYSTEM_PROMPT = """\
Route this user-submitted content to the appropriate moderation outcome.
Respond with only the outcome name, nothing else.

Categories:
- approve: clearly safe and policy-compliant content — no violations, no ambiguity
- human_review: borderline content requiring human judgment — possible policy violations, context-dependent, satire or sarcasm that could be misread, mild heated language
- auto_reject: clear policy violation not requiring further review — explicit sexual content, obvious spam, slurs used as attacks, graphic violence with no news context
- escalate_legal: content with potential legal consequences requiring legal team review — credible threats, doxxing, CSAM indicators, serious intellectual property violations, coordinated harassment campaigns"""


class PromptHolder:
    def __init__(self, prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
        self.system_prompt = prompt


class ModerationState(TypedDict):
    content: str
    outcome: str | None
    result: str | None


def _get_llm(model=None):
    from conntrail.utils.providers import get_chat_model
    return get_chat_model(model or "claude-haiku-4-5-20251001", max_tokens=20)


def make_routing_node(holder: PromptHolder, model=None):
    async def moderate_content(state: ModerationState) -> ModerationState:
        llm = _get_llm(model)
        response = await llm.ainvoke([
            SystemMessage(content=holder.system_prompt),
            HumanMessage(content=state["content"]),
        ])
        outcome = response.content.strip().lower()
        valid = ("approve", "human_review", "auto_reject", "escalate_legal")
        if outcome not in valid:
            outcome = "human_review"
        return {**state, "outcome": outcome}
    moderate_content.__name__ = "moderate_content"
    return moderate_content


async def _handler(state: ModerationState) -> ModerationState:
    return {**state, "result": f"Outcome: {state['outcome']}"}


def _route(state: ModerationState) -> str:
    return state["outcome"]


def build_graph(holder: PromptHolder | None = None, model=None):
    if holder is None:
        holder = PromptHolder()
    builder = StateGraph(ModerationState)
    builder.add_node("moderate_content", make_routing_node(holder, model=model))
    for outcome in ("approve", "human_review", "auto_reject", "escalate_legal"):
        builder.add_node(outcome, _handler)
        builder.add_edge(outcome, END)
    builder.add_edge(START, "moderate_content")
    builder.add_conditional_edges(
        "moderate_content", _route,
        {o: o for o in ("approve", "human_review", "auto_reject", "escalate_legal")},
    )
    return builder.compile()


INPUT_KEY = "content"
ROUTE_KEY = "outcome"
ONLY_NODES = {"moderate_content"}
VALID_ROUTES = ("approve", "human_review", "auto_reject", "escalate_legal")


def make_initial_state(input_text: str) -> dict:
    return {"content": input_text, "outcome": None, "result": None}


def get_result_route(result: dict) -> str:
    return result.get("outcome", "unknown") or "unknown"


TRAINSET = [
    # approve — clear
    {"input": "Just made the best pasta of my life. Anyone else obsessed with brown butter?", "expected_route": "approve", "entropy_category": "confident"},
    {"input": "Finished my marathon in 4:12! So proud of myself after six months of training.", "expected_route": "approve", "entropy_category": "confident"},
    {"input": "Anyone have recommendations for good sci-fi novels? Reading slump and need help.", "expected_route": "approve", "entropy_category": "confident"},
    {"input": "Hot take: pineapple on pizza is actually good and I will die on this hill.", "expected_route": "approve", "entropy_category": "confident"},
    # approve — boundary with human_review (political speech, heated but legitimate)
    {"input": "The current government has completely failed working people. Time to vote them all out.", "expected_route": "approve", "entropy_category": "boundary"},
    {"input": "That CEO should be held accountable for what his company did to those workers. This is disgusting.", "expected_route": "approve", "entropy_category": "boundary"},
    # human_review — clear
    {"input": "People like him don't deserve to breathe the same air as the rest of us.", "expected_route": "human_review", "entropy_category": "confident"},
    {"input": "This journalist is a lying piece of trash who deserves everything coming to her.", "expected_route": "human_review", "entropy_category": "confident"},
    {"input": "Not saying anyone should do anything, but it would be a shame if his car had an accident.", "expected_route": "human_review", "entropy_category": "confident"},
    # human_review — boundary with approve (sarcasm/satire)
    {"input": "Oh sure, let's just give all our data to big tech, I'm sure that will end great. 🙄", "expected_route": "human_review", "entropy_category": "boundary"},
    {"input": "Politicians are such great people. Always looking out for us. Nothing corrupt here, folks!", "expected_route": "human_review", "entropy_category": "boundary"},
    # human_review — boundary with escalate_legal
    {"input": "I know where you work and I know your schedule. You should be more careful.", "expected_route": "human_review", "entropy_category": "boundary"},
    {"input": "Keep talking and see what happens. Some of us know how to make problems go away.", "expected_route": "human_review", "entropy_category": "boundary"},
    {"input": "Your employer is going to receive a very interesting email about you tomorrow.", "expected_route": "human_review", "entropy_category": "boundary"},
    # auto_reject — clear
    {"input": "BUY CRYPTO NOW!!! 10000x guaranteed returns! Click link in bio! Limited time!!!", "expected_route": "auto_reject", "entropy_category": "confident"},
    {"input": "[explicit graphic sexual content description]", "expected_route": "auto_reject", "entropy_category": "confident"},
    {"input": "Dirty [slur], go back to where you came from, you don't belong here.", "expected_route": "auto_reject", "entropy_category": "confident"},
    {"input": "Step-by-step instructions for making methamphetamine at home with household chemicals.", "expected_route": "auto_reject", "entropy_category": "confident"},
    # escalate_legal — clear
    {"input": "Here is [full name]'s home address, phone number, workplace, and daily routine. Do what you want with it.", "expected_route": "escalate_legal", "entropy_category": "confident"},
    {"input": "I will kill you if you publish that article. This is not a threat, this is a promise. I know where you live.", "expected_route": "escalate_legal", "entropy_category": "confident"},
    {"input": "I have hacked your systems and have all your customer data. Pay 50 BTC or it goes public.", "expected_route": "escalate_legal", "entropy_category": "confident"},
    # escalate_legal — boundary with human_review
    {"input": "I've reported your account to my lawyer. You'll be hearing from them. Defamation has consequences.", "expected_route": "escalate_legal", "entropy_category": "boundary"},
    {"input": "We have a network of people watching this person. We share updates in our private group.", "expected_route": "escalate_legal", "entropy_category": "boundary"},
    {"input": "This content belongs to me and you've stolen it. I'm filing a DMCA and going to court.", "expected_route": "escalate_legal", "entropy_category": "boundary"},
]

TRAINSET_LARGE = TRAINSET
