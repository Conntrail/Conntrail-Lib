"""
Mental Health Triage adapter for the CPE-GEPA optimization loop.

Routes mental health disclosures to appropriate support pathways.
All inputs are synthetic — no real patient data used.

This agent is intentionally high-volatility: the boundary between
crisis_intervention and urgent_support, and between urgent_support and
general_support, is genuinely ambiguous in clinical practice. Phrases like
"I don't want to be here anymore" can mean very different things in different
contexts — making this one of the highest-entropy routing tasks in the literature.

Routing labels: crisis_intervention | urgent_support | general_support | resource_referral
"""
from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

DEFAULT_SYSTEM_PROMPT = """\
Route this mental health disclosure to the appropriate support pathway.
Respond with only the pathway name, nothing else.

Categories:
- crisis_intervention: active suicidal ideation with intent or plan, immediate self-harm risk, acute psychosis with danger to self or others — needs immediate intervention
- urgent_support: severe distress without confirmed immediate danger — panic attacks, acute emotional crisis, passive suicidal ideation without plan, needs same-day contact
- general_support: ongoing mental health struggles manageable with regular support — depression, anxiety, relationship difficulties, grief, burnout without acute risk
- resource_referral: stable but needs professional services — medication review, therapy referral, support groups, long-term care coordination"""


class PromptHolder:
    def __init__(self, prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
        self.system_prompt = prompt


class TriageState(TypedDict):
    disclosure: str
    pathway: str | None
    result: str | None


def _get_llm(model=None):
    from conntrail.utils.providers import get_chat_model
    return get_chat_model(model or "claude-haiku-4-5-20251001", max_tokens=20)


def make_routing_node(holder: PromptHolder, model=None):
    async def triage_disclosure(state: TriageState) -> TriageState:
        llm = _get_llm(model)
        response = await llm.ainvoke([
            SystemMessage(content=holder.system_prompt),
            HumanMessage(content=state["disclosure"]),
        ])
        pathway = response.content.strip().lower()
        valid = ("crisis_intervention", "urgent_support", "general_support", "resource_referral")
        if pathway not in valid:
            pathway = "general_support"
        return {**state, "pathway": pathway}
    triage_disclosure.__name__ = "triage_disclosure"
    return triage_disclosure


async def _handler(state: TriageState) -> TriageState:
    return {**state, "result": f"Routed to: {state['pathway']}"}


def _route(state: TriageState) -> str:
    return state["pathway"]


def build_graph(holder: PromptHolder | None = None, model=None):
    if holder is None:
        holder = PromptHolder()
    builder = StateGraph(TriageState)
    builder.add_node("triage_disclosure", make_routing_node(holder, model=model))
    for pathway in ("crisis_intervention", "urgent_support", "general_support", "resource_referral"):
        builder.add_node(pathway, _handler)
        builder.add_edge(pathway, END)
    builder.add_edge(START, "triage_disclosure")
    builder.add_conditional_edges(
        "triage_disclosure", _route,
        {p: p for p in ("crisis_intervention", "urgent_support", "general_support", "resource_referral")},
    )
    return builder.compile()


INPUT_KEY = "disclosure"
ROUTE_KEY = "pathway"
ONLY_NODES = {"triage_disclosure"}
VALID_ROUTES = ("crisis_intervention", "urgent_support", "general_support", "resource_referral")


def make_initial_state(input_text: str) -> dict:
    return {"disclosure": input_text, "pathway": None, "result": None}


def get_result_route(result: dict) -> str:
    return result.get("pathway", "unknown") or "unknown"


TRAINSET = [
    # crisis_intervention — clear
    {"input": "I have pills in my hand right now. I've decided tonight is the night.", "expected_route": "crisis_intervention", "entropy_category": "confident"},
    {"input": "I've written my goodbye letters and I'm not planning to be here tomorrow.", "expected_route": "crisis_intervention", "entropy_category": "confident"},
    {"input": "I'm standing on the roof of my building right now, I just want it to stop.", "expected_route": "crisis_intervention", "entropy_category": "confident"},
    {"input": "I have a loaded gun and I've been sitting with it for an hour trying to decide.", "expected_route": "crisis_intervention", "entropy_category": "confident"},
    # crisis_intervention — boundary with urgent_support
    {"input": "I've been cutting myself every night this week and I'm scared I'm going to go too far.", "expected_route": "crisis_intervention", "entropy_category": "boundary"},
    {"input": "I keep fantasising about stepping in front of traffic. I haven't done anything but it's getting harder to resist.", "expected_route": "crisis_intervention", "entropy_category": "boundary"},
    # urgent_support — clear
    {"input": "I haven't slept in four days, I'm having a complete breakdown, I can't stop shaking and crying.", "expected_route": "urgent_support", "entropy_category": "confident"},
    {"input": "I'm having a severe panic attack, my chest is tight, I can't breathe, I don't know what's happening.", "expected_route": "urgent_support", "entropy_category": "confident"},
    {"input": "My partner just left me and I feel completely out of control — I'm scared of what I might do to myself.", "expected_route": "urgent_support", "entropy_category": "confident"},
    # urgent_support — boundary with crisis
    {"input": "I don't want to be here anymore. I don't have a plan but I think about it every single day.", "expected_route": "urgent_support", "entropy_category": "boundary"},
    {"input": "Sometimes I think everyone would be better off without me. I wouldn't act on it but the thought won't leave.", "expected_route": "urgent_support", "entropy_category": "boundary"},
    {"input": "I've been having thoughts of hurting myself but I'd never actually do it — I just don't know how much longer I can cope.", "expected_route": "urgent_support", "entropy_category": "boundary"},
    # urgent_support — boundary with general
    {"input": "I haven't been able to get out of bed for a week. I'm not eating. I feel completely hollow.", "expected_route": "urgent_support", "entropy_category": "boundary"},
    {"input": "I'm so exhausted I can't function. I've cried every day for two weeks and I don't know why.", "expected_route": "urgent_support", "entropy_category": "boundary"},
    # general_support — clear
    {"input": "I've been feeling low and unmotivated for a few months. I think I might be depressed but I'm managing okay.", "expected_route": "general_support", "entropy_category": "confident"},
    {"input": "Work stress is getting to me and my relationship is strained. I'd like to talk it through.", "expected_route": "general_support", "entropy_category": "confident"},
    {"input": "I lost my mum six months ago and I'm still grieving. Some days are harder than others.", "expected_route": "general_support", "entropy_category": "confident"},
    {"input": "I struggle with social anxiety — meeting new people is really hard for me and I want strategies.", "expected_route": "general_support", "entropy_category": "confident"},
    # general_support — boundary with resource_referral
    {"input": "I've been managing my anxiety with mindfulness but it's not enough anymore. I think I need proper help.", "expected_route": "general_support", "entropy_category": "boundary"},
    # resource_referral — clear
    {"input": "My current antidepressant isn't working well. I want to speak to someone about switching medications.", "expected_route": "resource_referral", "entropy_category": "confident"},
    {"input": "I've been stable for a while and want to find a long-term therapist to work through childhood trauma.", "expected_route": "resource_referral", "entropy_category": "confident"},
    {"input": "I think I might have ADHD — I want to be assessed and understand my options for support.", "expected_route": "resource_referral", "entropy_category": "confident"},
    {"input": "I'm doing okay but my psychiatrist retired and I need to transfer my care to someone new.", "expected_route": "resource_referral", "entropy_category": "confident"},
    {"input": "I'd like to join a support group for people with bipolar disorder — can you help me find one?", "expected_route": "resource_referral", "entropy_category": "confident"},
]

TRAINSET_LARGE = TRAINSET
