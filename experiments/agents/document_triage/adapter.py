"""
Document Triage adapter for the CPE-GEPA optimization loop.

Routes incoming documents to processing pipelines based on document type and urgency.
Routing labels: legal_review | compliance_check | standard_processing | urgent_escalation
"""
from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

DEFAULT_SYSTEM_PROMPT = """\
Classify the incoming document into exactly one processing category.
Respond with only the category name, nothing else.

Categories:
- legal_review: contracts, NDAs, litigation notices, legal disputes, employment agreements
- compliance_check: regulatory filings, audit reports, compliance assessments, policy updates
- standard_processing: internal memos, status reports, meeting notes, newsletters, routine updates
- urgent_escalation: emergencies, data breaches, board notices, enforcement actions, critical incidents"""


class PromptHolder:
    def __init__(self, prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
        self.system_prompt = prompt


class TriageState(TypedDict):
    document: str
    category: str | None
    result: str | None


def _get_llm(model=None):
    from conntrail.utils.providers import get_chat_model
    return get_chat_model(model or "claude-haiku-4-5-20251001", max_tokens=20)


def make_routing_node(holder: PromptHolder, model=None):
    async def triage_document(state: TriageState) -> TriageState:
        llm = _get_llm(model)
        response = await llm.ainvoke([
            SystemMessage(content=holder.system_prompt),
            HumanMessage(content=state["document"]),
        ])
        category = response.content.strip().lower()
        if category not in ("legal_review", "compliance_check", "standard_processing", "urgent_escalation"):
            category = "standard_processing"
        return {**state, "category": category}
    triage_document.__name__ = "triage_document"
    return triage_document


async def _handler(state: TriageState) -> TriageState:
    return {**state, "result": f"Routed to: {state['category']}"}


def _route(state: TriageState) -> str:
    return state["category"]


def build_graph(holder: PromptHolder | None = None, model=None):
    if holder is None:
        holder = PromptHolder()
    builder = StateGraph(TriageState)
    builder.add_node("triage_document", make_routing_node(holder, model=model))
    for cat in ("legal_review", "compliance_check", "standard_processing", "urgent_escalation"):
        builder.add_node(cat, _handler)
        builder.add_edge(cat, END)
    builder.add_edge(START, "triage_document")
    builder.add_conditional_edges(
        "triage_document", _route,
        {c: c for c in ("legal_review", "compliance_check", "standard_processing", "urgent_escalation")},
    )
    return builder.compile()


INPUT_KEY = "document"
ROUTE_KEY = "category"
ONLY_NODES = {"triage_document"}
VALID_ROUTES = ("legal_review", "compliance_check", "standard_processing", "urgent_escalation")


def make_initial_state(input_text: str) -> dict:
    return {"document": input_text, "category": None, "result": None}


def get_result_route(result: dict) -> str:
    return result.get("category", "unknown") or "unknown"


TRAINSET = [
    # legal_review
    {"input": "NDA Agreement — Third-party vendor NDA with IP clauses awaiting legal signature.", "expected_route": "legal_review", "entropy_category": "confident"},
    {"input": "Litigation Notice — Cease and desist letter received from competitor alleging patent infringement.", "expected_route": "legal_review", "entropy_category": "confident"},
    {"input": "Master Service Agreement — New enterprise client MSA requires legal review before signing.", "expected_route": "legal_review", "entropy_category": "confident"},
    {"input": "Contract Amendment — Supplier contract modification changing payment terms and liability clauses.", "expected_route": "legal_review", "entropy_category": "confident"},
    {"input": "Executive Employment Agreement — Non-compete, equity vesting, and IP assignment require legal sign-off.", "expected_route": "legal_review", "entropy_category": "boundary"},
    # compliance_check
    {"input": "SOX Internal Audit Report — Q3 controls testing results for Sarbanes-Oxley compliance.", "expected_route": "compliance_check", "entropy_category": "confident"},
    {"input": "GDPR Data Processing Assessment — Annual review of EU data processing activities.", "expected_route": "compliance_check", "entropy_category": "confident"},
    {"input": "SEC Annual Disclosure Filing — Mandatory regulatory submission for public company transparency.", "expected_route": "compliance_check", "entropy_category": "confident"},
    {"input": "AML Policy Update — Anti-money laundering procedures revised per new FinCEN guidance.", "expected_route": "compliance_check", "entropy_category": "boundary"},
    {"input": "Environmental Compliance Audit — EPA documentation for manufacturing facility annual review.", "expected_route": "compliance_check", "entropy_category": "confident"},
    # standard_processing
    {"input": "Q3 Sales Performance Summary — Monthly results from regional sales team leads.", "expected_route": "standard_processing", "entropy_category": "confident"},
    {"input": "Sprint Retrospective Notes — Meeting minutes from last week's product review session.", "expected_route": "standard_processing", "entropy_category": "confident"},
    {"input": "Facility Closure Reminder — Office maintenance scheduled next Thursday, all staff affected.", "expected_route": "standard_processing", "entropy_category": "confident"},
    {"input": "Monthly Company Newsletter — Employee spotlight, upcoming events, and Q4 roadmap highlights.", "expected_route": "standard_processing", "entropy_category": "confident"},
    {"input": "Project Status Update — Weekly software development milestone report for stakeholder review.", "expected_route": "standard_processing", "entropy_category": "confident"},
    # urgent_escalation
    {"input": "DATA BREACH ALERT — Customer PII exposed in production database. Immediate executive response required.", "expected_route": "urgent_escalation", "entropy_category": "confident"},
    {"input": "Emergency Board Meeting — Directors convened for tomorrow regarding unsolicited acquisition proposal.", "expected_route": "urgent_escalation", "entropy_category": "confident"},
    {"input": "Production Outage — All customer-facing services down, 10,000 users affected. CEO notified.", "expected_route": "urgent_escalation", "entropy_category": "confident"},
    {"input": "Regulatory Enforcement Action — Agency audit notice with mandatory 48-hour response window.", "expected_route": "urgent_escalation", "entropy_category": "boundary"},
    {"input": "Critical Security Advisory — Zero-day exploit in customer-facing application, immediate patch required.", "expected_route": "urgent_escalation", "entropy_category": "fragile"},
]

TRAINSET_LARGE = TRAINSET
