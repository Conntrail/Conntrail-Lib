"""
Financial Query Router adapter for the CPE-GEPA optimization loop.

Routes financial support queries to appropriate handling teams.
Emotionally charged inputs ("someone stole my money!") deliberately test whether
semantic intensity attribution appears — useful for the paper's attribution_dimension analysis.

Routing labels: fraud_investigation | account_services | investment_advice | general_inquiry
"""
from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

DEFAULT_SYSTEM_PROMPT = """\
Route this financial customer query to the appropriate team.
Respond with only the category name, nothing else.

Categories:
- fraud_investigation: unauthorized transactions, suspicious activity, account compromise, card theft
- account_services: balance inquiries, card replacement, address updates, transfers, account access
- investment_advice: portfolio questions, product recommendations, investment strategies, financial planning
- general_inquiry: product information, branch hours, fee questions, account opening, onboarding"""


class PromptHolder:
    def __init__(self, prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
        self.system_prompt = prompt


class FinancialState(TypedDict):
    query: str
    department: str | None
    result: str | None


def _get_llm(model=None):
    from conntrail.utils.providers import get_chat_model
    return get_chat_model(model or "claude-haiku-4-5-20251001", max_tokens=20)


def make_routing_node(holder: PromptHolder, model=None):
    async def route_query(state: FinancialState) -> FinancialState:
        llm = _get_llm(model)
        response = await llm.ainvoke([
            SystemMessage(content=holder.system_prompt),
            HumanMessage(content=state["query"]),
        ])
        department = response.content.strip().lower()
        if department not in ("fraud_investigation", "account_services", "investment_advice", "general_inquiry"):
            department = "general_inquiry"
        return {**state, "department": department}
    route_query.__name__ = "route_query"
    return route_query


async def _handler(state: FinancialState) -> FinancialState:
    return {**state, "result": f"Routed to: {state['department']}"}


def _route(state: FinancialState) -> str:
    return state["department"]


def build_graph(holder: PromptHolder | None = None, model=None):
    if holder is None:
        holder = PromptHolder()
    builder = StateGraph(FinancialState)
    builder.add_node("route_query", make_routing_node(holder, model=model))
    for dept in ("fraud_investigation", "account_services", "investment_advice", "general_inquiry"):
        builder.add_node(dept, _handler)
        builder.add_edge(dept, END)
    builder.add_edge(START, "route_query")
    builder.add_conditional_edges(
        "route_query", _route,
        {d: d for d in ("fraud_investigation", "account_services", "investment_advice", "general_inquiry")},
    )
    return builder.compile()


INPUT_KEY = "query"
ROUTE_KEY = "department"
ONLY_NODES = {"route_query"}
VALID_ROUTES = ("fraud_investigation", "account_services", "investment_advice", "general_inquiry")


def make_initial_state(input_text: str) -> dict:
    return {"query": input_text, "department": None, "result": None}


def get_result_route(result: dict) -> str:
    return result.get("department", "unknown") or "unknown"


TRAINSET = [
    # fraud_investigation
    {"input": "There are two charges on my card from last night that I did not make — from a merchant I've never visited.", "expected_route": "fraud_investigation", "entropy_category": "confident"},
    {"input": "My account was accessed from a foreign country I've never visited and I did not authorize this.", "expected_route": "fraud_investigation", "entropy_category": "confident"},
    {"input": "Someone stole my wallet including my bank card. Please freeze my account immediately.", "expected_route": "fraud_investigation", "entropy_category": "confident"},
    {"input": "A wire transfer of $3,200 left my account this morning that I absolutely did not initiate.", "expected_route": "fraud_investigation", "entropy_category": "confident"},
    {"input": "Someone stole all my money! There are withdrawals I didn't make — my account was drained overnight.", "expected_route": "fraud_investigation", "entropy_category": "boundary"},
    # account_services
    {"input": "Can you tell me my current account balance and last five transactions?", "expected_route": "account_services", "entropy_category": "confident"},
    {"input": "My debit card went through the washing machine and is damaged. I need a replacement sent out.", "expected_route": "account_services", "entropy_category": "confident"},
    {"input": "I recently moved and need to update my mailing address on file for statements.", "expected_route": "account_services", "entropy_category": "confident"},
    {"input": "I'm locked out of online banking after too many failed login attempts. Please reset my access.", "expected_route": "account_services", "entropy_category": "confident"},
    {"input": "How do I set up a recurring monthly transfer from my checking to my savings account?", "expected_route": "account_services", "entropy_category": "confident"},
    # investment_advice
    {"input": "I have $50,000 to invest for retirement in 20 years. What funds would you recommend?", "expected_route": "investment_advice", "entropy_category": "confident"},
    {"input": "My portfolio is heavily weighted in tech stocks. How should I diversify to reduce risk?", "expected_route": "investment_advice", "entropy_category": "confident"},
    {"input": "Is now a good time to move my savings into index funds given current market conditions?", "expected_route": "investment_advice", "entropy_category": "confident"},
    {"input": "What are the management fees and historical returns on your ESG investment products?", "expected_route": "investment_advice", "entropy_category": "confident"},
    {"input": "I'm a first-time investor and want to compare your ETF options for a low-risk approach.", "expected_route": "investment_advice", "entropy_category": "boundary"},
    # general_inquiry
    {"input": "I'd like to open a new joint savings account for me and my spouse. What are the steps?", "expected_route": "general_inquiry", "entropy_category": "confident"},
    {"input": "What are your branch and phone support hours on weekends?", "expected_route": "general_inquiry", "entropy_category": "confident"},
    {"input": "What is the fee for sending an international wire transfer?", "expected_route": "general_inquiry", "entropy_category": "boundary"},
    {"input": "Do you offer student banking accounts with no monthly fees for under-25s?", "expected_route": "general_inquiry", "entropy_category": "confident"},
    {"input": "I forgot my username — how do I recover access to my online account?", "expected_route": "general_inquiry", "entropy_category": "fragile"},
]

TRAINSET_LARGE = TRAINSET
