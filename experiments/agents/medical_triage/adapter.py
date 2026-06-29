"""
Medical Triage Chatbot adapter for the CPE-GEPA optimization loop.

Routes patient symptom descriptions to care pathways.
All inputs are synthetic — no real patient data used.

This agent deliberately covers a high-stakes routing domain where boundary cases
(severe headache: urgent vs emergency?) produce high entropy — useful for the paper's
analysis of CPE signal in safety-critical routing.

Routing labels: emergency_care | urgent_care | primary_care | self_care
"""
from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

DEFAULT_SYSTEM_PROMPT = """\
Route this patient symptom description to the appropriate care pathway.
Respond with only the pathway name, nothing else.

Categories:
- emergency_care: life-threatening symptoms — chest pain, difficulty breathing, stroke signs, loss of consciousness, severe bleeding
- urgent_care: serious but stable — high fever, severe pain, acute injury, conditions needing same-day evaluation
- primary_care: non-urgent — chronic conditions, follow-up visits, prescription renewals, routine consultations
- self_care: mild symptoms manageable at home — common cold, minor headache, small cuts, mild sore throat"""


class PromptHolder:
    def __init__(self, prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
        self.system_prompt = prompt


class TriageState(TypedDict):
    symptoms: str
    pathway: str | None
    result: str | None


def _get_llm(model=None):
    from conntrail.utils.providers import get_chat_model
    return get_chat_model(model or "claude-haiku-4-5-20251001", max_tokens=20)


def make_routing_node(holder: PromptHolder, model=None):
    async def triage_patient(state: TriageState) -> TriageState:
        llm = _get_llm(model)
        response = await llm.ainvoke([
            SystemMessage(content=holder.system_prompt),
            HumanMessage(content=state["symptoms"]),
        ])
        pathway = response.content.strip().lower()
        if pathway not in ("emergency_care", "urgent_care", "primary_care", "self_care"):
            pathway = "primary_care"
        return {**state, "pathway": pathway}
    triage_patient.__name__ = "triage_patient"
    return triage_patient


async def _handler(state: TriageState) -> TriageState:
    return {**state, "result": f"Routed to: {state['pathway']}"}


def _route(state: TriageState) -> str:
    return state["pathway"]


def build_graph(holder: PromptHolder | None = None, model=None):
    if holder is None:
        holder = PromptHolder()
    builder = StateGraph(TriageState)
    builder.add_node("triage_patient", make_routing_node(holder, model=model))
    for pathway in ("emergency_care", "urgent_care", "primary_care", "self_care"):
        builder.add_node(pathway, _handler)
        builder.add_edge(pathway, END)
    builder.add_edge(START, "triage_patient")
    builder.add_conditional_edges(
        "triage_patient", _route,
        {p: p for p in ("emergency_care", "urgent_care", "primary_care", "self_care")},
    )
    return builder.compile()


INPUT_KEY = "symptoms"
ROUTE_KEY = "pathway"
ONLY_NODES = {"triage_patient"}
VALID_ROUTES = ("emergency_care", "urgent_care", "primary_care", "self_care")


def make_initial_state(input_text: str) -> dict:
    return {"symptoms": input_text, "pathway": None, "result": None}


def get_result_route(result: dict) -> str:
    return result.get("pathway", "unknown") or "unknown"


TRAINSET = [
    # emergency_care
    {"input": "Severe chest pain radiating to my left arm, shortness of breath, profuse sweating for 20 minutes.", "expected_route": "emergency_care", "entropy_category": "confident"},
    {"input": "Patient is unresponsive and not breathing normally after collapsing suddenly in the street.", "expected_route": "emergency_care", "entropy_category": "confident"},
    {"input": "Sudden onset of the worst headache of my life, vision is blurring, stiff neck.", "expected_route": "emergency_care", "entropy_category": "confident"},
    {"input": "Throat rapidly swelling after eating shellfish — difficulty swallowing and breathing.", "expected_route": "emergency_care", "entropy_category": "confident"},
    {"input": "Face drooping on one side, arm suddenly weak, slurred speech — started 10 minutes ago.", "expected_route": "emergency_care", "entropy_category": "confident"},
    {"input": "Coughing up significant blood, severe difficulty breathing, lips are turning blue.", "expected_route": "emergency_care", "entropy_category": "confident"},
    # urgent_care
    {"input": "Fever of 104°F for two days, not responding to paracetamol, very lethargic.", "expected_route": "urgent_care", "entropy_category": "confident"},
    {"input": "Severe abdominal pain in lower right side, tender to touch, started 3 hours ago.", "expected_route": "urgent_care", "entropy_category": "confident"},
    {"input": "Deep laceration on forearm from a kitchen accident, still bleeding despite 15 minutes of pressure.", "expected_route": "urgent_care", "entropy_category": "confident"},
    {"input": "Wrist very swollen and bruised after a fall, can't move it at all, possible fracture.", "expected_route": "urgent_care", "entropy_category": "confident"},
    {"input": "Child has high fever with stiff neck and extreme sensitivity to light for 6 hours.", "expected_route": "urgent_care", "entropy_category": "boundary"},
    {"input": "Severe migraine with vomiting not responding to usual medication, going on 8 hours.", "expected_route": "urgent_care", "entropy_category": "boundary"},
    # primary_care
    {"input": "Chronic lower back pain for three months that is gradually worsening, need imaging referral.", "expected_route": "primary_care", "entropy_category": "confident"},
    {"input": "Need a prescription refill for my blood pressure medication, running low.", "expected_route": "primary_care", "entropy_category": "confident"},
    {"input": "Follow-up on diabetes management — last HbA1c was 7.4 and I want to review my diet plan.", "expected_route": "primary_care", "entropy_category": "confident"},
    {"input": "Annual physical exam overdue by six months, need routine bloodwork and checkup.", "expected_route": "primary_care", "entropy_category": "confident"},
    {"input": "Persistent dry cough for three weeks with no fever, not getting better on its own.", "expected_route": "primary_care", "entropy_category": "confident"},
    {"input": "Mild anxiety and difficulty sleeping for the past month, want to discuss options.", "expected_route": "primary_care", "entropy_category": "confident"},
    # self_care
    {"input": "Runny nose, sneezing, and mild congestion for two days — feels like a regular cold.", "expected_route": "self_care", "entropy_category": "confident"},
    {"input": "Mild headache behind the eyes after staring at screens all day, started this afternoon.", "expected_route": "self_care", "entropy_category": "confident"},
    {"input": "Small cut on my finger from cooking, already cleaned it — just want to confirm care steps.", "expected_route": "self_care", "entropy_category": "confident"},
    {"input": "Mild sore throat since this morning, no fever, just feels a bit scratchy.", "expected_route": "self_care", "entropy_category": "confident"},
    {"input": "Seasonal allergies acting up — itchy eyes and sneezing, happens every spring for me.", "expected_route": "self_care", "entropy_category": "confident"},
    {"input": "Mild upset stomach after a big meal, some bloating, no vomiting or fever.", "expected_route": "self_care", "entropy_category": "confident"},
]

TRAINSET_LARGE = TRAINSET
