"""
Phase 0 baseline tests — verify all 4 agents run correctly without Conntrail.

These tests must pass before any Conntrail development begins.
They confirm our test subjects are functioning correctly as a clean baseline.
"""
import pytest
from testing.harness.fixtures import ALL_INPUTS


@pytest.mark.baseline
@pytest.mark.integration
class TestCustomerSupportBaseline:
    """Customer support agent runs and routes correctly without Conntrail."""

    @pytest.fixture(scope="class")
    def graph(self):
        from testing.agents.customer_support.agent import build_graph
        return build_graph()

    @pytest.mark.parametrize("test_input", ALL_INPUTS["customer_support"])
    async def test_runs_without_error(self, graph, test_input):
        result = await graph.ainvoke({
            "message": test_input.text,
            "category": None,
            "response": None,
        })
        assert result["category"] in ("refund", "escalation", "order_info", "general")
        assert result["response"] is not None
        assert len(result["response"]) > 0

    async def test_confident_refund_routes_correctly(self, graph):
        result = await graph.ainvoke({
            "message": "I want a refund for my broken order #12345",
            "category": None,
            "response": None,
        })
        assert result["category"] == "refund"

    async def test_urgent_escalates(self, graph):
        result = await graph.ainvoke({
            "message": "I am furious! I want to speak to a manager RIGHT NOW!",
            "category": None,
            "response": None,
        })
        assert result["category"] == "escalation"


@pytest.mark.baseline
@pytest.mark.integration
class TestReActAgentBaseline:
    """ReAct agent runs and routes correctly without Conntrail."""

    @pytest.fixture(scope="class")
    def graph(self):
        from testing.agents.react_agent.agent import build_graph
        return build_graph()

    async def test_calculator_tool_used(self, graph):
        from langchain_core.messages import HumanMessage
        result = await graph.ainvoke({
            "messages": [HumanMessage(content="What is 42 * 17?")]
        })
        # Should use calculator tool — last message should contain 714
        last_msg = result["messages"][-1].content
        assert "714" in last_msg

    async def test_direct_answer_no_tool(self, graph):
        from langchain_core.messages import HumanMessage
        result = await graph.ainvoke({
            "messages": [HumanMessage(content="What is the capital of France?")]
        })
        last_msg = result["messages"][-1].content
        assert "paris" in last_msg.lower()


@pytest.mark.baseline
@pytest.mark.integration
class TestSupervisorBaseline:
    """Multi-agent supervisor routes to correct specialist without Conntrail."""

    @pytest.fixture(scope="class")
    def graph(self):
        from testing.agents.multi_agent_supervisor.agent import build_graph
        return build_graph()

    async def test_code_task_routes_to_code_agent(self, graph):
        result = await graph.ainvoke({
            "task": "Write a Python function to calculate fibonacci numbers.",
            "assigned_agent": None,
            "result": None,
        })
        assert result["assigned_agent"] == "code_agent"
        assert result["result"] is not None

    async def test_writing_task_routes_to_writing_agent(self, graph):
        result = await graph.ainvoke({
            "task": "Write a professional email declining a meeting invitation.",
            "assigned_agent": None,
            "result": None,
        })
        assert result["assigned_agent"] == "writing_agent"


@pytest.mark.baseline
@pytest.mark.integration
class TestAdaptiveRAGBaseline:
    """Adaptive RAG agent selects correct retrieval strategy without Conntrail."""

    @pytest.fixture(scope="class")
    def graph(self):
        from testing.agents.adaptive_rag.agent import build_graph
        return build_graph()

    async def test_internal_query_uses_vector_search(self, graph):
        result = await graph.ainvoke({
            "query": "What were Q4 2024 sales results?",
            "strategy": None,
            "retrieved_docs": [],
            "answer": None,
        })
        assert result["strategy"] == "vector_search"
        assert result["answer"] is not None

    async def test_factual_query_uses_direct_answer(self, graph):
        result = await graph.ainvoke({
            "query": "What is the boiling point of water?",
            "strategy": None,
            "retrieved_docs": [],
            "answer": None,
        })
        assert result["strategy"] == "direct_answer"
