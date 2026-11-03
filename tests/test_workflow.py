import unittest
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END

from src.agent import (
    AgentState,
    classify_intent,
    should_continue,
    create_workflow,
)
from src.schemas import UserIntent, UpdateMemoryResponse


class TestWorkflow(unittest.TestCase):
    def test_workflow_structure_and_nodes(self):
        """Workflow should instantiate nodes and compile with checkpointer."""
        workflow = create_workflow()
        self.assertIsNotNone(workflow)
        # Check that workflow was compiled with a checkpointer
        self.assertIsNotNone(workflow.checkpointer)
        self.assertIsInstance(workflow.checkpointer, InMemorySaver)

    def test_routing_logic(self):
        """should_continue should route next_step correctly."""
        self.assertEqual(should_continue({"next_step": "qa_agent"}), "qa_agent")
        self.assertEqual(should_continue({"next_step": "summarization_agent"}), "summarization_agent")
        self.assertEqual(should_continue({"next_step": "calculation_agent"}), "calculation_agent")
        self.assertEqual(should_continue({"next_step": "end"}), "end")
        self.assertEqual(should_continue({}), "end")

    def test_classify_intent_node_qa(self):
        """classify_intent node should set next_step to qa_agent and track actions_taken."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = UserIntent(
            intent_type="qa",
            confidence=0.95,
            reasoning="Direct question about an invoice"
        )
        mock_llm.with_structured_output.return_value = mock_structured

        state: AgentState = {
            "messages": [HumanMessage(content="What is the invoice amount?")],
            "user_input": "What is the invoice amount?",
            "intent": None,
            "next_step": "classify_intent",
            "conversation_summary": "",
            "active_documents": [],
            "current_response": None,
            "tools_used": [],
            "session_id": "test-session",
            "user_id": "test-user",
            "actions_taken": [],
        }
        config = {"configurable": {"llm": mock_llm}}

        result = classify_intent(state, config)
        self.assertEqual(result["next_step"], "qa_agent")
        self.assertEqual(result["actions_taken"], ["classify_intent"])
        self.assertEqual(result["intent"].intent_type, "qa")

    def test_classify_intent_node_summarization(self):
        """classify_intent node should set next_step to summarization_agent."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = UserIntent(
            intent_type="summarization",
            confidence=0.98,
            reasoning="Requesting summary"
        )
        mock_llm.with_structured_output.return_value = mock_structured

        state: AgentState = {
            "messages": [],
            "user_input": "Summarize all contracts",
            "intent": None,
            "next_step": "classify_intent",
            "conversation_summary": "",
            "active_documents": [],
            "current_response": None,
            "tools_used": [],
            "session_id": "test-session",
            "user_id": "test-user",
            "actions_taken": [],
        }
        config = {"configurable": {"llm": mock_llm}}

        result = classify_intent(state, config)
        self.assertEqual(result["next_step"], "summarization_agent")
        self.assertEqual(result["actions_taken"], ["classify_intent"])

    def test_classify_intent_node_calculation(self):
        """classify_intent node should set next_step to calculation_agent."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = UserIntent(
            intent_type="calculation",
            confidence=0.99,
            reasoning="Requesting calculation"
        )
        mock_llm.with_structured_output.return_value = mock_structured

        state: AgentState = {
            "messages": [],
            "user_input": "Add up all invoice totals",
            "intent": None,
            "next_step": "classify_intent",
            "conversation_summary": "",
            "active_documents": [],
            "current_response": None,
            "tools_used": [],
            "session_id": "test-session",
            "user_id": "test-user",
            "actions_taken": [],
        }
        config = {"configurable": {"llm": mock_llm}}

        result = classify_intent(state, config)
        self.assertEqual(result["next_step"], "calculation_agent")
        self.assertEqual(result["actions_taken"], ["classify_intent"])

    def test_actions_taken_accumulation_reducer(self):
        """Verify operator.add accumulates actions_taken across graph execution."""
        def dummy_node_1(state: AgentState):
            return {"actions_taken": ["node_1"], "next_step": "node_2"}

        def dummy_node_2(state: AgentState):
            return {"actions_taken": ["node_2"], "next_step": "end"}

        builder = StateGraph(AgentState)
        builder.add_node("node_1", dummy_node_1)
        builder.add_node("node_2", dummy_node_2)
        builder.set_entry_point("node_1")
        builder.add_edge("node_1", "node_2")
        builder.add_edge("node_2", END)

        saver = InMemorySaver()
        app = builder.compile(checkpointer=saver)

        initial_state = {
            "messages": [],
            "user_input": "test",
            "intent": None,
            "next_step": "node_1",
            "conversation_summary": "",
            "active_documents": [],
            "current_response": None,
            "tools_used": [],
            "session_id": "thread-1",
            "user_id": "user-1",
            "actions_taken": ["start"],
        }
        config = {"configurable": {"thread_id": "thread-1"}}
        final = app.invoke(initial_state, config=config)

        # Due to operator.add, actions_taken must accumulate
        self.assertEqual(final["actions_taken"], ["start", "node_1", "node_2"])

    def test_qa_agent_node(self):
        """Verify qa_agent invokes react agent and returns state update."""
        from unittest.mock import patch
        from src.agent import qa_agent

        mock_result = {"messages": [AIMessage(content="Answer for QA")]}
        mock_tools_used = ["document_search"]

        with patch("src.agent.invoke_react_agent", return_value=(mock_result, mock_tools_used)):
            state: AgentState = {
                "messages": [],
                "user_input": "What is the cost?",
                "intent": None,
                "next_step": "qa_agent",
                "conversation_summary": "",
                "active_documents": [],
                "current_response": None,
                "tools_used": [],
                "session_id": "thread-qa",
                "user_id": "user-qa",
                "actions_taken": [],
            }
            config = {"configurable": {"llm": MagicMock(), "tools": []}}
            update = qa_agent(state, config)
            self.assertEqual(update["actions_taken"], ["qa_agent"])
            self.assertEqual(update["next_step"], "update_memory")
            self.assertEqual(update["tools_used"], ["document_search"])

    def test_summarization_agent_node(self):
        """Verify summarization_agent invokes react agent and returns state update."""
        from unittest.mock import patch
        from src.agent import summarization_agent

        mock_result = {"messages": [AIMessage(content="Summary of documents")]}
        mock_tools_used = ["document_reader"]

        with patch("src.agent.invoke_react_agent", return_value=(mock_result, mock_tools_used)):
            state: AgentState = {
                "messages": [],
                "user_input": "Summarize invoices",
                "intent": None,
                "next_step": "summarization_agent",
                "conversation_summary": "",
                "active_documents": [],
                "current_response": None,
                "tools_used": [],
                "session_id": "thread-sum",
                "user_id": "user-sum",
                "actions_taken": [],
            }
            config = {"configurable": {"llm": MagicMock(), "tools": []}}
            update = summarization_agent(state, config)
            self.assertEqual(update["actions_taken"], ["summarization_agent"])
            self.assertEqual(update["next_step"], "update_memory")
            self.assertEqual(update["tools_used"], ["document_reader"])

    def test_calculation_agent_node(self):
        """Verify calculation_agent invokes react agent and returns state update."""
        from unittest.mock import patch
        from src.agent import calculation_agent

        mock_result = {"messages": [AIMessage(content="Calculation result is 22000")]}
        mock_tools_used = ["calculator"]

        with patch("src.agent.invoke_react_agent", return_value=(mock_result, mock_tools_used)):
            state: AgentState = {
                "messages": [],
                "user_input": "Calculate total",
                "intent": None,
                "next_step": "calculation_agent",
                "conversation_summary": "",
                "active_documents": [],
                "current_response": None,
                "tools_used": [],
                "session_id": "thread-calc",
                "user_id": "user-calc",
                "actions_taken": [],
            }
            config = {"configurable": {"llm": MagicMock(), "tools": []}}
            update = calculation_agent(state, config)
            self.assertEqual(update["actions_taken"], ["calculation_agent"])
            self.assertEqual(update["next_step"], "update_memory")
            self.assertEqual(update["tools_used"], ["calculator"])

    def test_update_memory_node(self):
        """Verify update_memory extracts structured summary and active documents."""
        from src.agent import update_memory

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = UpdateMemoryResponse(
            summary="User asked about INV-001.",
            document_ids=["INV-001"],
        )
        mock_llm.with_structured_output.return_value = mock_structured

        state: AgentState = {
            "messages": [HumanMessage(content="INV-001"), AIMessage(content="Found")],
            "user_input": "INV-001",
            "intent": None,
            "next_step": "update_memory",
            "conversation_summary": "",
            "active_documents": [],
            "current_response": None,
            "tools_used": [],
            "session_id": "thread-mem",
            "user_id": "user-mem",
            "actions_taken": [],
        }
        config = {"configurable": {"llm": mock_llm}}
        update = update_memory(state, config)
        self.assertEqual(update["actions_taken"], ["update_memory"])
        self.assertEqual(update["next_step"], "end")
        self.assertEqual(update["conversation_summary"], "User asked about INV-001.")
        self.assertEqual(update["active_documents"], ["INV-001"])


if __name__ == "__main__":
    unittest.main()
