import os
import shutil
import unittest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage

from src.assistant import DocumentAssistant
from src.schemas import UserIntent, AnswerResponse, UpdateMemoryResponse


class TestAssistantIntegration(unittest.TestCase):
    def setUp(self):
        self.session_dir = "./sessions_test"
        self.logs_dir = "./logs_test"
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.session_dir):
            shutil.rmtree(self.session_dir, ignore_errors=True)
        if os.path.exists(self.logs_dir):
            shutil.rmtree(self.logs_dir, ignore_errors=True)

    @patch("src.assistant.ChatOpenAI")
    def test_assistant_session_lifecycle_and_persistence(self, mock_chat_openai):
        """Test starting, processing, auto-saving session files and tool logs."""
        mock_llm_instance = MagicMock()
        mock_chat_openai.return_value = mock_llm_instance

        assistant = DocumentAssistant(
            openai_api_key="sk-fake-test-key",
            session_storage_path=self.session_dir,
        )
        assistant.tool_logger.logs_dir = self.logs_dir

        session_id = assistant.start_session(user_id="integration_tester")
        self.assertIsNotNone(session_id)
        self.assertEqual(assistant.current_session.user_id, "integration_tester")

        # Mock the workflow invoke directly to verify assistant state and file persistence
        mock_final_state = {
            "messages": [AIMessage(content="Invoice INV-001 total is $22,000.")],
            "intent": UserIntent(
                intent_type="qa",
                confidence=0.95,
                reasoning="Information query on invoice",
            ),
            "tools_used": ["document_search", "document_reader"],
            "active_documents": ["INV-001"],
            "actions_taken": ["classify_intent", "qa_agent", "update_memory"],
            "conversation_summary": "Discussed INV-001",
        }

        with patch.object(assistant.workflow, "invoke", return_value=mock_final_state):
            result = assistant.process_message("What is the total of INV-001?")

            self.assertTrue(result["success"])
            self.assertEqual(result["response"], "Invoice INV-001 total is $22,000.")
            self.assertEqual(result["intent"]["intent_type"], "qa")
            self.assertIn("document_search", result["tools_used"])
            self.assertIn("INV-001", result["sources"])
            self.assertEqual(result["actions_taken"], ["classify_intent", "qa_agent", "update_memory"])

            # Verify session file was auto-saved to session_storage_path
            expected_file = os.path.join(self.session_dir, f"{session_id}.json")
            self.assertTrue(os.path.exists(expected_file))

            # Verify resuming session loads saved data
            assistant2 = DocumentAssistant(
                openai_api_key="sk-fake-test-key",
                session_storage_path=self.session_dir,
            )
            resumed_id = assistant2.start_session(user_id="integration_tester", session_id=session_id)
            self.assertEqual(resumed_id, session_id)
            self.assertIn("INV-001", assistant2.current_session.document_context)
            self.assertEqual(len(assistant2.current_session.conversation_history), 1)


if __name__ == "__main__":
    unittest.main()
