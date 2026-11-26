import unittest
from datetime import datetime
from pydantic import ValidationError

from src.schemas import (
    AnswerResponse,
    UserIntent,
    DocumentChunk,
    SummarizationResponse,
    CalculationResponse,
    UpdateMemoryResponse,
    SessionState,
)


class TestSchemas(unittest.TestCase):
    def test_answer_response_creation(self):
        """AnswerResponse should accept required fields and auto-set timestamp."""
        ar = AnswerResponse(
            question="What is the total?",
            answer="The total is $10,000.",
            sources=["INV-001"],
            confidence=0.95,
        )
        self.assertEqual(ar.question, "What is the total?")
        self.assertEqual(ar.answer, "The total is $10,000.")
        self.assertEqual(ar.sources, ["INV-001"])
        self.assertTrue(0.0 <= ar.confidence <= 1.0)
        self.assertIsInstance(ar.timestamp, datetime)

    def test_answer_response_confidence_bounds(self):
        """AnswerResponse must reject confidence scores outside [0.0, 1.0]."""
        with self.assertRaises(ValidationError):
            AnswerResponse(
                question="What is the total?",
                answer="The total is $10,000.",
                confidence=1.5,
            )
        with self.assertRaises(ValidationError):
            AnswerResponse(
                question="What is the total?",
                answer="The total is $10,000.",
                confidence=-0.1,
            )

    def test_user_intent_valid_types(self):
        """UserIntent should accept only allowed intent_type values."""
        valid_intents = ["qa", "summarization", "calculation", "unknown"]
        for intent_str in valid_intents:
            intent = UserIntent(
                intent_type=intent_str,
                confidence=0.9,
                reasoning=f"Classified as {intent_str}.",
            )
            self.assertEqual(intent.intent_type, intent_str)
            self.assertTrue(0.0 <= intent.confidence <= 1.0)

    def test_user_intent_invalid_type_raises(self):
        """Invalid intent_type should raise a ValidationError."""
        with self.assertRaises(ValidationError):
            UserIntent(
                intent_type="invalid_intent",  # type: ignore[arg-type]
                confidence=0.5,
                reasoning="Invalid type for testing.",
            )

    def test_user_intent_confidence_bounds(self):
        """UserIntent must reject confidence scores outside [0.0, 1.0]."""
        with self.assertRaises(ValidationError):
            UserIntent(
                intent_type="qa",
                confidence=1.2,
                reasoning="Out of bounds confidence.",
            )
        with self.assertRaises(ValidationError):
            UserIntent(
                intent_type="qa",
                confidence=-0.5,
                reasoning="Negative confidence.",
            )

    def test_summarization_response(self):
        """SummarizationResponse should store summary and key points."""
        sr = SummarizationResponse(
            original_length=500,
            summary="A brief summary.",
            key_points=["Point 1", "Point 2"],
            document_ids=["CON-001"],
        )
        self.assertEqual(sr.original_length, 500)
        self.assertEqual(len(sr.key_points), 2)
        self.assertIsInstance(sr.timestamp, datetime)

    def test_calculation_response(self):
        """CalculationResponse should store math result, expression and explanation."""
        cr = CalculationResponse(
            expression="5000 + 12500",
            result=17500.0,
            explanation="Sum of two line items",
            units="USD",
        )
        self.assertEqual(cr.result, 17500.0)
        self.assertEqual(cr.units, "USD")
        self.assertIsInstance(cr.timestamp, datetime)

    def test_session_state(self):
        """SessionState should store session data cleanly."""
        state = SessionState(
            session_id="test-session-123",
            user_id="user-456",
        )
        self.assertEqual(state.session_id, "test-session-123")
        self.assertEqual(state.conversation_history, [])
        self.assertEqual(state.document_context, [])


if __name__ == "__main__":
    unittest.main()

