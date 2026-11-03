import unittest

from src.prompts import (
    get_chat_prompt_template,
    get_intent_classification_prompt,
    QA_SYSTEM_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    CALCULATION_SYSTEM_PROMPT,
)


class TestPrompts(unittest.TestCase):
    def test_qa_prompt_uses_qa_system_prompt(self):
        """QA intent should select QA_SYSTEM_PROMPT."""
        template = get_chat_prompt_template("qa")
        rendered = template.invoke({"chat_history": [], "input": "Hello"}).to_messages()
        system_msg = rendered[0]
        self.assertEqual(system_msg.content.strip(), QA_SYSTEM_PROMPT.strip())

    def test_summarization_prompt_uses_summarization_system_prompt(self):
        """Summarization intent should select SUMMARIZATION_SYSTEM_PROMPT."""
        template = get_chat_prompt_template("summarization")
        rendered = template.invoke({"chat_history": [], "input": "Summarize this"}).to_messages()
        system_msg = rendered[0]
        self.assertEqual(system_msg.content.strip(), SUMMARIZATION_SYSTEM_PROMPT.strip())

    def test_calculation_prompt_uses_calculation_system_prompt(self):
        """Calculation intent should select CALCULATION_SYSTEM_PROMPT."""
        template = get_chat_prompt_template("calculation")
        rendered = template.invoke({"chat_history": [], "input": "Compute totals"}).to_messages()
        system_msg = rendered[0]
        self.assertEqual(system_msg.content.strip(), CALCULATION_SYSTEM_PROMPT.strip())

    def test_default_fallback_to_qa_prompt(self):
        """Unknown or empty intent should fallback to QA_SYSTEM_PROMPT."""
        template = get_chat_prompt_template("unknown")
        rendered = template.invoke({"chat_history": [], "input": "Other query"}).to_messages()
        self.assertEqual(rendered[0].content.strip(), QA_SYSTEM_PROMPT.strip())

    def test_intent_classification_prompt_variables(self):
        """Intent classification prompt must expect user_input and conversation_history."""
        prompt_tmpl = get_intent_classification_prompt()
        self.assertIn("user_input", prompt_tmpl.input_variables)
        self.assertIn("conversation_history", prompt_tmpl.input_variables)

        formatted = prompt_tmpl.format(
            user_input="How much was invoice INV-001?",
            conversation_history="User: Hello\nAssistant: Hi!"
        )
        self.assertIn("qa", formatted)
        self.assertIn("summarization", formatted)
        self.assertIn("calculation", formatted)
        self.assertIn("confidence", formatted)
        self.assertIn("reasoning", formatted)
        self.assertIn("INV-001", formatted)

    def test_calculation_prompt_instructions(self):
        """CALCULATION_SYSTEM_PROMPT should instruct using calculator tool for all calculations."""
        self.assertIn("calculator tool", CALCULATION_SYSTEM_PROMPT.lower())
        self.assertIn("document", CALCULATION_SYSTEM_PROMPT.lower())


if __name__ == "__main__":
    unittest.main()
