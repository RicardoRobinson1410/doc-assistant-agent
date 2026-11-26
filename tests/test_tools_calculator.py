import os
import shutil
import unittest
from src.tools import ToolLogger, create_calculator_tool


class TestCalculatorTool(unittest.TestCase):
    def setUp(self):
        self.logs_dir = "./logs_test"
        self.logger = ToolLogger(logs_dir=self.logs_dir)
        self.calculator = create_calculator_tool(self.logger)

    def tearDown(self):
        if os.path.exists(self.logs_dir):
            shutil.rmtree(self.logs_dir, ignore_errors=True)

    def test_simple_addition(self):
        """Calculator should correctly compute a simple addition."""
        result = self.calculator.run("2 + 3")
        self.assertIn("2 + 3", result)
        self.assertIn("5", result)

    def test_complex_expression(self):
        """Calculator should evaluate valid complex expressions."""
        result = self.calculator.run("(10 + 5) * 2")
        self.assertIn("30", result)

    def test_division_and_decimals(self):
        """Calculator should evaluate float division correctly."""
        result = self.calculator.run("10 / 4")
        self.assertIn("2.5", result)

    def test_invalid_expression_rejected(self):
        """Calculator must reject unsafe or malformed input."""
        result = self.calculator.run("__import__('os').system('rm -rf /')")
        self.assertIn("Invalid expression", result)

    def test_power_operator_rejected(self):
        """Calculator must reject power operator **."""
        result = self.calculator.run("2 ** 8")
        self.assertIn("Invalid expression", result)

    def test_zero_division_handled(self):
        """Calculator should handle division by zero gracefully without crashing."""
        result = self.calculator.run("10 / 0")
        self.assertTrue("Error" in result or "division by zero" in result)

    def test_logging_occurs(self):
        """ToolLogger should record tool usage on each call."""
        initial_len = len(self.logger.get_logs())
        _ = self.calculator.run("1 + 1")
        logs = self.logger.get_logs()
        self.assertEqual(len(logs), initial_len + 1)
        last_log = logs[-1]
        self.assertEqual(last_log["tool_name"], "calculator")
        self.assertIn("input", last_log)
        self.assertIn("output", last_log)

    def test_document_tools(self):
        """Test document search, reader, and statistics tools."""
        from src.retrieval import SimulatedRetriever
        from src.tools import (
            create_document_search_tool,
            create_document_reader_tool,
            create_document_statistics_tool,
            get_all_tools,
        )

        retriever = SimulatedRetriever()
        search_tool = create_document_search_tool(retriever, self.logger)
        reader_tool = create_document_reader_tool(retriever, self.logger)
        stats_tool = create_document_statistics_tool(retriever, self.logger)

        # Search test
        search_res = search_tool.run({"query": "invoice"})
        self.assertIn("INV-001", search_res)

        # Reader test
        read_res = reader_tool.run({"doc_id": "INV-001"})
        self.assertIn("Acme Corporation", read_res)

        # Stats test
        stats_res = stats_tool.run({})
        self.assertIn("Total Documents", stats_res)

        # get_all_tools test
        all_tools = get_all_tools(retriever, self.logger)
        self.assertEqual(len(all_tools), 4)


if __name__ == "__main__":
    unittest.main()