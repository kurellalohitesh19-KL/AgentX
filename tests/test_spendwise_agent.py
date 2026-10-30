"""Integration tests for SpendWise AI ReAct Agent Workflows.

Tests:
1. Expense categorization workflow
2. Duplicate detection workflow
3. Spending anomaly detection workflow
4. Subscription analysis workflow
5. Budget planning workflow
6. Multi-tool audit workflow
7. Grounding nudge enforcement (rejects memory-only answers on turn 1)
8. Tool error recovery workflow
9. Repeated tool call protection
10. Iteration limit enforcement
11. Empty dataset handling
12. CLI --json judging contract with financial queries

Run: python -m unittest discover tests
"""

import json
import os
import shutil
import subprocess
import sys
import unittest

from agent.core import AgentExecutor
from agent.llm import DryRunLLM
from tools import build_tools

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SYNTHETIC_SRC = os.path.join(TEST_DATA_DIR, "synthetic_transactions.csv")
SYNTHETIC_TXN = os.path.join(TEST_DATA_DIR, "transactions.csv")


class ScriptedLLM:
    """Helper LLM that returns a predetermined sequence of replies for targeted edge cases."""

    def __init__(self, replies):
        self.replies = list(replies)

    def generate(self, system, messages):
        return self.replies.pop(0) if self.replies else '{"final_answer": "Complete."}'


class TestSpendWiseAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orig_data_dir = os.environ.get("AGENTX_DATA_DIR")
        os.environ["AGENTX_DATA_DIR"] = TEST_DATA_DIR
        # Ensure a transactions.csv exists in test data directory for fast testing
        if os.path.exists(SYNTHETIC_SRC) and not os.path.exists(SYNTHETIC_TXN):
            shutil.copy(SYNTHETIC_SRC, SYNTHETIC_TXN)

    @classmethod
    def tearDownClass(cls):
        if cls.orig_data_dir is not None:
            os.environ["AGENTX_DATA_DIR"] = cls.orig_data_dir
        else:
            os.environ.pop("AGENTX_DATA_DIR", None)
        if os.path.exists(SYNTHETIC_TXN):
            os.remove(SYNTHETIC_TXN)

    def setUp(self):
        self.tools = build_tools()

    def test_expense_categorization_workflow(self):
        llm = DryRunLLM()
        answer, trace = AgentExecutor(llm, self.tools).run("Categorize my expenses in transactions.csv")
        self.assertTrue(len(trace) >= 1)
        self.assertEqual(trace[0]["tool"], "categorize_transactions")
        self.assertIn("category_breakdown", trace[0]["observation"])
        self.assertTrue(len(answer) > 0)

    def test_duplicate_detection_workflow(self):
        llm = DryRunLLM()
        answer, trace = AgentExecutor(llm, self.tools).run("Detect duplicate transactions and double charges")
        self.assertTrue(len(trace) >= 1)
        self.assertEqual(trace[0]["tool"], "detect_duplicates")
        self.assertIn("confirmed_duplicate_ids", trace[0]["observation"])
        self.assertIn("Duplicate payment analysis complete", answer)

    def test_anomaly_detection_workflow(self):
        llm = DryRunLLM()
        answer, trace = AgentExecutor(llm, self.tools).run("Find spending anomalies and surges")
        self.assertTrue(len(trace) >= 1)
        self.assertEqual(trace[0]["tool"], "detect_anomalies")
        self.assertIn("anomalous_transactions", trace[0]["observation"])
        self.assertIn("Spending anomaly hunt complete", answer)

    def test_subscription_analysis_workflow(self):
        llm = DryRunLLM()
        answer, trace = AgentExecutor(llm, self.tools).run("Audit my recurring subscriptions")
        self.assertTrue(len(trace) >= 1)
        self.assertEqual(trace[0]["tool"], "detect_subscriptions")
        self.assertIn("confirmed_subscriptions", trace[0]["observation"])
        self.assertIn("Subscription analysis complete", answer)

    def test_budget_planning_workflow(self):
        llm = DryRunLLM()
        answer, trace = AgentExecutor(llm, self.tools).run("Plan my 50/30/20 budget with income 60000")
        self.assertTrue(len(trace) >= 1)
        self.assertEqual(trace[0]["tool"], "plan_budget")
        self.assertIn("budget_framework_50_30_20", trace[0]["observation"])
        self.assertIn("50/30/20", answer)

    def test_multi_tool_audit_workflow(self):
        llm = DryRunLLM()
        answer, trace = AgentExecutor(llm, self.tools).run("Run a full financial audit on my transactions")
        self.assertTrue(len(trace) >= 3)
        tools_executed = [step["tool"] for step in trace]
        self.assertIn("categorize_transactions", tools_executed)
        self.assertIn("detect_duplicates", tools_executed)
        self.assertIn("detect_anomalies", tools_executed)
        self.assertIn("Financial Audit Complete", answer)

    def test_tool_error_and_recovery(self):
        llm = DryRunLLM()
        answer, trace = AgentExecutor(llm, self.tools).run("Simulate a tool error and test recovery")
        self.assertTrue(len(trace) >= 2)
        # First step attempts non_existent_tool
        self.assertEqual(trace[0]["tool"], "non_existent_tool")
        self.assertTrue(trace[0]["observation"].startswith("ERROR"))
        # Second step recovers with categorize_transactions
        self.assertEqual(trace[1]["tool"], "categorize_transactions")
        self.assertIn("category_breakdown", trace[1]["observation"])
        self.assertIn("Recovered", answer)

    def test_grounding_nudge_enforcement(self):
        # Model attempts to finish turn 1 from memory with no tools called
        replies = [
            '{"thought": "I know from memory.", "final_answer": "You spent $500."}',
            '{"thought": "Nudged. Calling tool.", "action": "categorize_transactions", "args": {"filename": "transactions.csv", "limit": 2}}',
            '{"thought": "Now I have facts.", "final_answer": "Verified you spent money in categories."}',
        ]
        executor = AgentExecutor(ScriptedLLM(replies), self.tools)
        answer, trace = executor.run("Check my spending")
        # Should have executed a tool on step 2 before returning final answer
        self.assertTrue(len(trace) >= 1)
        self.assertEqual(trace[0]["tool"], "categorize_transactions")
        self.assertEqual(answer, "Verified you spent money in categories.")

    def test_repeat_call_protection(self):
        # Repeated identical tool calls force final answer
        replies = [
            '{"thought": "t", "action": "list_files", "args": {}}',
            '{"thought": "t", "action": "list_files", "args": {}}',
            '{"thought": "t", "action": "list_files", "args": {}}',
            '{"thought": "t", "action": "list_files", "args": {}}',
            '{"thought": "t", "final_answer": "Stopped looping."}',
        ]
        executor = AgentExecutor(ScriptedLLM(replies), self.tools, max_iterations=5)
        answer, trace = executor.run("Check files repeatedly")
        self.assertLessEqual(len(trace), 5)
        self.assertEqual(answer, "Stopped looping.")

    def test_iteration_limit_enforcement(self):
        replies = ['{"thought": "t", "action": "list_files", "args": {}}'] * 10
        executor = AgentExecutor(ScriptedLLM(replies), self.tools, max_iterations=2)
        answer, trace = executor.run("Infinite loop test")
        self.assertEqual(len(trace), 2)
        self.assertTrue(len(answer) > 0)

    def test_empty_dataset_handling(self):
        # Calling tool on nonexistent file returns error observation, agent handles gracefully
        replies = [
            '{"thought": "Reading empty data.", "action": "categorize_transactions", "args": {"filename": "non_existent_data.csv"}}',
            '{"thought": "Observed error honestly.", "final_answer": "No records found; file does not exist."}',
        ]
        executor = AgentExecutor(ScriptedLLM(replies), self.tools)
        answer, trace = executor.run("Audit non_existent_data.csv")
        self.assertEqual(len(trace), 1)
        self.assertIn("error", trace[0]["observation"].lower())
        self.assertIn("No records found", answer)

    def test_cli_json_contract_financial_query(self):
        out = subprocess.run(
            [sys.executable, "main.py", "--json", "--dry-run", "audit my transactions"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(out.returncode, 0)
        output_lines = out.stdout.strip().splitlines()
        self.assertTrue(len(output_lines) >= 1)
        last_line = output_lines[-1]
        data = json.loads(last_line)
        self.assertIn("answer", data)
        self.assertIn("trace", data)
        self.assertTrue(len(data["trace"]) >= 1)
        # Verify genuine tool execution in trace
        trace_tools = [step["tool"] for step in data["trace"]]
        self.assertIn("categorize_transactions", trace_tools)


if __name__ == "__main__":
    unittest.main()
