"""Unit & Integration tests for Dashboard modules, Report Generator, and CSV Uploads.

Run: python -m unittest discover tests
"""

import json
import os
import shutil
import subprocess
import sys
import unittest

from main import run_agent
from tools.report_generator import FinancialReportGenerator
from tools.transaction_loader import load_transactions, validate_transactions

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SYNTHETIC_CSV = os.path.join(TEST_DATA_DIR, "synthetic_transactions.csv")


class TestDashboardAndReports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orig_data_dir = os.environ.get("AGENTX_DATA_DIR")
        os.environ["AGENTX_DATA_DIR"] = TEST_DATA_DIR

    @classmethod
    def tearDownClass(cls):
        if cls.orig_data_dir is not None:
            os.environ["AGENTX_DATA_DIR"] = cls.orig_data_dir
        else:
            os.environ.pop("AGENTX_DATA_DIR", None)

    def test_dashboard_import(self):
        """Verifies app.py can be parsed and imported without syntax or missing dependency errors."""
        import app
        self.assertTrue(hasattr(app, "CUSTOM_CSS"))
        self.assertTrue(hasattr(app, "get_dataset"))
        self.assertTrue(hasattr(app, "get_cached_audit"))

    def test_csv_upload_validation(self):
        """Tests transaction validation and normalization on an uploaded CSV fixture."""
        df = load_transactions(SYNTHETIC_CSV)
        metrics = validate_transactions(df)
        self.assertIn("data_quality_score", metrics)
        self.assertIn("duplicate_ids_count", metrics)
        self.assertIn("amount_stats", metrics)
        self.assertEqual(metrics["total_rows"], 14)

    def test_report_generator_full_audit(self):
        """Tests that FinancialReportGenerator executes all tools and produces a complete report."""
        generator = FinancialReportGenerator(filename=SYNTHETIC_CSV)
        audit = generator.generate_full_audit(
            monthly_income=60000.0,
            fixed_commitments=10000.0,
            savings_goal_pct=20.0,
        )
        self.assertIn("total_records", audit)
        self.assertIn("health_score", audit)
        self.assertIn("recoverable_leak_estimate", audit)
        self.assertIn("categorization", audit)
        self.assertIn("duplicates", audit)
        self.assertIn("anomalies", audit)
        self.assertIn("subscriptions", audit)
        self.assertIn("budget", audit)

        # Test Markdown Export
        md_text = generator.export_markdown_report(audit)
        self.assertIn("# SpendWise AI — Executive Financial Audit Report", md_text)
        self.assertIn("## 1. Executive Summary", md_text)
        self.assertIn("## 6. Budget Recommendations", md_text)

        # Test CSV Export
        csv_text = generator.export_csv_summary(audit)
        self.assertIn("category", csv_text)
        self.assertIn("total_spend", csv_text)

    def test_agent_to_dashboard_integration(self):
        """Tests that run_agent() from main.py responds with a valid answer and trace for dashboard use."""
        answer, trace = run_agent("Audit my expenses", dry_run=True, verbose=False)
        self.assertTrue(len(answer) > 0)
        self.assertTrue(len(trace) >= 1)
        self.assertIn("tool", trace[0])
        self.assertIn("observation", trace[0])

    def test_cli_json_contract_intact(self):
        """Confirms that main.py --json produces valid single-line JSON adhering to judging contract."""
        out = subprocess.run(
            [sys.executable, "main.py", "--json", "--dry-run", "test"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(out.returncode, 0)
        lines = out.stdout.strip().splitlines()
        self.assertEqual(len(lines), 1)
        data = json.loads(lines[0])
        self.assertIn("answer", data)
        self.assertIn("trace", data)
        self.assertTrue(len(data["trace"]) >= 2)


if __name__ == "__main__":
    unittest.main()
