"""Offline unit tests for SpendWise Financial Analytics Tools.

Tests:
1. Categorizer (tools/categorizer.py)
2. DuplicateDetector (tools/duplicate_detector.py)
3. AnomalyDetector (tools/anomaly_detector.py)
4. SubscriptionDetector (tools/subscription_detector.py)
5. BudgetPlanner (tools/budget_planner.py)

Run: python -m unittest discover tests
"""

import json
import os
import unittest
import pandas as pd

from tools import build_tools
from tools.categorizer import Categorizer
from tools.duplicate_detector import DuplicateDetector
from tools.anomaly_detector import AnomalyDetector
from tools.subscription_detector import SubscriptionDetector
from tools.budget_planner import BudgetPlanner

SYNTHETIC_DIR = os.path.join(os.path.dirname(__file__), "data")
SYNTHETIC_CSV = os.path.join(SYNTHETIC_DIR, "synthetic_transactions.csv")


class TestSpendWiseTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create an enriched fixture for testing cadence, price creeps, and surges
        cls.enriched_csv = os.path.join(SYNTHETIC_DIR, "enriched_test_transactions.csv")
        data = [
            # Subscription with price creep (Monthly Spotify-like)
            {"transaction_id": "SUB01", "timestamp": "2024-01-10 10:00:00", "category": "Entertainment", "amount": 100.0, "status": "SUCCESS", "receiver_bank": "BankA", "transaction_type": "P2M"},
            {"transaction_id": "SUB02", "timestamp": "2024-02-10 10:00:00", "category": "Entertainment", "amount": 100.0, "status": "SUCCESS", "receiver_bank": "BankA", "transaction_type": "P2M"},
            {"transaction_id": "SUB03", "timestamp": "2024-03-11 10:00:00", "category": "Entertainment", "amount": 120.0, "status": "SUCCESS", "receiver_bank": "BankA", "transaction_type": "P2M"},
            {"transaction_id": "SUB04", "timestamp": "2024-04-10 10:00:00", "category": "Entertainment", "amount": 120.0, "status": "SUCCESS", "receiver_bank": "BankA", "transaction_type": "P2M"},

            # Irregular recurring (Erratic intervals)
            {"transaction_id": "IRR01", "timestamp": "2024-01-02 12:00:00", "category": "Shopping", "amount": 250.0, "status": "SUCCESS", "receiver_bank": "BankB", "transaction_type": "P2M"},
            {"transaction_id": "IRR02", "timestamp": "2024-01-05 12:00:00", "category": "Shopping", "amount": 250.0, "status": "SUCCESS", "receiver_bank": "BankB", "transaction_type": "P2M"},
            {"transaction_id": "IRR03", "timestamp": "2024-02-28 12:00:00", "category": "Shopping", "amount": 250.0, "status": "SUCCESS", "receiver_bank": "BankB", "transaction_type": "P2M"},

            # Baseline transactions for Grocery to test anomalies (12 items: 11 normal ~$50, 1 extreme outlier $1500)
            {"transaction_id": "GRO01", "timestamp": "2024-01-03 08:00:00", "category": "Grocery", "amount": 48.0, "status": "SUCCESS", "receiver_bank": "BankC", "transaction_type": "P2M"},
            {"transaction_id": "GRO02", "timestamp": "2024-01-07 08:00:00", "category": "Grocery", "amount": 52.0, "status": "SUCCESS", "receiver_bank": "BankC", "transaction_type": "P2M"},
            {"transaction_id": "GRO03", "timestamp": "2024-01-14 08:00:00", "category": "Grocery", "amount": 50.0, "status": "SUCCESS", "receiver_bank": "BankC", "transaction_type": "P2M"},
            {"transaction_id": "GRO04", "timestamp": "2024-01-21 08:00:00", "category": "Grocery", "amount": 55.0, "status": "SUCCESS", "receiver_bank": "BankC", "transaction_type": "P2M"},
            {"transaction_id": "GRO05", "timestamp": "2024-01-28 08:00:00", "category": "Grocery", "amount": 49.0, "status": "SUCCESS", "receiver_bank": "BankC", "transaction_type": "P2M"},
            {"transaction_id": "GRO06", "timestamp": "2024-02-04 08:00:00", "category": "Grocery", "amount": 51.0, "status": "SUCCESS", "receiver_bank": "BankC", "transaction_type": "P2M"},
            {"transaction_id": "GRO07", "timestamp": "2024-02-11 08:00:00", "category": "Grocery", "amount": 53.0, "status": "SUCCESS", "receiver_bank": "BankC", "transaction_type": "P2M"},
            {"transaction_id": "GRO08", "timestamp": "2024-02-18 08:00:00", "category": "Grocery", "amount": 47.0, "status": "SUCCESS", "receiver_bank": "BankC", "transaction_type": "P2M"},
            {"transaction_id": "GRO09", "timestamp": "2024-02-25 08:00:00", "category": "Grocery", "amount": 50.0, "status": "SUCCESS", "receiver_bank": "BankC", "transaction_type": "P2M"},
            {"transaction_id": "GRO10", "timestamp": "2024-03-03 08:00:00", "category": "Grocery", "amount": 52.0, "status": "SUCCESS", "receiver_bank": "BankC", "transaction_type": "P2M"},
            {"transaction_id": "GRO11", "timestamp": "2024-03-10 08:00:00", "category": "Grocery", "amount": 49.0, "status": "SUCCESS", "receiver_bank": "BankC", "transaction_type": "P2M"},
            {"transaction_id": "GRO12", "timestamp": "2024-03-15 18:00:00", "category": "Grocery", "amount": 1500.0, "status": "SUCCESS", "receiver_bank": "BankC", "transaction_type": "P2M"},
        ]
        pd.DataFrame(data).to_csv(cls.enriched_csv, index=False)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.enriched_csv):
            os.remove(cls.enriched_csv)

    def test_tool_registry_contains_all_tools(self):
        tools = build_tools()
        self.assertIn("categorize_transactions", tools)
        self.assertIn("detect_duplicates", tools)
        self.assertIn("detect_anomalies", tools)
        self.assertIn("detect_subscriptions", tools)
        self.assertIn("plan_budget", tools)
        # Core starter tools preserved
        self.assertIn("calculator", tools)
        self.assertIn("list_files", tools)
        self.assertIn("read_file", tools)
        self.assertIn("sql_query", tools)
        self.assertIn("web_search", tools)

    # --------------------------------------------------------------------------
    # 1. CATEGORIZER TESTS
    # --------------------------------------------------------------------------
    def test_categorizer_summary(self):
        tool = Categorizer()
        res = json.loads(tool.execute(filename=SYNTHETIC_CSV))
        self.assertEqual(res["status"], "success")
        self.assertIn("category_breakdown", res)
        self.assertTrue(len(res["category_breakdown"]) > 0)
        # Check spending totals exist and are rounded
        first_cat = res["category_breakdown"][0]
        self.assertIn("category", first_cat)
        self.assertIn("total_spend", first_cat)

    def test_categorizer_single_transaction_audit(self):
        tool = Categorizer()
        res = json.loads(tool.execute(filename=SYNTHETIC_CSV, transaction_id="SYN001"))
        self.assertEqual(res["status"], "success")
        item = res["result"]
        self.assertEqual(item["transaction_id"], "SYN001")
        self.assertEqual(item["assigned_category"], "Groceries")
        self.assertEqual(item["confidence_score"], 1.0)
        self.assertIn("direct match", item["classification_reason"].lower())

    def test_categorizer_transaction_not_found(self):
        tool = Categorizer()
        res = json.loads(tool.execute(filename=SYNTHETIC_CSV, transaction_id="NON_EXISTENT_ID"))
        self.assertIn("error", res)

    # --------------------------------------------------------------------------
    # 2. DUPLICATE DETECTOR TESTS
    # --------------------------------------------------------------------------
    def test_duplicate_detector_confirmed_ids(self):
        tool = DuplicateDetector()
        res = json.loads(tool.execute(filename=SYNTHETIC_CSV, window_hours=48))
        self.assertEqual(res["status"], "success")
        # SYN001 is intentionally duplicated in synthetic_transactions.csv
        confirmed = res["confirmed_duplicate_ids"]
        self.assertTrue(len(confirmed) >= 1)
        self.assertEqual(confirmed[0]["transaction_id"], "SYN001")
        self.assertEqual(confirmed[0]["occurrences"], 2)
        self.assertEqual(confirmed[0]["potential_leak"], 450.0)

    def test_duplicate_detector_potential_payments(self):
        tool = DuplicateDetector()
        res = json.loads(tool.execute(filename=SYNTHETIC_CSV, window_hours=24))
        self.assertEqual(res["status"], "success")
        # SYN003 ($1200) and SYN004 ($1200) occurred 2 minutes apart (SYN004 FAILED)
        potentials = res["potential_duplicate_payments"]
        self.assertTrue(len(potentials) >= 1)
        retry_case = [p for p in potentials if p["first_transaction"]["amount"] == 1200.0]
        self.assertTrue(len(retry_case) > 0)
        self.assertIn("RETRY", retry_case[0]["risk_level"])

    # --------------------------------------------------------------------------
    # 3. ANOMALY DETECTOR TESTS
    # --------------------------------------------------------------------------
    def test_anomaly_detector_insufficient_history(self):
        tool = AnomalyDetector()
        # Synthetic data has only 2 Food transactions (< 10)
        res = json.loads(tool.execute(filename=SYNTHETIC_CSV, category="Food"))
        self.assertEqual(res["status"], "success")
        baselines = res["category_baselines"]
        self.assertEqual(len(baselines), 1)
        self.assertEqual(baselines[0]["status"], "INSUFFICIENT_HISTORY")

    def test_anomaly_detector_outlier_detection(self):
        tool = AnomalyDetector()
        # Enriched data has 12 Grocery items with 1 extreme outlier ($1500 vs median $50)
        res = json.loads(tool.execute(filename=self.enriched_csv, category="Grocery"))
        self.assertEqual(res["status"], "success")
        anomalies = res["anomalous_transactions"]
        self.assertTrue(len(anomalies) >= 1)
        extreme = anomalies[0]
        self.assertEqual(extreme["transaction_id"], "GRO12")
        self.assertEqual(extreme["amount"], 1500.0)
        self.assertEqual(extreme["severity"], "EXTREME")
        self.assertIn("ratio_to_median", extreme)
        self.assertIn("z_score", extreme)

    # --------------------------------------------------------------------------
    # 4. SUBSCRIPTION DETECTOR TESTS
    # --------------------------------------------------------------------------
    def test_subscription_cadence_and_price_creep(self):
        tool = SubscriptionDetector()
        res = json.loads(tool.execute(filename=self.enriched_csv, min_cycles=3))
        self.assertEqual(res["status"], "success")
        confirmed = res["confirmed_subscriptions"]
        self.assertTrue(len(confirmed) >= 1)
        sub = [s for s in confirmed if "Entertainment" in str(s["entity"])][0]
        self.assertEqual(sub["cadence"], "MONTHLY")
        self.assertEqual(sub["cycle_count"], 4)
        # Price creep check (went from 100 to 120)
        self.assertIsNotNone(sub["price_creep"])
        self.assertTrue(sub["price_creep"]["detected"])
        self.assertEqual(sub["price_creep"]["increase_per_cycle"], 20.0)
        self.assertEqual(sub["price_creep"]["annualized_extra_cost"], 240.0)
        # Missing merchant flag
        self.assertIn("merchant", res["summary"]["missing_fields_identified"])

    def test_subscription_ignores_irregular_intervals(self):
        tool = SubscriptionDetector()
        res = json.loads(tool.execute(filename=self.enriched_csv, min_cycles=3))
        # Shopping transactions have irregular intervals
        irregular = res["irregular_recurring_patterns"]
        shopping_irr = [irr for irr in irregular if "Shopping" in str(irr["entity"])]
        self.assertTrue(len(shopping_irr) > 0)
        self.assertFalse(shopping_irr[0]["is_confirmed_subscription"])

    # --------------------------------------------------------------------------
    # 5. BUDGET PLANNER TESTS
    # --------------------------------------------------------------------------
    def test_budget_planner_no_income_does_not_hallucinate(self):
        tool = BudgetPlanner()
        # When no monthly_income is passed
        res = json.loads(tool.execute(filename=SYNTHETIC_CSV, monthly_income=None))
        self.assertEqual(res["status"], "partial_profile")
        self.assertEqual(res["income_status"], "NOT_PROVIDED")
        self.assertIn("does not invent", res["notice"].lower())
        self.assertIn("actual_monthly_run_rate", res)

    def test_budget_planner_50_30_20_allocation(self):
        tool = BudgetPlanner()
        res = json.loads(tool.execute(
            filename=SYNTHETIC_CSV,
            monthly_income=50000.0,
            fixed_commitments=5000.0,
            savings_goal_pct=20.0,
        ))
        self.assertEqual(res["status"], "success")
        profile = res["income_profile"]
        self.assertEqual(profile["user_provided_monthly_income"], 50000.0)
        self.assertEqual(profile["fixed_commitments"], 5000.0)

        # 50/30/20 targets
        framework = res["budget_framework_50_30_20"]
        self.assertEqual(framework["needs_50"]["target_allocation"], 25000.0)
        self.assertEqual(framework["wants_30"]["target_allocation"], 15000.0)
        self.assertEqual(framework["savings_20"]["target_monthly_savings"], 10000.0)

        # Proposed limits and health score
        self.assertIn("proposed_category_limits", res)
        self.assertIn("financial_health_score", res)
        score = res["financial_health_score"]["score"]
        self.assertTrue(0 <= score <= 100)


if __name__ == "__main__":
    unittest.main()
