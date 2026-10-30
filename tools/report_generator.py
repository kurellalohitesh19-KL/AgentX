"""SpendWise AI - Financial Audit Report Generator.

Synthesizes findings from all 5 financial analytics tools into executive-ready
markdown reports and structured CSV data exports.
"""

from __future__ import annotations

import io
import json
import os
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

import pandas as pd

from tools.categorizer import Categorizer
from tools.duplicate_detector import DuplicateDetector
from tools.anomaly_detector import AnomalyDetector
from tools.subscription_detector import SubscriptionDetector
from tools.budget_planner import BudgetPlanner
from tools.transaction_loader import load_transactions, validate_transactions


def _round_currency(val: float | Decimal) -> float:
    return float(Decimal(str(val)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


class FinancialReportGenerator:
    """Generates structured executive audit reports and exportable documents."""

    def __init__(self, filename: str = "transactions.csv"):
        self.filename = filename

    def generate_full_audit(
        self,
        monthly_income: Optional[float] = None,
        fixed_commitments: float = 0.0,
        savings_goal_pct: float = 20.0,
    ) -> Dict[str, Any]:
        """Runs all 5 tools and returns a consolidated audit dictionary."""
        # 1. Dataset Health
        df = load_transactions(self.filename)
        validation = validate_transactions(df)

        # 2. Categorization
        cat_tool = Categorizer()
        cat_data = json.loads(cat_tool.execute(filename=self.filename, limit=10))

        # 3. Duplicate Detection
        dup_tool = DuplicateDetector()
        dup_data = json.loads(dup_tool.execute(filename=self.filename, window_hours=24, max_results=10))

        # 4. Anomaly Detection
        anom_tool = AnomalyDetector()
        anom_data = json.loads(anom_tool.execute(filename=self.filename, sensitivity=2.5, max_results=10))

        # 5. Subscription Watch
        sub_tool = SubscriptionDetector()
        sub_data = json.loads(sub_tool.execute(filename=self.filename, min_cycles=3, max_results=10))

        # 6. Budget Plan
        bud_tool = BudgetPlanner()
        bud_data = json.loads(bud_tool.execute(
            filename=self.filename,
            monthly_income=monthly_income,
            fixed_commitments=fixed_commitments,
            savings_goal_pct=savings_goal_pct,
        ))

        # Consolidated Summary
        leak_total = dup_data.get("summary", {}).get("potential_recoverable_leak_total", 0.0)
        annual_sub_cost = sub_data.get("summary", {}).get("total_estimated_annual_commitment", 0.0)
        anomalies_count = anom_data.get("summary", {}).get("total_anomalies_detected", 0)
        confirmed_dup_count = dup_data.get("summary", {}).get("confirmed_duplicate_id_clusters", 0)
        potential_dup_count = dup_data.get("summary", {}).get("potential_duplicate_payment_clusters", 0)

        health_score = bud_data.get("financial_health_score", {}).get("score", 75)

        return {
            "dataset_name": os.path.basename(self.filename),
            "total_records": len(df),
            "health_score": health_score,
            "recoverable_leak_estimate": leak_total,
            "annual_subscription_commitment": annual_sub_cost,
            "anomalies_detected": anomalies_count,
            "confirmed_duplicates": confirmed_dup_count,
            "potential_duplicates": potential_dup_count,
            "validation": validation,
            "categorization": cat_data,
            "duplicates": dup_data,
            "anomalies": anom_data,
            "subscriptions": sub_data,
            "budget": bud_data,
        }

    def export_markdown_report(self, audit: Dict[str, Any]) -> str:
        """Formats the audit findings into a clean, executive-ready Markdown report."""
        lines = [
            f"# SpendWise AI — Executive Financial Audit Report",
            f"**Dataset Evaluated:** `{audit['dataset_name']}` | **Total Transactions:** `{audit['total_records']:,}`",
            f"**Overall Financial Health Score:** `{audit['health_score']}/100`",
            f"",
            f"---",
            f"",
            f"## 1. Executive Summary & Core Metrics",
            f"",
            f"| Metric | Finding | Status / Risk |",
            f"| :--- | :--- | :--- |",
            f"| **Identified Recoverable Leaks** | ₹{audit['recoverable_leak_estimate']:,.2f} | Potential immediate savings from duplicate charges |",
            f"| **Annual Recurring Commitment** | ₹{audit['annual_subscription_commitment']:,.2f} | Projected annualized recurring expenses |",
            f"| **Statistical Spending Anomalies** | {audit['anomalies_detected']:,} transactions | Outliers exceeding category baseline IQR fences |",
            f"| **Confirmed Duplicate IDs** | {audit['confirmed_duplicates']} clusters | System / ledger replay collisions |",
            f"| **Potential Duplicate Payments** | {audit['potential_duplicates']} occurrences | Consecutive matching amounts within 24h |",
            f"",
            f"---",
            f"",
            f"## 2. Category Expenditure Breakdown",
            f"",
            f"| Category | Transaction Count | Total Spend (INR) | % of Total Spend |",
            f"| :--- | :--- | :--- | :--- |",
        ]

        cats = audit.get("categorization", {}).get("category_breakdown", [])
        for c in cats:
            lines.append(
                f"| **{c['category']}** | {c['transaction_count']:,} | ₹{c['total_spend']:,.2f} | {c['percent_of_total_spend']}% |"
            )

        lines.extend([
            f"",
            f"---",
            f"",
            f"## 3. Duplicate Payment & Leakage Findings",
            f"",
        ])

        dups = audit.get("duplicates", {}).get("potential_duplicate_payments", [])
        if dups:
            lines.extend([
                f"| Risk Level | Amount | Time Delta | First Transaction | Second Transaction | Explanation |",
                f"| :--- | :--- | :--- | :--- | :--- | :--- |",
            ])
            for d in dups[:5]:
                t1 = d["first_transaction"]["transaction_id"]
                t2 = d["second_transaction"]["transaction_id"]
                lines.append(
                    f"| `{d['risk_level']}` | ₹{d['amount']:,.2f} | {d['time_interval_minutes']}m | `{t1}` | `{t2}` | {d['explanation']} |"
                )
        else:
            lines.append("No suspicious duplicate payments detected within time window.")

        lines.extend([
            f"",
            f"---",
            f"",
            f"## 4. Top Statistical Spending Anomalies (IQR & Z-Score)",
            f"",
        ])

        anoms = audit.get("anomalies", {}).get("anomalous_transactions", [])
        if anoms:
            lines.extend([
                f"| Transaction ID | Date | Category | Amount | Ratio to Median | Z-Score | Severity |",
                f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
            ])
            for a in anoms[:5]:
                lines.append(
                    f"| `{a['transaction_id']}` | {a['timestamp']} | **{a['category']}** | ₹{a['amount']:,.2f} | `{a['ratio_to_median']}` | `{a['z_score']}` | `{a['severity']}` |"
                )
        else:
            lines.append("No statistical spending outliers detected at current sensitivity threshold.")

        lines.extend([
            f"",
            f"---",
            f"",
            f"## 5. Recurring Payments & Subscriptions",
            f"",
        ])

        subs = audit.get("subscriptions", {}).get("confirmed_subscriptions", [])
        if subs:
            lines.extend([
                f"| Entity / Category | Cadence | Cycles | Latest Amount | Est. Annual Cost | Price Creep |",
                f"| :--- | :--- | :--- | :--- | :--- | :--- |",
            ])
            for s in subs[:5]:
                entity_label = str(s["entity"])
                creep_info = f"+{s['price_creep']['percentage_increase']}" if s.get("price_creep") else "None"
                lines.append(
                    f"| {entity_label} | `{s['cadence']}` | {s['cycle_count']} | ₹{s['latest_amount']:,.2f} | ₹{s['estimated_annual_cost']:,.2f} | {creep_info} |"
                )
        else:
            lines.append("No confirmed periodic subscriptions with regular intervals identified.")

        lines.extend([
            f"",
            f"---",
            f"",
            f"## 6. Budget Recommendations & 50/30/20 Framework",
            f"",
        ])

        b = audit.get("budget", {})
        if b.get("status") == "success":
            fw = b.get("budget_framework_50_30_20", {})
            lines.extend([
                f"- **Monthly Income:** ₹{b['income_profile']['user_provided_monthly_income']:,.2f}",
                f"- **Needs (50% Target):** Budget = ₹{fw['needs_50']['target_allocation']:,.2f} | Actual = ₹{fw['needs_50']['actual_monthly_spending']:,.2f} (`{fw['needs_50']['status']}`)",
                f"- **Wants (30% Target):** Budget = ₹{fw['wants_30']['target_allocation']:,.2f} | Actual = ₹{fw['wants_30']['actual_monthly_spending']:,.2f} (`{fw['wants_30']['status']}`)",
                f"- **Savings Target:** ₹{fw['savings_20']['target_monthly_savings']:,.2f} | Net Monthly Surplus = ₹{fw['savings_20']['actual_monthly_savings']:,.2f} (`{fw['savings_20']['status']}`)",
                f"- **Potential Monthly Savings Unlocked:** `₹{b['potential_monthly_savings_unlocked']:,.2f}`",
                f"",
                f"> **Recommendation:** {b['recommendation_summary']}",
            ])
        else:
            lines.append(
                "Provide user income in Smart Budget Planner to generate personalized 50/30/20 target limits."
            )

        lines.extend([
            f"",
            f"---",
            f"*Report generated by SpendWise AI — Expense Auditor & Anomaly Hunter (Finance & Risk AgentX Challenge 2026).*",
        ])
        return "\n".join(lines)

    def export_csv_summary(self, audit: Dict[str, Any]) -> str:
        """Exports the category and anomaly breakdown as a tabular CSV string."""
        cats = audit.get("categorization", {}).get("category_breakdown", [])
        if not cats:
            return "category,transaction_count,total_spend,average_spend,percent_of_total_spend\n"
        df_cat = pd.DataFrame(cats)
        buf = io.StringIO()
        df_cat.to_csv(buf, index=False)
        return buf.getvalue()
