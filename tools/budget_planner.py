"""SpendWise AI - Personalized Budget Planner & 50/30/20 Optimizer Tool.

Calculates data-backed category spending limits, 50/30/20 allocations, and exact savings targets.
Operates on actual historical expense totals and explicit user-provided income.
Strictly avoids fabricating salary or financial details when unprovided.
Uses Decimal rounding for exact currency calculations.
"""

from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

import pandas as pd

from tools import Tool
from tools.transaction_loader import load_transactions

NEEDS_CATEGORIES = {
    "grocery",
    "groceries",
    "utilities",
    "utilities & bills",
    "transport",
    "transportation",
    "fuel",
    "fuel & vehicle",
    "healthcare",
    "healthcare & medical",
    "education",
    "housing",
    "housing & rent",
}

WANTS_CATEGORIES = {
    "food",
    "food & dining",
    "entertainment",
    "entertainment & leisure",
    "shopping",
    "shopping & retail",
    "other",
    "other / uncategorized",
    "peer-to-peer transfer",
    "merchant payment",
}


def _round_currency(val: float | Decimal) -> float:
    return float(Decimal(str(val)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


class BudgetPlanner(Tool):
    name = "plan_budget"
    description = (
        "Generate a personalized 50/30/20 budget plan, proposed category spending limits, "
        "and exact savings calculations based on actual spending and user-provided income. "
        "Never fabricates unprovided financial details."
    )
    args_doc = (
        '{"filename": "transactions.csv", "monthly_income": 60000.0, '
        '"fixed_commitments": 10000.0, "savings_goal_pct": 20.0}'
    )

    def execute(
        self,
        filename: str = "transactions.csv",
        monthly_income: Optional[float] = None,
        fixed_commitments: float = 0.0,
        savings_goal_pct: float = 20.0,
    ) -> str:
        try:
            df = load_transactions(filename)
        except Exception as e:
            return json.dumps({"error": f"Failed to load transactions: {e}"})

        if df.empty:
            return json.dumps({"error": "Transaction dataset is empty."})

        # Filter successful transactions
        if "status" in df.columns:
            spend_df = df[df["status"] == "SUCCESS"].copy()
        else:
            spend_df = df.copy()

        spend_df["amount"] = pd.to_numeric(spend_df["amount"], errors="coerce")
        spend_df = spend_df.dropna(subset=["amount"])

        # Determine timeframe span in months
        months_covered = 1.0
        if "timestamp" in spend_df.columns:
            ts = pd.to_datetime(spend_df["timestamp"], errors="coerce").dropna()
            if not ts.empty:
                span_days = (ts.max() - ts.min()).total_seconds() / 86400.0
                months_covered = max(1.0, round(span_days / 30.4, 2))

        # Calculate actual monthly spending per category
        cat_totals = spend_df.groupby("category")["amount"].sum().to_dict()
        cat_monthly: Dict[str, float] = {
            cat: _round_currency(tot / months_covered) for cat, tot in cat_totals.items()
        }

        total_actual_monthly_spend = _round_currency(sum(cat_monthly.values()))

        # Separate actual monthly spending into Needs vs Wants
        actual_needs = 0.0
        actual_wants = 0.0
        for cat, amt in cat_monthly.items():
            if cat.lower() in NEEDS_CATEGORIES:
                actual_needs += amt
            else:
                actual_wants += amt

        actual_needs = _round_currency(actual_needs)
        actual_wants = _round_currency(actual_wants)

        # Handle unprovided income gracefully (No hallucination)
        user_inc = float(monthly_income) if monthly_income and float(monthly_income) > 0 else None
        if user_inc is None:
            return json.dumps({
                "status": "partial_profile",
                "income_status": "NOT_PROVIDED",
                "notice": (
                    "Monthly income was not provided. SpendWise AI does not invent or assume "
                    "user salary figures. Current expense breakdown and run-rates are calculated below. "
                    "To generate 50/30/20 target limits and exact savings surplus, provide 'monthly_income'."
                ),
                "actual_monthly_run_rate": {
                    "months_evaluated": months_covered,
                    "total_monthly_spending": total_actual_monthly_spend,
                    "needs_monthly_spending": actual_needs,
                    "wants_monthly_spending": actual_wants,
                    "category_monthly_breakdown": cat_monthly,
                },
            }, indent=2)

        # Full Budget Optimization with User-Provided Income
        commitments = float(fixed_commitments) if fixed_commitments else 0.0
        goal_pct = float(savings_goal_pct) if savings_goal_pct else 20.0

        target_needs = _round_currency(user_inc * 0.50)
        target_wants = _round_currency(user_inc * 0.30)
        target_savings = _round_currency(user_inc * (goal_pct / 100.0))

        needs_variance = _round_currency(actual_needs + commitments - target_needs)
        wants_variance = _round_currency(actual_wants - target_wants)

        current_net_savings = _round_currency(user_inc - total_actual_monthly_spend - commitments)
        savings_variance = _round_currency(current_net_savings - target_savings)

        # Propose Category Spending Caps (Trimming over-budget areas)
        proposed_caps: Dict[str, Dict[str, Any]] = {}
        for cat, curr_amt in cat_monthly.items():
            is_need = cat.lower() in NEEDS_CATEGORIES
            if is_need:
                # Needs allowed up to historical share of target needs
                proposed_cap = curr_amt
                status = "ON_TRACK"
            else:
                # Wants capped to keep total wants under 30%
                if wants_variance > 0 and actual_wants > 0:
                    trim_ratio = target_wants / actual_wants
                    proposed_cap = _round_currency(curr_amt * trim_ratio)
                    status = "REDUCE_SPENDING"
                else:
                    proposed_cap = curr_amt
                    status = "WITHIN_BUDGET"

            proposed_caps[cat] = {
                "current_monthly_spend": curr_amt,
                "proposed_monthly_cap": proposed_cap,
                "potential_monthly_savings": _round_currency(max(0.0, curr_amt - proposed_cap)),
                "status": status,
            }

        # Calculate Financial Health Score (0 - 100)
        health_score = self._compute_health_score(
            income=user_inc,
            total_spend=total_actual_monthly_spend + commitments,
            needs_spend=actual_needs + commitments,
            wants_spend=actual_wants,
            target_needs=target_needs,
            target_wants=target_wants,
            target_savings=target_savings,
        )

        result = {
            "status": "success",
            "income_profile": {
                "user_provided_monthly_income": _round_currency(user_inc),
                "fixed_commitments": _round_currency(commitments),
                "target_savings_goal_percentage": f"{goal_pct}%",
                "months_evaluated": months_covered,
            },
            "financial_health_score": {
                "score": health_score,
                "rating": "EXCELLENT" if health_score >= 80 else "FAIR" if health_score >= 60 else "NEEDS_OPTIMIZATION",
            },
            "budget_framework_50_30_20": {
                "needs_50": {
                    "target_allocation": target_needs,
                    "actual_monthly_spending": _round_currency(actual_needs + commitments),
                    "variance": needs_variance,
                    "status": "OVER_BUDGET" if needs_variance > 0 else "WITHIN_BUDGET",
                },
                "wants_30": {
                    "target_allocation": target_wants,
                    "actual_monthly_spending": actual_wants,
                    "variance": wants_variance,
                    "status": "OVER_BUDGET" if wants_variance > 0 else "WITHIN_BUDGET",
                },
                "savings_20": {
                    "target_monthly_savings": target_savings,
                    "actual_monthly_savings": current_net_savings,
                    "variance": savings_variance,
                    "status": "MEETING_GOAL" if savings_variance >= 0 else "BELOW_TARGET",
                },
            },
            "potential_monthly_savings_unlocked": _round_currency(max(0.0, wants_variance)),
            "proposed_category_limits": proposed_caps,
            "recommendation_summary": (
                f"With a monthly income of {_round_currency(user_inc)}, allocating 50% ({_round_currency(target_needs)}) to Needs, "
                f"30% ({_round_currency(target_wants)}) to Wants, and 20% ({_round_currency(target_savings)}) to Savings "
                + (
                    f"requires reducing Wants spending by {_round_currency(wants_variance)} per month."
                    if wants_variance > 0
                    else "is currently well within sustainable bounds."
                )
            ),
        }
        return json.dumps(result, indent=2)

    def _compute_health_score(
        self,
        income: float,
        total_spend: float,
        needs_spend: float,
        wants_spend: float,
        target_needs: float,
        target_wants: float,
        target_savings: float,
    ) -> int:
        score = 0.0

        # Savings Rate component (max 40 pts)
        savings = income - total_spend
        savings_rate = (savings / income) if income > 0 else 0.0
        if savings_rate >= 0.20:
            score += 40.0
        elif savings_rate > 0.0:
            score += 40.0 * (savings_rate / 0.20)

        # Needs control component (max 30 pts)
        needs_ratio = (needs_spend / income) if income > 0 else 1.0
        if needs_ratio <= 0.50:
            score += 30.0
        elif needs_ratio <= 0.70:
            score += 30.0 * ((0.70 - needs_ratio) / 0.20)

        # Wants control component (max 30 pts)
        wants_ratio = (wants_spend / income) if income > 0 else 1.0
        if wants_ratio <= 0.30:
            score += 30.0
        elif wants_ratio <= 0.50:
            score += 30.0 * ((0.50 - wants_ratio) / 0.20)

        return int(max(0, min(100, round(score))))
