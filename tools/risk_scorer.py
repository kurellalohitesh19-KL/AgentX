"""SpendWise AI - Explainable Financial Risk Scoring Engine.

Computes transparent, multi-factor risk ratings (0 - 100) and actionable recommendations
for transactions based on statistical deviations, duplicate payment proximity, and velocity.
Strictly avoids labeling statistical anomalies as criminal fraud.
"""

from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from tools import Tool
from tools.transaction_loader import load_transactions, filter_valid_expenses


def _round_currency(val: float | Decimal) -> float:
    return float(Decimal(str(val)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


class TransactionRiskScorer:
    """Calculates explainable financial risk scores and recommendations."""

    def __init__(self, df: pd.DataFrame):
        self.raw_df = df
        self.df = filter_valid_expenses(df) if "is_expense" in df.columns else df.copy()
        self._category_medians: Dict[str, float] = {}
        self._category_iqr_fences: Dict[str, float] = {}
        self._calculate_baselines()

    def _calculate_baselines(self):
        if self.df.empty or "amount" not in self.df.columns or "category" not in self.df.columns:
            return

        valid = self.df.dropna(subset=["amount"]).copy()
        valid["amount"] = pd.to_numeric(valid["amount"], errors="coerce")

        for cat, group in valid.groupby("category"):
            amts = group["amount"].dropna().values
            if len(amts) >= 5:
                med = float(np.median(amts))
                q3 = float(np.percentile(amts, 75))
                q1 = float(np.percentile(amts, 25))
                iqr = q3 - q1
                self._category_medians[str(cat)] = med
                self._category_iqr_fences[str(cat)] = q3 + (1.5 * iqr)
            elif len(amts) > 0:
                self._category_medians[str(cat)] = float(np.median(amts))
                self._category_iqr_fences[str(cat)] = float(np.max(amts))

    def evaluate_transaction(
        self,
        row: pd.Series | Dict[str, Any],
        duplicate_ids_set: Optional[set] = None,
        suspected_duplicate_keys: Optional[Dict[tuple, int]] = None,
    ) -> Dict[str, Any]:
        """Evaluates a single transaction and returns score, category, reasons, and actions."""
        reasons: List[str] = []
        score = 0.0

        txn_id = str(row.get("transaction_id", "N/A"))
        amt = float(row.get("amount", 0.0)) if pd.notna(row.get("amount")) else 0.0
        cat = str(row.get("category", "Other"))

        # 1. Statistical Outlier Component (0 to 40 points)
        med = self._category_medians.get(cat, 0.0)
        fence = self._category_iqr_fences.get(cat, 0.0)

        if med > 0:
            ratio = amt / med
            if ratio >= 15.0:
                score += 40.0
                reasons.append(f"Extreme statistical outlier: amount of ₹{amt:,.2f} is {ratio:.1f}x the {cat} median (₹{med:,.2f}). (+40 pts)")
            elif ratio >= 8.0:
                score += 30.0
                reasons.append(f"Severe spending spike: amount of ₹{amt:,.2f} is {ratio:.1f}x the {cat} median (₹{med:,.2f}). (+30 pts)")
            elif ratio >= 4.0:
                score += 20.0
                reasons.append(f"Significant spending deviation: amount is {ratio:.1f}x the {cat} median. (+20 pts)")
            elif ratio >= 2.0:
                score += 10.0
                reasons.append(f"Moderate spending deviation: amount is {ratio:.1f}x the {cat} median. (+10 pts)")

        if fence > 0 and amt > fence and score < 30.0:
            score += 10.0
            reasons.append(f"Exceeds category statistical IQR upper fence of ₹{fence:,.2f}. (+10 pts)")

        # 2. Duplicate / Ledger Replay Component (0 to 35 points)
        if duplicate_ids_set and txn_id in duplicate_ids_set:
            score += 35.0
            reasons.append(f"Confirmed key collision: Transaction ID '{txn_id}' appears multiple times in ledger. (+35 pts)")

        if suspected_duplicate_keys:
            key = (amt, cat)
            occurrences = suspected_duplicate_keys.get(key, 0)
            if occurrences >= 2:
                score += 20.0
                reasons.append(f"Matching amount (₹{amt:,.2f}) and category repeated within evaluation window. (+20 pts)")

        # 3. Off-Peak / Time Context Component (0 to 15 points)
        ts_val = row.get("timestamp")
        if pd.notna(ts_val):
            try:
                dt = pd.to_datetime(ts_val)
                hour = dt.hour
                if 1 <= hour <= 4:
                    score += 15.0
                    reasons.append(f"Off-peak execution: Transaction occurred at {dt.strftime('%H:%M:%S')} (unusual nocturnal window). (+15 pts)")
            except Exception:
                pass

        # 4. Status Flag Check
        status_val = str(row.get("status", "SUCCESS")).upper()
        if status_val == "FAILED":
            reasons.append("Status note: Payment failed/declined; zero capital debit occurred.")

        score = max(0.0, min(100.0, round(score, 1)))

        # Categorize Risk
        if score >= 75.0:
            risk_cat = "CRITICAL"
            action = "Immediate review required: verify authorization and examine for duplicate charge or abnormal billing."
        elif score >= 50.0:
            risk_cat = "HIGH"
            action = "Audit receipt: confirm pricing and verify that transaction aligns with expected expenditure."
        elif score >= 25.0:
            risk_cat = "MEDIUM"
            action = "Monitor: review during routine monthly expense audit."
        else:
            risk_cat = "LOW"
            action = "Standard verified transaction — no administrative action needed."

        return {
            "transaction_id": txn_id,
            "timestamp": str(row.get("timestamp", "N/A")),
            "amount": _round_currency(amt),
            "category": cat,
            "status": status_val,
            "risk_score": score,
            "risk_category": risk_cat,
            "reasons": reasons if reasons else ["Normal transaction within baseline category parameters."],
            "recommended_action": action,
            "evidence_disclaimer": "Heuristic financial risk rating based on statistical deviations and proximity. Not a claim of criminal fraud.",
        }

    def evaluate_dataset(self, min_score: float = 0.0, limit: int = 50) -> List[Dict[str, Any]]:
        """Evaluates transactions across the active dataset and returns scored records."""
        if self.raw_df.empty:
            return []

        # Find duplicate IDs
        dup_ids = set(self.raw_df[self.raw_df["transaction_id"].duplicated(keep=False)]["transaction_id"].dropna().tolist())

        # Count identical amounts and categories
        dup_counts = self.raw_df.groupby(["amount", "category"]).size().to_dict()

        results = []
        for _, row in self.raw_df.iterrows():
            item = self.evaluate_transaction(row, duplicate_ids_set=dup_ids, suspected_duplicate_keys=dup_counts)
            if item["risk_score"] >= min_score:
                results.append(item)

        results.sort(key=lambda x: x["risk_score"], reverse=True)
        return results[:limit]


class RiskScorerTool(Tool):
    name = "score_transaction_risks"
    description = (
        "Calculate explainable multi-factor financial risk scores (0-100) and actionable recommendations "
        "for transactions based on statistical deviations, duplicate proximity, and timing without claiming fraud."
    )
    args_doc = (
        '{"filename": "transactions.csv", "min_score": 50, "limit": 20}'
    )

    def execute(
        self,
        filename: str = "transactions.csv",
        min_score: float = 50.0,
        limit: int = 20,
    ) -> str:
        try:
            df = load_transactions(filename)
        except Exception as e:
            return json.dumps({"error": f"Failed to load transactions: {e}"})

        if df.empty:
            return json.dumps({"error": "Transaction dataset is empty."})

        scorer = TransactionRiskScorer(df)
        scored = scorer.evaluate_dataset(min_score=float(min_score), limit=int(limit))

        summary = {
            "total_evaluated": len(df),
            "flagged_high_or_critical": sum(1 for s in scored if s["risk_category"] in ("HIGH", "CRITICAL")),
            "min_score_filter": float(min_score),
        }

        return json.dumps({
            "status": "success",
            "summary": summary,
            "scored_transactions": scored,
        }, indent=2)
