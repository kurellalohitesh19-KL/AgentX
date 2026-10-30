"""SpendWise AI - Subscription & Recurring Payment Detector Tool.

Identifies periodic recurring charges (monthly, weekly, quarterly) using interval statistics.
Requires adequate cadence evidence (minimum cycles, low interval variance) before
classifying payments as subscriptions.
Detects price creep / rate escalations over time.
Explicitly identifies and reports missing merchant or customer fields without fabricating data.
"""

from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from tools import Tool
from tools.transaction_loader import load_transactions


def _round_currency(val: float | Decimal) -> float:
    return float(Decimal(str(val)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


class SubscriptionDetector(Tool):
    name = "detect_subscriptions"
    description = (
        "Identify recurring payments and subscriptions with verified interval cadence "
        "(monthly, weekly, quarterly). Detects price creep and reports missing merchant/customer "
        "fields without fabricating identities."
    )
    args_doc = (
        '{"filename": "transactions.csv", "min_cycles": 3, "tolerance_days": 4, "max_results": 20}'
    )

    def execute(
        self,
        filename: str = "transactions.csv",
        min_cycles: int = 3,
        tolerance_days: float = 4.0,
        max_results: int = 20,
    ) -> str:
        try:
            df = load_transactions(filename)
        except Exception as e:
            return json.dumps({"error": f"Failed to load transactions: {e}"})

        if df.empty:
            return json.dumps({"error": "Transaction dataset is empty."})

        # Identify missing fields explicitly
        missing_fields: List[str] = []
        if "merchant" not in df.columns or df["merchant"].isna().all():
            missing_fields.append("merchant")
        if "customer_id" not in df.columns:
            missing_fields.append("customer_id")

        min_c = max(2, int(min_cycles)) if min_cycles else 3
        tol_days = float(tolerance_days) if tolerance_days else 4.0
        limit = int(max_results) if max_results else 20

        # Successful or valid transactions only
        if "status" in df.columns:
            valid_df = df[df["status"] == "SUCCESS"].copy()
        else:
            valid_df = df.copy()

        valid_df["ts"] = pd.to_datetime(valid_df["timestamp"], errors="coerce")
        valid_df["amount"] = pd.to_numeric(valid_df["amount"], errors="coerce")
        valid_df = valid_df.dropna(subset=["ts", "amount"]).sort_values(by="ts")

        # Determine grouping keys based on actual fields present
        has_merchant = "merchant" not in missing_fields
        if has_merchant:
            group_keys = ["merchant"]
        else:
            # Group by category, payment type, destination bank and rounded amount
            group_keys = ["category"]
            if "transaction_type" in valid_df.columns:
                group_keys.append("transaction_type")
            if "receiver_bank" in valid_df.columns:
                group_keys.append("receiver_bank")

        confirmed_subs: List[Dict[str, Any]] = []
        irregular_recurring: List[Dict[str, Any]] = []
        total_annual_commitment = 0.0

        for key_vals, group in valid_df.groupby(group_keys):
            if len(group) < min_c:
                continue

            sub_info = self._analyze_cadence(
                group, key_vals, group_keys, min_c, tol_days
            )
            if not sub_info:
                continue

            if sub_info["is_confirmed_subscription"]:
                confirmed_subs.append(sub_info)
                total_annual_commitment += sub_info.get("estimated_annual_cost", 0.0)
            else:
                irregular_recurring.append(sub_info)

        # Sort confirmed subscriptions by annual commitment descending
        confirmed_subs.sort(key=lambda x: x.get("estimated_annual_cost", 0.0), reverse=True)

        result = {
            "status": "success",
            "summary": {
                "total_records_evaluated": len(valid_df),
                "confirmed_subscriptions_count": len(confirmed_subs),
                "irregular_recurring_count": len(irregular_recurring),
                "total_estimated_annual_commitment": _round_currency(total_annual_commitment),
                "missing_fields_identified": missing_fields,
            },
            "confirmed_subscriptions": confirmed_subs[:limit],
            "irregular_recurring_patterns": irregular_recurring[:limit],
            "methodology": (
                "Verified cadence consistency using mean interval and standard deviation. "
                "Transactions with erratic intervals are not classified as subscriptions. "
                + (
                    "Notice: Merchant identifier is absent in dataset; recurring grouping used "
                    + f"{group_keys} without fabricating merchant names."
                    if not has_merchant
                    else "Grouped by verified merchant identifier."
                )
            ),
        }
        return json.dumps(result, indent=2)

    def _analyze_cadence(
        self,
        group: pd.DataFrame,
        key_vals: Any,
        group_keys: List[str],
        min_cycles: int,
        tolerance_days: float,
    ) -> Optional[Dict[str, Any]]:
        sorted_g = group.sort_values(by="ts")
        times = sorted_g["ts"].tolist()
        amounts = sorted_g["amount"].tolist()
        txn_ids = sorted_g["transaction_id"].tolist()

        n = len(times)
        if n < min_cycles:
            return None

        # Compute intervals in days
        intervals = [(times[i + 1] - times[i]).total_seconds() / 86400.0 for i in range(n - 1)]
        mean_interval = float(np.mean(intervals))
        std_interval = float(np.std(intervals, ddof=1)) if len(intervals) > 1 else 0.0

        # Cadence classification
        cadence = "IRREGULAR"
        annual_multiplier = 0.0
        is_subscription = False

        if 26.0 <= mean_interval <= 34.0 and std_interval <= tolerance_days:
            cadence = "MONTHLY"
            annual_multiplier = 12.0
            is_subscription = True
        elif 6.0 <= mean_interval <= 8.5 and std_interval <= (tolerance_days / 2.0):
            cadence = "WEEKLY"
            annual_multiplier = 52.0
            is_subscription = True
        elif 13.0 <= mean_interval <= 16.0 and std_interval <= (tolerance_days / 2.0):
            cadence = "BI-WEEKLY"
            annual_multiplier = 26.0
            is_subscription = True
        elif 85.0 <= mean_interval <= 96.0 and std_interval <= (tolerance_days * 2.0):
            cadence = "QUARTERLY"
            annual_multiplier = 4.0
            is_subscription = True

        latest_amount = float(amounts[-1])
        earliest_amount = float(amounts[0])
        annual_cost = latest_amount * annual_multiplier if is_subscription else 0.0

        # Price creep detection
        price_creep = None
        if amounts[-1] > amounts[0]:
            diff = amounts[-1] - amounts[0]
            pct_increase = round((diff / amounts[0]) * 100.0, 1) if amounts[0] > 0 else 0.0
            price_creep = {
                "detected": True,
                "initial_amount": _round_currency(earliest_amount),
                "latest_amount": _round_currency(latest_amount),
                "increase_per_cycle": _round_currency(diff),
                "percentage_increase": f"+{pct_increase}%",
                "annualized_extra_cost": _round_currency(diff * annual_multiplier),
            }

        # Format label
        label_dict = dict(zip(group_keys, [key_vals] if len(group_keys) == 1 else list(key_vals)))

        return {
            "entity": label_dict,
            "is_confirmed_subscription": is_subscription,
            "cadence": cadence,
            "cycle_count": n,
            "mean_interval_days": round(mean_interval, 1),
            "interval_std_dev_days": round(std_interval, 2),
            "latest_amount": _round_currency(latest_amount),
            "estimated_annual_cost": _round_currency(annual_cost),
            "first_transaction_date": times[0].strftime("%Y-%m-%d"),
            "latest_transaction_date": times[-1].strftime("%Y-%m-%d"),
            "price_creep": price_creep,
            "evidence_transaction_ids": [str(t) for t in txn_ids[:5]],
        }
