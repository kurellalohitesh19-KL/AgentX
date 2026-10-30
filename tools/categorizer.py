"""SpendWise AI - Automatic Expense Categorizer Tool.

Categorizes transactions using available transaction attributes (categories,
transaction types, descriptions/notes, and banking metadata).
Returns deterministic category classification, rationale, confidence score,
and supporting evidence without fabricating missing fields.
"""

from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

import pandas as pd

from tools import Tool
from tools.transaction_loader import load_transactions, resolve_data_path

# Standard SpendWise Category Hierarchy
TAXONOMY_MAP = {
    # Needs
    "grocery": "Groceries",
    "groceries": "Groceries",
    "supermarket": "Groceries",
    "food": "Food & Dining",
    "restaurant": "Food & Dining",
    "dining": "Food & Dining",
    "utilities": "Utilities & Bills",
    "electricity": "Utilities & Bills",
    "water": "Utilities & Bills",
    "gas": "Utilities & Bills",
    "transport": "Transportation",
    "transportation": "Transportation",
    "fuel": "Fuel & Vehicle",
    "petrol": "Fuel & Vehicle",
    "healthcare": "Healthcare & Medical",
    "medical": "Healthcare & Medical",
    "health": "Healthcare & Medical",
    "pharmacy": "Healthcare & Medical",
    "education": "Education",
    "tuition": "Education",
    "housing": "Housing & Rent",
    "rent": "Housing & Rent",
    # Wants
    "entertainment": "Entertainment & Leisure",
    "movie": "Entertainment & Leisure",
    "streaming": "Entertainment & Leisure",
    "shopping": "Shopping & Retail",
    "apparel": "Shopping & Retail",
    "electronics": "Shopping & Retail",
    # Transfers / Other
    "recharge": "Telecom & Recharge",
    "bill payment": "Utilities & Bills",
    "p2p": "Peer-to-Peer Transfer",
    "p2m": "Merchant Payment",
    "other": "Other / Uncategorized",
}


def _round_currency(val: float | Decimal) -> float:
    return float(Decimal(str(val)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


class Categorizer(Tool):
    name = "categorize_transactions"
    description = (
        "Inspect and audit transaction expense categories with classification evidence, "
        "reasons, and confidence levels. Supports category breakdown summaries or inspecting "
        "specific transactions."
    )
    args_doc = (
        '{"filename": "transactions.csv", "transaction_id": "TXN0000000001", '
        '"filter_category": "Other", "limit": 20}'
    )

    def execute(
        self,
        filename: str = "transactions.csv",
        transaction_id: Optional[str] = None,
        filter_category: Optional[str] = None,
        limit: int = 20,
    ) -> str:
        try:
            df = load_transactions(filename)
        except Exception as e:
            return json.dumps({"error": f"Failed to load transactions: {e}"})

        if df.empty:
            return json.dumps({"error": "Transaction dataset is empty."})

        # Single transaction audit
        if transaction_id:
            return self._audit_single_transaction(df, str(transaction_id).strip())

        # Categorization analysis across dataset
        return self._categorization_summary(df, filter_category, int(limit))

    def _audit_single_transaction(self, df: pd.DataFrame, txn_id: str) -> str:
        matches = df[df["transaction_id"] == txn_id]
        if matches.empty:
            return json.dumps({
                "error": f"Transaction ID '{txn_id}' not found.",
                "available_samples": df["transaction_id"].head(3).tolist(),
            })

        row = matches.iloc[0]
        raw_cat = str(row.get("category", row.get("merchant_category", "Other"))).strip()
        txn_type = str(row.get("transaction_type", row.get("transaction type", ""))).strip()
        amt = float(row["amount"]) if pd.notna(row.get("amount")) else 0.0

        assigned_cat, reason, confidence = self._classify_record(raw_cat, txn_type)

        evidence = {
            "transaction_id": txn_id,
            "timestamp": str(row.get("timestamp")),
            "amount": _round_currency(amt),
            "status": str(row.get("status", "SUCCESS")),
            "raw_category": raw_cat,
            "assigned_category": assigned_cat,
            "transaction_type": txn_type or "UNSPECIFIED",
            "classification_reason": reason,
            "confidence_score": confidence,
            "sender_bank": str(row.get("sender_bank", "N/A")),
            "receiver_bank": str(row.get("receiver_bank", "N/A")),
        }
        return json.dumps({"status": "success", "result": evidence}, indent=2)

    def _classify_record(self, raw_cat: str, txn_type: str) -> tuple[str, str, float]:
        cat_lower = raw_cat.lower().strip()
        type_lower = txn_type.lower().strip()

        # Direct verified category match
        if cat_lower in TAXONOMY_MAP and cat_lower not in ("other", "p2p", "p2m"):
            return (
                TAXONOMY_MAP[cat_lower],
                f"Direct match from verified transaction field category='{raw_cat}'.",
                1.0,
            )

        # Reclassification via transaction type
        if type_lower == "recharge":
            return (
                "Telecom & Recharge",
                "Reclassified from transaction_type='Recharge'.",
                0.95,
            )
        if type_lower == "bill payment":
            return (
                "Utilities & Bills",
                "Reclassified from transaction_type='Bill Payment'.",
                0.95,
            )

        if cat_lower == "other":
            if type_lower == "p2p":
                return (
                    "Peer-to-Peer Transfer",
                    "Category marked 'Other'; classified as P2P transfer based on transaction type.",
                    0.80,
                )
            if type_lower == "p2m":
                return (
                    "General Merchant Payment",
                    "Category marked 'Other'; classified as P2M payment based on transaction type.",
                    0.75,
                )
            return (
                "Other / Uncategorized",
                "Generic category without specific merchant or type descriptors.",
                0.50,
            )

        return (
            raw_cat.title(),
            f"Preserved original field value '{raw_cat}'.",
            0.70,
        )

    def _categorization_summary(
        self,
        df: pd.DataFrame,
        filter_category: Optional[str],
        limit: int,
    ) -> str:
        total_records = len(df)
        total_volume = float(df["amount"].sum()) if "amount" in df.columns else 0.0

        cat_groups = df.groupby("category", dropna=False).agg(
            count=("transaction_id", "count"),
            total_amount=("amount", "sum"),
            avg_amount=("amount", "mean"),
        ).reset_index()

        cat_groups = cat_groups.sort_values(by="total_amount", ascending=False)

        summary_list = []
        for _, row in cat_groups.iterrows():
            c_name = str(row["category"])
            c_count = int(row["count"])
            c_tot = _round_currency(row["total_amount"])
            c_avg = _round_currency(row["avg_amount"])
            pct_volume = round((c_tot / total_volume * 100), 2) if total_volume > 0 else 0.0

            summary_list.append({
                "category": c_name,
                "transaction_count": c_count,
                "total_spend": c_tot,
                "average_spend": c_avg,
                "percent_of_total_spend": pct_volume,
            })

        # Sample records needing attention (e.g., Other or specific filtered category)
        sample_records = []
        target_cat = filter_category.strip() if filter_category else None
        if target_cat:
            filtered_df = df[df["category"].str.lower() == target_cat.lower()]
        else:
            filtered_df = df[df["category"].str.lower().isin(["other", "uncategorized"])]

        for _, r in filtered_df.head(limit).iterrows():
            txn_type = str(r.get("transaction_type", r.get("transaction type", "")))
            assigned, reason, conf = self._classify_record(str(r["category"]), txn_type)
            sample_records.append({
                "transaction_id": str(r["transaction_id"]),
                "timestamp": str(r.get("timestamp")),
                "amount": _round_currency(float(r["amount"])),
                "raw_category": str(r["category"]),
                "recommended_category": assigned,
                "confidence": conf,
                "reason": reason,
            })

        result = {
            "status": "success",
            "total_transactions": total_records,
            "total_spending": _round_currency(total_volume),
            "category_breakdown": summary_list[:limit],
            "filtered_samples": sample_records,
            "notes": (
                "Category mapping grounds spending in verified transaction metadata. "
                "No customer identities were assumed."
            ),
        }
        return json.dumps(result, indent=2)
