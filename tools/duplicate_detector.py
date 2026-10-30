"""SpendWise AI - Duplicate Payment & Ledger Anomaly Detector Tool.

Distinguishes between:
1. Confirmed Duplicate Identifiers (Definitive ledger replay / ID collision)
2. Potential Duplicate Payments (Same amount, category, bank within short time windows)

Provides complete transaction evidence and potential recoverable leak totals.
Avoids false fraud accusations by classifying events as payment gateway retries,
ledger duplicates, or possible double debits.
"""

from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

import pandas as pd

from tools import Tool
from tools.transaction_loader import load_transactions


def _round_currency(val: float | Decimal) -> float:
    return float(Decimal(str(val)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


class DuplicateDetector(Tool):
    name = "detect_duplicates"
    description = (
        "Detect duplicate payments and ledger replays. Distinguishes confirmed duplicate IDs "
        "from possible double-charges or payment retries within time windows. Returns evidence "
        "and potential recoverable leak calculations."
    )
    args_doc = (
        '{"filename": "transactions.csv", "window_hours": 24, "max_results": 20}'
    )

    def execute(
        self,
        filename: str = "transactions.csv",
        window_hours: float = 24.0,
        max_results: int = 20,
    ) -> str:
        try:
            df = load_transactions(filename)
        except Exception as e:
            return json.dumps({"error": f"Failed to load transactions: {e}"})

        if df.empty:
            return json.dumps({"error": "Transaction dataset is empty."})

        w_hours = float(window_hours) if window_hours else 24.0
        limit = int(max_results) if max_results else 20

        # Layer 1: Confirmed Duplicate Identifiers
        confirmed_id_findings, confirmed_leak = self._find_confirmed_duplicate_ids(df)

        # Layer 2: Potential Duplicate Payments (behavioral / time window)
        potential_payment_findings, potential_leak = self._find_potential_duplicate_payments(
            df, window_hours=w_hours, limit=limit
        )

        total_leak = _round_currency(confirmed_leak + potential_leak)

        result = {
            "status": "success",
            "summary": {
                "total_records_audited": len(df),
                "confirmed_duplicate_id_clusters": len(confirmed_id_findings),
                "potential_duplicate_payment_clusters": len(potential_payment_findings),
                "potential_recoverable_leak_total": total_leak,
                "window_hours_evaluated": w_hours,
            },
            "confirmed_duplicate_ids": confirmed_id_findings[:limit],
            "potential_duplicate_payments": potential_payment_findings[:limit],
            "audit_methodology": (
                "Identified confirmed ID replays via key collision. "
                "Identified behavioral duplicate payments via identical amount, category, "
                "and banking endpoints within specified window without assuming customer identities."
            ),
        }
        return json.dumps(result, indent=2)

    def _find_confirmed_duplicate_ids(
        self, df: pd.DataFrame
    ) -> tuple[List[Dict[str, Any]], float]:
        """Detects identical transaction IDs (ledger replay bugs)."""
        dup_mask = df["transaction_id"].duplicated(keep=False)
        if not dup_mask.any():
            return [], 0.0

        dup_df = df[dup_mask].sort_values(by=["transaction_id", "timestamp"])
        findings: List[Dict[str, Any]] = []
        total_leak = 0.0

        for txn_id, group in dup_df.groupby("transaction_id"):
            records = []
            for _, r in group.iterrows():
                records.append({
                    "timestamp": str(r.get("timestamp")),
                    "amount": _round_currency(float(r["amount"])),
                    "status": str(r.get("status", "SUCCESS")),
                    "category": str(r.get("category", "N/A")),
                    "sender_bank": str(r.get("sender_bank", "N/A")),
                    "receiver_bank": str(r.get("receiver_bank", "N/A")),
                })

            # Calculate leak: all occurrences after the first successful charge
            successful_charges = [
                float(r["amount"])
                for _, r in group.iterrows()
                if str(r.get("status", "")).upper() == "SUCCESS"
            ]
            leak = sum(successful_charges[1:]) if len(successful_charges) > 1 else 0.0
            total_leak += leak

            findings.append({
                "type": "CONFIRMED_DUPLICATE_IDENTIFIER",
                "transaction_id": str(txn_id),
                "occurrences": len(group),
                "potential_leak": _round_currency(leak),
                "evidence_transactions": records,
                "finding": f"Transaction ID '{txn_id}' appears {len(group)} times in the transaction log.",
            })

        return findings, total_leak

    def _find_potential_duplicate_payments(
        self,
        df: pd.DataFrame,
        window_hours: float,
        limit: int,
    ) -> tuple[List[Dict[str, Any]], float]:
        """Detects distinct transactions sharing identical amount and category within a tight time window."""
        valid_df = df.dropna(subset=["timestamp", "amount"]).copy()
        if valid_df.empty:
            return [], 0.0

        # Grouping attributes
        group_cols = ["amount", "category"]
        if "merchant" in valid_df.columns and valid_df["merchant"].notna().any():
            group_cols.append("merchant")
        elif "sender_bank" in valid_df.columns and "receiver_bank" in valid_df.columns:
            group_cols.extend(["sender_bank", "receiver_bank"])

        valid_df["ts"] = pd.to_datetime(valid_df["timestamp"], errors="coerce")
        valid_df = valid_df.dropna(subset=["ts"]).sort_values(by="ts")

        findings: List[Dict[str, Any]] = []
        total_leak = 0.0

        window_seconds = window_hours * 3600.0

        # Iterate through candidate groups
        grouped = valid_df.groupby(group_cols)
        for keys, group in grouped:
            if len(group) < 2:
                continue

            group_rows = group.to_dict(orient="records")
            n = len(group_rows)

            cluster_matches = []
            for i in range(n - 1):
                t1 = group_rows[i]["ts"]
                t2 = group_rows[i + 1]["ts"]
                delta_sec = (t2 - t1).total_seconds()

                if 0 <= delta_sec <= window_seconds:
                    cluster_matches.append((group_rows[i], group_rows[i + 1], delta_sec))

            for r1, r2, delta_sec in cluster_matches:
                s1 = str(r1.get("status", "SUCCESS")).upper()
                s2 = str(r2.get("status", "SUCCESS")).upper()
                amt = float(r1["amount"])
                minutes_apart = round(delta_sec / 60.0, 1)

                # Assess risk and categorization
                if (s1 == "FAILED" and s2 == "SUCCESS") or (s1 == "SUCCESS" and s2 == "FAILED"):
                    risk = "LOW (RETRY / FAILED ATTEMPT)"
                    leak = 0.0
                    explanation = (
                        f"One payment succeeded and one failed {minutes_apart} minutes apart. "
                        f"Expected payment gateway retry or failed follow-up attempt."
                    )
                elif s1 == "SUCCESS" and s2 == "SUCCESS":
                    if minutes_apart <= 15:
                        risk = "HIGH (SUSPECTED DOUBLE CHARGE)"
                        leak = amt
                        explanation = (
                            f"Identical successful payments occurred {minutes_apart} minutes apart. "
                            f"Probable duplicate POS swipe or payment gateway double billing."
                        )
                    else:
                        risk = "MEDIUM (POTENTIAL REPEAT PAYMENT)"
                        leak = amt
                        explanation = (
                            f"Identical payment repeated within {minutes_apart} minutes ({round(minutes_apart/60, 1)} hours). "
                            f"May be intentional repeated purchase or duplicate debit."
                        )
                else:
                    risk = "LOW"
                    leak = 0.0
                    explanation = f"Transactions occurred {minutes_apart} minutes apart with mixed statuses."

                total_leak += leak
                findings.append({
                    "type": "POTENTIAL_DUPLICATE_PAYMENT",
                    "risk_level": risk,
                    "time_interval_minutes": minutes_apart,
                    "amount": _round_currency(amt),
                    "potential_recoverable_leak": _round_currency(leak),
                    "explanation": explanation,
                    "first_transaction": {
                        "transaction_id": str(r1["transaction_id"]),
                        "timestamp": str(r1["timestamp"]),
                        "amount": _round_currency(float(r1["amount"])),
                        "status": s1,
                        "category": str(r1.get("category")),
                        "sender_bank": str(r1.get("sender_bank", "N/A")),
                        "receiver_bank": str(r1.get("receiver_bank", "N/A")),
                    },
                    "second_transaction": {
                        "transaction_id": str(r2["transaction_id"]),
                        "timestamp": str(r2["timestamp"]),
                        "amount": _round_currency(float(r2["amount"])),
                        "status": s2,
                        "category": str(r2.get("category")),
                        "sender_bank": str(r2.get("sender_bank", "N/A")),
                        "receiver_bank": str(r2.get("receiver_bank", "N/A")),
                    },
                })

                if len(findings) >= limit:
                    break
            if len(findings) >= limit:
                break

        return findings, total_leak
