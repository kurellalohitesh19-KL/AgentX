"""SpendWise AI - Statistical Spending Anomaly & Surge Detector Tool.

Uses explainable statistical baselines (IQR fences, Z-scores, and periodic category surges).
Handles insufficient history with transparent guardrails.
Avoids false fraud claims by framing findings as statistical expense anomalies.
Provides full transaction evidence and deviation metrics.
"""

from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from tools import Tool
from tools.transaction_loader import load_transactions

MIN_HISTORY_RECORDS = 10


def _round_currency(val: float | Decimal) -> float:
    return float(Decimal(str(val)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


class AnomalyDetector(Tool):
    name = "detect_anomalies"
    description = (
        "Detect statistical spending anomalies, outliers, and category spending surges. "
        "Uses explainable IQR and Z-score baselines per category. Handles insufficient history "
        "gracefully without making false fraud claims."
    )
    args_doc = (
        '{"filename": "transactions.csv", "category": "Food", '
        '"method": "auto", "sensitivity": 2.5, "max_results": 20}'
    )

    def execute(
        self,
        filename: str = "transactions.csv",
        category: Optional[str] = None,
        method: str = "auto",
        sensitivity: float = 2.5,
        max_results: int = 20,
    ) -> str:
        try:
            df = load_transactions(filename)
        except Exception as e:
            return json.dumps({"error": f"Failed to load transactions: {e}"})

        if df.empty:
            return json.dumps({"error": "Transaction dataset is empty."})

        # Filter only successful transactions for spending baseline (or all if status column absent)
        if "status" in df.columns:
            valid_df = df[df["status"] == "SUCCESS"].copy()
        else:
            valid_df = df.copy()

        valid_df["amount"] = pd.to_numeric(valid_df["amount"], errors="coerce")
        valid_df = valid_df.dropna(subset=["amount"])

        limit = int(max_results) if max_results else 20
        sens = float(sensitivity) if sensitivity else 2.5

        # Category filter if requested
        if category and category.strip():
            target_cat = category.strip()
            cat_df = valid_df[valid_df["category"].str.lower() == target_cat.lower()]
            if cat_df.empty:
                return json.dumps({
                    "error": f"No transactions found for category '{target_cat}'.",
                    "available_categories": valid_df["category"].unique().tolist(),
                })
            anomalies, surges, summary_item = self._analyze_category(cat_df, target_cat, sens, limit)
            summaries = [summary_item] if summary_item else []
        else:
            # Multi-category sweep
            anomalies = []
            surges = []
            summaries = []
            for cat_name, cat_df in valid_df.groupby("category"):
                c_anom, c_surge, c_sum = self._analyze_category(cat_df, str(cat_name), sens, limit)
                anomalies.extend(c_anom)
                surges.extend(c_surge)
                if c_sum:
                    summaries.append(c_sum)

            # Sort anomalies by z-score descending
            anomalies.sort(key=lambda x: x.get("z_score", 0.0), reverse=True)

        result = {
            "status": "success",
            "summary": {
                "total_transactions_evaluated": len(valid_df),
                "total_anomalies_detected": len(anomalies),
                "category_surges_detected": len(surges),
                "sensitivity_threshold": sens,
            },
            "anomalous_transactions": anomalies[:limit],
            "category_spending_surges": surges[:limit],
            "category_baselines": summaries[:limit],
            "statistical_basis": (
                "Anomalies detected using IQR upper fences (Q3 + 1.5*IQR) and Z-score deviations "
                "relative to category historical distributions. Findings represent statistical "
                "spending outliers, not malicious fraud accusations."
            ),
        }
        return json.dumps(result, indent=2)

    def _analyze_category(
        self,
        df: pd.DataFrame,
        cat_name: str,
        sensitivity: float,
        limit: int,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        n = len(df)
        if n < MIN_HISTORY_RECORDS:
            summary = {
                "category": cat_name,
                "status": "INSUFFICIENT_HISTORY",
                "sample_count": n,
                "message": (
                    f"Only {n} records available. Minimum {MIN_HISTORY_RECORDS} records "
                    f"required for statistical significance."
                ),
            }
            return [], [], summary

        amounts = df["amount"].values
        mean_val = float(np.mean(amounts))
        std_val = float(np.std(amounts, ddof=1)) if n > 1 else 0.0
        median_val = float(np.median(amounts))
        q1 = float(np.percentile(amounts, 25))
        q3 = float(np.percentile(amounts, 75))
        iqr = q3 - q1
        iqr_upper_fence = q3 + (1.5 * iqr)
        iqr_extreme_fence = q3 + (3.0 * iqr)

        summary = {
            "category": cat_name,
            "status": "ESTABLISHED_BASELINE",
            "sample_count": n,
            "mean": _round_currency(mean_val),
            "median": _round_currency(median_val),
            "std_dev": _round_currency(std_val),
            "q1": _round_currency(q1),
            "q3": _round_currency(q3),
            "iqr": _round_currency(iqr),
            "upper_fence_1_5_iqr": _round_currency(iqr_upper_fence),
        }

        anomalies: List[Dict[str, Any]] = []

        # Find outliers
        for _, row in df.iterrows():
            amt = float(row["amount"])
            z_score = (amt - mean_val) / std_val if std_val > 0 else 0.0
            is_iqr_outlier = amt > iqr_upper_fence
            is_z_outlier = z_score >= sensitivity

            if is_iqr_outlier or is_z_outlier:
                ratio_to_median = round(amt / median_val, 1) if median_val > 0 else 0.0
                severity = (
                    "EXTREME"
                    if (amt > iqr_extreme_fence or z_score >= 3.5)
                    else "HIGH"
                    if z_score >= 2.8
                    else "MODERATE"
                )

                anomalies.append({
                    "transaction_id": str(row["transaction_id"]),
                    "timestamp": str(row.get("timestamp")),
                    "category": cat_name,
                    "amount": _round_currency(amt),
                    "category_median": _round_currency(median_val),
                    "category_mean": _round_currency(mean_val),
                    "z_score": round(float(z_score), 2),
                    "ratio_to_median": f"{ratio_to_median}x",
                    "severity": severity,
                    "explanation": (
                        f"Amount of {_round_currency(amt)} is {ratio_to_median}x the {cat_name} "
                        f"median ({_round_currency(median_val)}) with Z-score {round(float(z_score), 2)}. "
                        f"Exceeds IQR upper fence of {_round_currency(iqr_upper_fence)}."
                    ),
                    "sender_bank": str(row.get("sender_bank", "N/A")),
                    "receiver_bank": str(row.get("receiver_bank", "N/A")),
                })

        # Detect monthly spending surges in category
        surges = self._detect_period_surges(df, cat_name)

        return anomalies, surges, summary

    def _detect_period_surges(
        self, df: pd.DataFrame, cat_name: str
    ) -> List[Dict[str, Any]]:
        """Detects monthly surges where spending surges > 1.4x above trailing median."""
        if "timestamp" not in df.columns:
            return []

        ts_series = pd.to_datetime(df["timestamp"], errors="coerce")
        temp_df = df.copy()
        temp_df["period"] = ts_series.dt.to_period("M")
        valid_periods = temp_df.dropna(subset=["period"])

        if valid_periods["period"].nunique() < 2:
            return []

        monthly_totals = (
            valid_periods.groupby("period")["amount"].sum().reset_index()
        )
        monthly_totals["period_str"] = monthly_totals["period"].astype(str)

        median_monthly = float(monthly_totals["amount"].median())
        surges: List[Dict[str, Any]] = []

        for _, row in monthly_totals.iterrows():
            m_spend = float(row["amount"])
            if median_monthly > 0 and (m_spend / median_monthly) >= 1.4:
                pct_surge = round(((m_spend - median_monthly) / median_monthly) * 100, 1)
                surges.append({
                    "category": cat_name,
                    "period": str(row["period_str"]),
                    "period_spending": _round_currency(m_spend),
                    "category_monthly_median": _round_currency(median_monthly),
                    "surge_percentage": f"+{pct_surge}%",
                    "finding": (
                        f"Monthly spend in {cat_name} for {row['period_str']} ({_round_currency(m_spend)}) "
                        f"surged by {pct_surge}% over the monthly median ({_round_currency(median_monthly)})."
                    ),
                })

        return surges
