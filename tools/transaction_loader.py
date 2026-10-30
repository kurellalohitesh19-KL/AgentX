"""SpendWise AI - Transaction Data Loader & Schema Normalization Engine.

Provides robust, schema-agnostic loading and validation for transaction datasets.
Adheres strictly to the AgentX architecture, utilizing AGENTX_DATA_DIR.
Preserves all original fields without inventing missing columns or customer IDs.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import config

# ==============================================================================
# CANONICAL SCHEMA DEFINITION & COLUMN ALIASES
# ==============================================================================
# Canonical fields required/supported by SpendWise AI:
# - transaction_id (str): Unique transaction identifier
# - timestamp (pd.Timestamp): Normalized transaction timestamp (ISO-8601)
# - amount (float): Clean transaction numeric amount
# - category (str): Spending/merchant category
# - status (str): Transaction state (e.g., SUCCESS, FAILED, PENDING)
#
# Optional canonical fields (preserved if present, NEVER fabricated):
# - transaction_type (str): E.g., P2P, P2M, Bill Payment, Recharge
# - merchant (str): Merchant or vendor name (ONLY if present in raw data)
# - sender_bank (str): Originating financial institution
# - receiver_bank (str): Beneficiary financial institution
# - sender_state (str): Geographical region
# - device_type (str): Client device
# - network_type (str): Network connectivity
# - fraud_flag (int/str): Preserved for reference, but NOT used for anomaly detection

COLUMN_ALIASES: Dict[str, List[str]] = {
    "transaction_id": [
        "transaction id",
        "transaction_id",
        "txn_id",
        "txnid",
        "id",
        "trans_id",
        "reference_id",
        "ref_id",
    ],
    "timestamp": [
        "timestamp",
        "date",
        "datetime",
        "trans_date",
        "transaction_date",
        "created_at",
        "time",
    ],
    "amount": [
        "amount (inr)",
        "amount(inr)",
        "amount",
        "amt",
        "cost",
        "price",
        "trans_amount",
        "transaction_amount",
        "value",
    ],
    "category": [
        "merchant_category",
        "category",
        "expense_category",
        "spending_category",
        "category_name",
        "type_of_expense",
    ],
    "status": [
        "transaction_status",
        "status",
        "txn_status",
        "state",
        "payment_status",
    ],
    "transaction_type": [
        "transaction type",
        "transaction_type",
        "type",
        "txn_type",
        "payment_type",
        "transfer_type",
        "mode",
    ],
    "merchant": [
        "merchant",
        "merchant_name",
        "vendor",
        "vendor_name",
        "payee",
        "store",
        "recipient",
        "beneficiary",
    ],
    "sender_bank": [
        "sender_bank",
        "sender bank",
        "from_bank",
        "source_bank",
    ],
    "receiver_bank": [
        "receiver_bank",
        "receiver bank",
        "to_bank",
        "destination_bank",
    ],
    "sender_state": [
        "sender_state",
        "sender state",
        "state",
        "location",
        "region",
    ],
    "device_type": [
        "device_type",
        "device type",
        "device",
    ],
    "network_type": [
        "network_type",
        "network type",
        "network",
    ],
    "fraud_flag": [
        "fraud_flag",
        "fraud flag",
        "is_fraud",
        "fraud",
    ],
}


def resolve_data_path(
    filename: Optional[str] = "transactions.csv",
    data_dir: Optional[str] = None,
) -> str:
    """Resolves transaction CSV file path using AGENTX_DATA_DIR / config.DATA_DIR or direct path.

    Supports 'transactions.csv' as well as fallback to 'upi_transactions_2024.csv'.
    """
    target_name = filename if filename else "transactions.csv"

    # 1. If explicit data_dir provided, look inside it first
    if data_dir:
        candidate = os.path.join(data_dir, os.path.basename(target_name))
        if os.path.exists(candidate):
            return candidate

    # 2. If target_name is already a valid absolute or relative path, use it directly
    if os.path.exists(target_name):
        return os.path.abspath(target_name)

    # 3. Resolve using AGENTX_DATA_DIR or config.DATA_DIR
    base_dir = os.environ.get("AGENTX_DATA_DIR") or config.DATA_DIR
    path = os.path.join(base_dir, os.path.basename(target_name))

    if os.path.exists(path):
        return path

    # 4. Fallback check if default 'transactions.csv' was requested but upi file exists
    if os.path.basename(target_name) in ("transactions.csv", ""):
        fallback_path = os.path.join(base_dir, "upi_transactions_2024.csv")
        if os.path.exists(fallback_path):
            return fallback_path

    raise FileNotFoundError(
        f"Transaction data file '{target_name}' not found in data directory: '{base_dir}'. "
        f"Available files: {', '.join(os.listdir(base_dir)) if os.path.exists(base_dir) else 'Directory does not exist'}."
    )


def clean_amount_value(val: Any) -> float:
    """Parses and cleans transaction amounts from numbers, currency strings, and formatted text."""
    if pd.isna(val):
        return float("nan")
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)

    # String cleaning: strip whitespace, currency symbols, and commas
    text = str(val).strip()
    text = re.sub(r"[₹$€£\s,]", "", text)
    text = re.sub(r"^(?:inr|rs\.?)\s*", "", text, flags=re.I)
    try:
        return float(text)
    except (ValueError, TypeError):
        return float("nan")


def clean_amount_series(series: pd.Series) -> pd.Series:
    """Vectorized cleaning of an amount pandas Series."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return series.apply(clean_amount_value)


def normalize_transactions(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Normalizes raw transaction DataFrame into SpendWise canonical schema.

    - Adds canonical columns: 'transaction_id', 'timestamp', 'amount', 'category', 'status'
    - Preserves all original columns and their original names.
    - Standardizes datatypes: amount (float), timestamp (pd.to_datetime), status (str.upper).
    - Never fabricates missing customer IDs or merchant names.

    Returns:
        Tuple of (normalized_df, column_mapping_dict)
    """
    norm_df = df.copy()
    col_mapping: Dict[str, str] = {}
    normalized_headers = {
        re.sub(r"[\s_]+", " ", str(col).strip().lower()): col
        for col in df.columns
    }

    # Match canonical fields using aliases
    for canonical_name, aliases in COLUMN_ALIASES.items():
        matched_col = None
        for alias in aliases:
            cleaned_alias = re.sub(r"[\s_]+", " ", alias.lower())
            if cleaned_alias in normalized_headers:
                matched_col = normalized_headers[cleaned_alias]
                break

        if matched_col is not None:
            col_mapping[canonical_name] = matched_col

    # 1. Transaction ID
    if "transaction_id" in col_mapping:
        norm_df["transaction_id"] = norm_df[col_mapping["transaction_id"]].astype(str).str.strip()
    else:
        # Generate synthetic fallback index only if no ID column exists, without fabricating false customer IDs
        norm_df["transaction_id"] = [f"ROW_{i+1:06d}" for i in range(len(norm_df))]

    # 2. Amount
    if "amount" in col_mapping:
        norm_df["amount"] = clean_amount_series(norm_df[col_mapping["amount"]])
    else:
        norm_df["amount"] = float("nan")

    # 3. Timestamp
    if "timestamp" in col_mapping:
        norm_df["timestamp"] = pd.to_datetime(
            norm_df[col_mapping["timestamp"]], errors="coerce"
        )
    else:
        norm_df["timestamp"] = pd.NaT

    # 4. Category
    if "category" in col_mapping:
        norm_df["category"] = (
            norm_df[col_mapping["category"]].fillna("Other").astype(str).str.strip()
        )
    else:
        norm_df["category"] = "Uncategorized"

    # 5. Status
    if "status" in col_mapping:
        norm_df["status"] = (
            norm_df[col_mapping["status"]]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
            .str.upper()
        )
    else:
        norm_df["status"] = "SUCCESS"

    # 6. Optional fields (preserved if present, left absent/None if not)
    for opt_field in (
        "transaction_type",
        "merchant",
        "sender_bank",
        "receiver_bank",
        "sender_state",
        "device_type",
        "network_type",
        "fraud_flag",
    ):
        if opt_field in col_mapping and opt_field not in norm_df.columns:
            norm_df[opt_field] = norm_df[col_mapping[opt_field]]

    # 7. Financial Flow & Validity Flags (Task 1: Financial Data Validation)
    status_str = norm_df["status"].astype(str).str.upper()
    norm_df["is_successful"] = status_str.isin(["SUCCESS", "COMPLETED", "SETTLED", "CLEARED"])

    cat_str = norm_df["category"].astype(str).str.lower()
    type_str = (
        norm_df["transaction_type"].astype(str).str.lower()
        if "transaction_type" in norm_df.columns
        else pd.Series([""] * len(norm_df), index=norm_df.index)
    )

    is_inc = cat_str.str.contains(r"income|salary|deposit|payroll|credit", regex=True) | type_str.str.contains(
        r"income|credit|deposit", regex=True
    )
    is_ref = cat_str.str.contains(r"refund|reversal|chargeback", regex=True) | type_str.str.contains(
        r"refund|reversal|chargeback", regex=True
    )

    norm_df["is_income"] = is_inc
    norm_df["is_refund"] = is_ref
    norm_df["is_expense"] = norm_df["is_successful"] & (~is_inc) & (~is_ref)

    return norm_df, col_mapping


def filter_valid_expenses(df: pd.DataFrame) -> pd.DataFrame:
    """Filters only completed outgoing expense transactions, strictly excluding failed
    transactions, incoming credits/income, and refunds/reversals.
    """
    if "is_expense" in df.columns:
        return df[df["is_expense"]].copy()

    # Fallback if raw or unnormalized
    valid_mask = pd.Series([True] * len(df), index=df.index)
    if "status" in df.columns:
        valid_mask &= df["status"].astype(str).str.upper().isin(["SUCCESS", "COMPLETED", "SETTLED"])
    if "category" in df.columns:
        valid_mask &= ~df["category"].astype(str).str.lower().str.contains(
            r"income|salary|deposit|refund|reversal", regex=True
        )
    return df[valid_mask].copy()


def load_transactions(
    filename: Optional[str] = "transactions.csv",
    data_dir: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Loads transaction CSV and returns the schema-normalized DataFrame."""
    resolved_path = resolve_data_path(filename, data_dir)
    raw_df = pd.read_csv(resolved_path, nrows=max_rows, low_memory=False)
    norm_df, _ = normalize_transactions(raw_df)
    return norm_df


def validate_transactions(df: pd.DataFrame) -> Dict[str, Any]:
    """Inspects dataset integrity, data types, anomalies, and returns a detailed quality dictionary."""
    total_rows = len(df)
    total_cols = len(df.columns)

    if total_rows == 0:
        return {
            "total_rows": 0,
            "total_columns": total_cols,
            "quality_score": 0.0,
            "quality_assessment": "EMPTY_DATASET",
            "warnings": ["Dataset has 0 rows."],
        }

    # Ensure canonical columns exist for validation
    if "transaction_id" not in df.columns or "amount" not in df.columns:
        norm_df, _ = normalize_transactions(df)
    else:
        norm_df = df

    # Duplicate IDs
    unique_ids = norm_df["transaction_id"].nunique()
    duplicate_ids_count = total_rows - unique_ids
    duplicate_samples = []
    if duplicate_ids_count > 0:
        duplicate_samples = (
            norm_df[norm_df["transaction_id"].duplicated()]["transaction_id"]
            .head(5)
            .tolist()
        )

    # Timestamp validity
    ts = pd.to_datetime(norm_df["timestamp"], errors="coerce")
    invalid_timestamps_count = int(ts.isna().sum())
    valid_ts = ts.dropna()
    timestamp_range = {}
    if not valid_ts.empty:
        timestamp_range = {
            "min": valid_ts.min().strftime("%Y-%m-%d %H:%M:%S"),
            "max": valid_ts.max().strftime("%Y-%m-%d %H:%M:%S"),
        }

    # Amount statistics & checks
    amt = pd.to_numeric(norm_df["amount"], errors="coerce")
    nan_amounts = int(amt.isna().sum())
    negative_amounts = int((amt < 0).sum())
    zero_amounts = int((amt == 0).sum())

    valid_amt = amt.dropna()
    amount_stats: Dict[str, float] = {}
    if not valid_amt.empty:
        amount_stats = {
            "count": float(len(valid_amt)),
            "min": float(valid_amt.min()),
            "max": float(valid_amt.max()),
            "mean": round(float(valid_amt.mean()), 2),
            "median": round(float(valid_amt.median()), 2),
            "std": round(float(valid_amt.std()), 2) if len(valid_amt) > 1 else 0.0,
        }

    # Status distribution
    status_counts = norm_df["status"].value_counts(dropna=False).to_dict()

    # Category distribution
    category_counts = norm_df["category"].value_counts(dropna=False).to_dict()

    # Missing values across all columns
    missing_by_column = {
        col: int(norm_df[col].isna().sum())
        for col in norm_df.columns
        if int(norm_df[col].isna().sum()) > 0
    }

    # Optional column check
    missing_optional_columns = []
    for opt in (
        "merchant",
        "transaction_type",
        "sender_bank",
        "receiver_bank",
        "sender_state",
    ):
        if opt not in norm_df.columns:
            missing_optional_columns.append(opt)

    # Compute Quality Score (0 - 100)
    score = 100.0
    warnings: List[str] = []

    if duplicate_ids_count > 0:
        penalty = min(25.0, (duplicate_ids_count / total_rows) * 100.0 * 2.0)
        score -= penalty
        warnings.append(
            f"Found {duplicate_ids_count} duplicate transaction IDs (samples: {duplicate_samples})."
        )

    if invalid_timestamps_count > 0:
        penalty = min(25.0, (invalid_timestamps_count / total_rows) * 100.0 * 2.0)
        score -= penalty
        warnings.append(f"Found {invalid_timestamps_count} invalid or unparseable timestamps.")

    if nan_amounts > 0:
        penalty = min(25.0, (nan_amounts / total_rows) * 100.0 * 2.0)
        score -= penalty
        warnings.append(f"Found {nan_amounts} NaN/unparseable transaction amounts.")

    if negative_amounts > 0:
        warnings.append(f"Found {negative_amounts} negative transaction amounts.")

    if zero_amounts > 0:
        warnings.append(f"Found {zero_amounts} zero-value transactions.")

    if "merchant" in missing_optional_columns:
        warnings.append(
            "Specific merchant/vendor name column is absent; category level aggregation will be used without fabricating merchant names."
        )

    score = max(0.0, round(score, 1))
    assessment = "EXCELLENT" if score >= 95 else "GOOD" if score >= 80 else "NEEDS_CLEANING"

    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "unique_transaction_ids": unique_ids,
        "duplicate_ids_count": duplicate_ids_count,
        "duplicate_id_samples": duplicate_samples,
        "invalid_timestamps_count": invalid_timestamps_count,
        "timestamp_range": timestamp_range,
        "invalid_amounts": {
            "nan_count": nan_amounts,
            "negative_count": negative_amounts,
            "zero_count": zero_amounts,
        },
        "amount_stats": amount_stats,
        "status_distribution": status_counts,
        "category_distribution": category_counts,
        "missing_by_column": missing_by_column,
        "missing_optional_columns": missing_optional_columns,
        "data_quality_score": score,
        "quality_assessment": assessment,
        "warnings": warnings,
    }


def generate_quality_report(
    df: pd.DataFrame,
    dataset_name: str = "transactions.csv",
) -> str:
    """Generates a structured, human-readable dataset quality report."""
    v = validate_transactions(df)
    stats = v.get("amount_stats", {})
    ts_range = v.get("timestamp_range", {})

    lines = [
        f"============================================================",
        f" SPENDWISE AI - DATASET QUALITY AUDIT REPORT: {dataset_name}",
        f"============================================================",
        f"Overall Quality Score : {v['data_quality_score']}/100 ({v['quality_assessment']})",
        f"Total Records (Rows)  : {v['total_rows']:,}",
        f"Total Columns         : {v['total_columns']}",
        f"Unique Transaction IDs: {v['unique_transaction_ids']:,} (Duplicates: {v['duplicate_ids_count']})",
        f"Timestamp Integrity   : {v['total_rows'] - v['invalid_timestamps_count']:,} valid / {v['invalid_timestamps_count']} invalid",
    ]

    if ts_range:
        lines.append(f"Date Range Covered    : {ts_range.get('min')} to {ts_range.get('max')}")

    if stats:
        lines.extend([
            f"------------------------------------------------------------",
            f" TRANSACTION AMOUNT DISTRIBUTION (INR / CURRENCY)",
            f"------------------------------------------------------------",
            f"Min Amount            : {stats.get('min', 0.0):,.2f}",
            f"Max Amount            : {stats.get('max', 0.0):,.2f}",
            f"Mean Amount           : {stats.get('mean', 0.0):,.2f}",
            f"Median Amount         : {stats.get('median', 0.0):,.2f}",
            f"Std Deviation         : {stats.get('std', 0.0):,.2f}",
            f"Negative Amounts      : {v['invalid_amounts']['negative_count']}",
            f"Zero-Value Amounts    : {v['invalid_amounts']['zero_count']}",
        ])

    lines.extend([
        f"------------------------------------------------------------",
        f" TRANSACTION STATUS BREAKDOWN",
        f"------------------------------------------------------------",
    ])
    for status, count in v.get("status_distribution", {}).items():
        pct = (count / v["total_rows"]) * 100 if v["total_rows"] > 0 else 0
        lines.append(f"  * {str(status):<15}: {count:>8,} ({pct:>5.1f}%)")

    lines.extend([
        f"------------------------------------------------------------",
        f" TOP SPENDING CATEGORIES",
        f"------------------------------------------------------------",
    ])
    for cat, count in list(v.get("category_distribution", {}).items())[:8]:
        pct = (count / v["total_rows"]) * 100 if v["total_rows"] > 0 else 0
        lines.append(f"  * {str(cat):<18}: {count:>8,} ({pct:>5.1f}%)")

    if v.get("warnings"):
        lines.extend([
            f"------------------------------------------------------------",
            f" AUDIT FINDINGS & WARNINGS",
            f"------------------------------------------------------------",
        ])
        for w in v["warnings"]:
            lines.append(f"  [!] {w}")

    lines.append(f"============================================================")
    return "\n".join(lines)


class TransactionDataset:
    """Encapsulates a loaded transaction dataset with cached quality metrics."""

    def __init__(
        self,
        filename: Optional[str] = "transactions.csv",
        data_dir: Optional[str] = None,
        max_rows: Optional[int] = None,
    ):
        self.filename = filename or "transactions.csv"
        self.filepath = resolve_data_path(self.filename, data_dir)
        self.df = load_transactions(self.filename, data_dir=data_dir, max_rows=max_rows)
        self._metrics: Optional[Dict[str, Any]] = None

    @property
    def metrics(self) -> Dict[str, Any]:
        if self._metrics is None:
            self._metrics = validate_transactions(self.df)
        return self._metrics

    def quality_report(self) -> str:
        return generate_quality_report(self.df, dataset_name=os.path.basename(self.filepath))

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return f"<TransactionDataset rows={len(self.df)} file='{self.filepath}'>"
