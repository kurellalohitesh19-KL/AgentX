"""Offline unit tests for SpendWise AI Transaction Loader & Validation Engine.

Run: python -m unittest discover tests
"""

import os
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from tools.transaction_loader import (
    TransactionDataset,
    clean_amount_series,
    clean_amount_value,
    generate_quality_report,
    load_transactions,
    normalize_transactions,
    resolve_data_path,
    validate_transactions,
)

SYNTHETIC_CSV_PATH = os.path.join(
    os.path.dirname(__file__), "data", "synthetic_transactions.csv"
)


class TestTransactionLoader(unittest.TestCase):
    def setUp(self):
        self.assertTrue(
            os.path.exists(SYNTHETIC_CSV_PATH),
            f"Synthetic test data not found at {SYNTHETIC_CSV_PATH}",
        )
        self.raw_df = pd.read_csv(SYNTHETIC_CSV_PATH)

    def test_clean_amount_value(self):
        self.assertEqual(clean_amount_value(500), 500.0)
        self.assertEqual(clean_amount_value(12.34), 12.34)
        self.assertEqual(clean_amount_value("1,500.50"), 1500.50)
        self.assertEqual(clean_amount_value("₹2,499.50"), 2499.50)
        self.assertEqual(clean_amount_value("$100.00"), 100.0)
        self.assertEqual(clean_amount_value("INR 350"), 350.0)
        self.assertEqual(clean_amount_value("Rs. 499.99"), 499.99)
        self.assertTrue(np.isnan(clean_amount_value("invalid_val")))
        self.assertTrue(np.isnan(clean_amount_value(None)))

    def test_clean_amount_series(self):
        s = pd.Series(["₹100", "200.50", "₹1,000", "invalid"])
        cleaned = clean_amount_series(s)
        self.assertEqual(cleaned[0], 100.0)
        self.assertEqual(cleaned[1], 200.50)
        self.assertEqual(cleaned[2], 1000.0)
        self.assertTrue(np.isnan(cleaned[3]))

    def test_normalize_transactions_schema(self):
        norm_df, mapping = normalize_transactions(self.raw_df)

        # Check canonical columns are added
        self.assertIn("transaction_id", norm_df.columns)
        self.assertIn("timestamp", norm_df.columns)
        self.assertIn("amount", norm_df.columns)
        self.assertIn("category", norm_df.columns)
        self.assertIn("status", norm_df.columns)

        # Check all original columns are strictly preserved
        for col in self.raw_df.columns:
            self.assertIn(col, norm_df.columns)

        # Check amount cleaning on the formatted row (SYN006: '₹2,499.50')
        row_6 = norm_df[norm_df["transaction_id"] == "SYN006"].iloc[0]
        self.assertEqual(row_6["amount"], 2499.50)

        # Check category mapping
        row_1 = norm_df[norm_df["transaction_id"] == "SYN001"].iloc[0]
        self.assertEqual(row_1["category"], "Grocery")
        self.assertEqual(row_1["status"], "SUCCESS")

    def test_no_hallucination_of_missing_columns(self):
        norm_df, _ = normalize_transactions(self.raw_df)
        # Raw data has NO 'merchant' column and NO customer ID column.
        # Ensure we do NOT fabricate a false merchant column or customer ID
        self.assertNotIn("customer_id", norm_df.columns)
        self.assertNotIn("user_id", norm_df.columns)
        if "merchant" in norm_df.columns:
            # If present only via canonical mapping, it should be empty/null, never fabricated
            self.assertTrue(norm_df["merchant"].isna().all())

    def test_validate_transactions_quality(self):
        norm_df, _ = normalize_transactions(self.raw_df)
        metrics = validate_transactions(norm_df)

        self.assertEqual(metrics["total_rows"], 14)
        # Duplicate ID test: SYN001 appears twice
        self.assertEqual(metrics["duplicate_ids_count"], 1)
        self.assertIn("SYN001", metrics["duplicate_id_samples"])

        # Invalid timestamp test: SYN011 has 'invalid-date-stamp'
        self.assertEqual(metrics["invalid_timestamps_count"], 1)

        # Invalid amount checks: negative (-50 in SYN012) and zero (0 in SYN013)
        self.assertEqual(metrics["invalid_amounts"]["negative_count"], 1)
        self.assertEqual(metrics["invalid_amounts"]["zero_count"], 1)

        # Status distribution check
        self.assertEqual(metrics["status_distribution"].get("SUCCESS"), 13)
        self.assertEqual(metrics["status_distribution"].get("FAILED"), 1)

        # Warnings check
        self.assertTrue(len(metrics["warnings"]) > 0)
        self.assertTrue(any("duplicate" in w.lower() for w in metrics["warnings"]))
        self.assertTrue(any("invalid or unparseable timestamps" in w.lower() for w in metrics["warnings"]))

    def test_generate_quality_report(self):
        norm_df, _ = normalize_transactions(self.raw_df)
        report = generate_quality_report(norm_df, dataset_name="synthetic_transactions.csv")
        self.assertIn("SPENDWISE AI - DATASET QUALITY AUDIT REPORT", report)
        self.assertIn("Total Records (Rows)  : 14", report)
        self.assertIn("SYN001", report)

    def test_resolve_data_path_and_agentx_data_dir(self):
        # Test default resolution
        path = resolve_data_path("transactions.csv")
        self.assertTrue(os.path.exists(path))

        # Test environment variable override
        test_dir = os.path.join(os.path.dirname(__file__), "data")
        with patch.dict(os.environ, {"AGENTX_DATA_DIR": test_dir}):
            synthetic_path = resolve_data_path("synthetic_transactions.csv")
            self.assertEqual(synthetic_path, os.path.join(test_dir, "synthetic_transactions.csv"))

        # Test missing file error
        with self.assertRaises(FileNotFoundError):
            resolve_data_path("non_existent_file_12345.csv")

    def test_transaction_dataset_wrapper(self):
        dataset = TransactionDataset(
            filename="synthetic_transactions.csv",
            data_dir=os.path.join(os.path.dirname(__file__), "data"),
        )
        self.assertEqual(len(dataset), 14)
        self.assertIn("total_rows", dataset.metrics)
        report = dataset.quality_report()
        self.assertIn("DATASET QUALITY AUDIT REPORT", report)


if __name__ == "__main__":
    unittest.main()
