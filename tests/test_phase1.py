"""Tests for Phase 1 synthetic data generation and SQLite operations."""

from pathlib import Path
import sqlite3
import tempfile
import unittest

import pandas as pd

from inventory_system.data_generation import generate_synthetic_data, validate_sales_data
from inventory_system.database import (
    fetch_sales,
    fetch_table_columns,
    fetch_table_counts,
    initialize_database,
)


class SyntheticDataTests(unittest.TestCase):
    def test_dataset_dimensions_and_quality(self) -> None:
        products, inventory, sales = generate_synthetic_data()

        self.assertEqual(len(products), 10)
        self.assertEqual(len(inventory), 10)
        self.assertEqual(len(sales), 1800)
        self.assertFalse(sales["product_id"].isna().any())
        self.assertTrue((sales["units_sold"] >= 0).all())
        self.assertEqual(pd.to_datetime(sales["sale_date"], errors="coerce").notna().all(), True)
        self.assertEqual(sales["product_id"].nunique(), 10)
        self.assertEqual(sales["sale_date"].nunique(), 180)
        self.assertFalse(sales.duplicated(["product_id", "sale_date"]).any())

    def test_generation_is_reproducible(self) -> None:
        first = generate_synthetic_data(seed=17)
        second = generate_synthetic_data(seed=17)
        for first_frame, second_frame in zip(first, second):
            pd.testing.assert_frame_equal(first_frame, second_frame)

    def test_products_have_different_weekly_patterns(self) -> None:
        _, _, sales = generate_synthetic_data()
        sales = sales.assign(weekday=pd.to_datetime(sales["sale_date"]).dt.dayofweek)
        daily_means = sales.groupby(["product_id", "weekday"])["units_sold"].mean()
        weekday_variation = daily_means.groupby(level="product_id").std()
        self.assertGreater(float(weekday_variation.mean()), 0)
        product_means = sales.groupby("product_id")["units_sold"].mean()
        self.assertGreater(float(product_means.max() - product_means.min()), 5)

    def test_sales_validation_rejects_negative_sales(self) -> None:
        invalid_sales = pd.DataFrame(
            [{"product_id": 1, "sale_date": "2025-01-01", "units_sold": -1}]
        )
        with self.assertRaises(ValueError):
            validate_sales_data(invalid_sales)


class DatabaseTests(unittest.TestCase):
    def test_database_round_trip_and_schema(self) -> None:
        products, inventory, sales = generate_synthetic_data()

        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "nested" / "inventory.db"
            initialize_database(database_path, products, inventory, sales)

            self.assertEqual(
                fetch_table_counts(database_path),
                {"products": 10, "inventory": 10, "sales": 1800},
            )
            self.assertEqual(len(fetch_sales(database_path)), 1800)
            schema = fetch_table_columns(database_path)
            self.assertEqual(
                [column["name"] for column in schema["inventory"]],
                [
                    "product_id",
                    "stock_quantity",
                    "unit_cost",
                    "selling_price",
                    "supplier_lead_time_days",
                    "updated_at",
                ],
            )

            initialize_database(database_path, products, inventory, sales)
            self.assertEqual(fetch_table_counts(database_path)["sales"], 1800)

    def test_database_constraints_reject_orphan_sales(self) -> None:
        products, inventory, sales = generate_synthetic_data()
        invalid_sales = sales.copy()
        invalid_sales.loc[0, "product_id"] = 999

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                initialize_database(
                    Path(temp_dir) / "inventory.db", products, inventory, invalid_sales
                )

    def test_sqlite_rejects_negative_stock(self) -> None:
        products, inventory, sales = generate_synthetic_data()

        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "inventory.db"
            initialize_database(database_path, products, inventory, sales)
            connection = sqlite3.connect(database_path)
            try:
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(
                        "UPDATE inventory SET stock_quantity = -1 WHERE product_id = 1"
                    )
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
