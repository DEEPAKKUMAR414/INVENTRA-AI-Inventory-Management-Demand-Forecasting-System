"""Tests for forecast-driven restocking and persistent inventory operations."""

from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
import pandas as pd

from inventory_system.data_generation import generate_synthetic_data
from inventory_system.database import (
    fetch_inventory,
    fetch_inventory_history,
    initialize_database,
    record_inventory_change,
    record_stock_purchase,
    refresh_sales,
    update_inventory_stock,
)
from inventory_system.forecasting import build_model, forecast_demand
from inventory_system.preprocessing import (
    FEATURE_COLUMNS,
    create_features,
    load_historical_data,
)
from inventory_system.restocking import (
    classify_stock_status,
    generate_restock_recommendations,
)


ROOT = Path(__file__).resolve().parents[1]


class RestockingCalculationTests(unittest.TestCase):
    def test_default_dataset_ends_yesterday_and_has_180_days(self) -> None:
        _products, _inventory, sales = generate_synthetic_data()
        dates = pd.to_datetime(sales["sale_date"])
        self.assertEqual(dates.nunique(), 180)
        self.assertEqual(dates.max().date(), pd.Timestamp.today().date() - pd.Timedelta(days=1))

    def test_safety_stock_reorder_point_and_purchase_quantity(self) -> None:
        products, inventory, sales = generate_synthetic_data()
        history = sales.merge(products[["product_id", "product_name"]], on="product_id")

        class FixedForecastModel:
            def predict(self, features: pd.DataFrame) -> np.ndarray:
                return np.full(len(features), 10.0)

        inventory = inventory.iloc[[0]].copy()
        inventory.loc[:, "stock_quantity"] = 20
        inventory.loc[:, "supplier_lead_time_days"] = 2
        rec = generate_restock_recommendations(
            FixedForecastModel(), history, products, inventory
        ).iloc[0]
        daily_std = history.loc[history["product_id"] == 1, "units_sold"].std(ddof=1)
        expected_safety_stock = 1.65 * daily_std * np.sqrt(2)
        self.assertAlmostEqual(rec["safety_stock"], expected_safety_stock)
        self.assertAlmostEqual(
            rec["reorder_point"], rec["average_daily_demand"] * 2 + expected_safety_stock
        )
        self.assertEqual(
            rec["recommended_order_quantity"],
            max(0, int(np.ceil(20 + expected_safety_stock - 20))),
        )

    def test_critical_low_and_healthy_classification(self) -> None:
        self.assertEqual(classify_stock_status(9, 10), "CRITICAL")
        self.assertEqual(classify_stock_status(10, 10), "LOW STOCK")
        self.assertEqual(classify_stock_status(12, 10), "LOW STOCK")
        self.assertEqual(classify_stock_status(12.01, 10), "HEALTHY")

    def test_products_use_individual_supplier_lead_times(self) -> None:
        products, inventory, sales = generate_synthetic_data()
        history = sales.merge(products[["product_id", "product_name"]], on="product_id")

        class FixedForecastModel:
            def predict(self, features: pd.DataFrame) -> np.ndarray:
                return np.full(len(features), 10.0)

        inventory = inventory.iloc[:2].copy()
        inventory.loc[:, "stock_quantity"] = 100
        recommendations = generate_restock_recommendations(
            FixedForecastModel(), history, products, inventory
        ).set_index("product_id")
        self.assertEqual(recommendations.loc[1, "supplier_lead_time_days"], 2)
        self.assertEqual(recommendations.loc[2, "supplier_lead_time_days"], 3)
        self.assertEqual(recommendations.loc[1, "predicted_lead_time_demand"], 20)
        self.assertEqual(recommendations.loc[2, "predicted_lead_time_demand"], 30)

    def test_invalid_inventory_values_are_rejected(self) -> None:
        products, inventory, sales = generate_synthetic_data()
        history = sales.merge(products[["product_id", "product_name"]], on="product_id")
        invalid = inventory.iloc[:1].copy()
        invalid.loc[:, "stock_quantity"] = -1
        with self.assertRaises(ValueError):
            generate_restock_recommendations(
                build_model(), history, products, invalid
            )

    def test_recommendations_use_actual_recursive_forecasts(self) -> None:
        products, inventory, sales = generate_synthetic_data()
        history = sales.merge(products[["product_id", "product_name"]], on="product_id")
        features = create_features(history)
        model = build_model()
        model.fit(features.loc[:, FEATURE_COLUMNS], features["units_sold"])
        recommendation = generate_restock_recommendations(
            model, history, products, inventory.iloc[:1]
        ).iloc[0]
        actual_forecast = forecast_demand(
            model,
            history,
            products,
            horizon=int(inventory.iloc[0]["supplier_lead_time_days"]),
            product_id=int(inventory.iloc[0]["product_id"]),
        )
        self.assertAlmostEqual(
            recommendation["predicted_lead_time_demand"],
            float(actual_forecast["predicted_units"].sum()),
        )
        self.assertGreaterEqual(recommendation["recommended_order_quantity"], 0)


class InventoryDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "inventory.db"
        products, inventory, sales = generate_synthetic_data()
        self.initial_stock = int(
            inventory.loc[inventory["product_id"] == 1, "stock_quantity"].iloc[0]
        )
        initialize_database(self.database_path, products, inventory, sales)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_stock_updates_are_persisted_and_audited(self) -> None:
        update_inventory_stock(self.database_path, 1, 123, "Cycle count")
        fresh_inventory = fetch_inventory(self.database_path)
        self.assertEqual(
            int(fresh_inventory.loc[fresh_inventory["product_id"] == 1, "stock_quantity"].iloc[0]),
            123,
        )
        history = fetch_inventory_history(self.database_path, product_id=1)
        self.assertEqual(len(history), 1)
        self.assertEqual(history.iloc[0]["previous_stock"], self.initial_stock)
        self.assertEqual(history.iloc[0]["new_stock"], 123)
        with sqlite3.connect(self.database_path) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT stock_quantity FROM inventory WHERE product_id = 1"
                ).fetchone()[0],
                123,
            )

    def test_change_and_purchase_are_recorded_transactionally(self) -> None:
        record_inventory_change(self.database_path, 1, -2, "Damaged stock")
        purchase_id = record_stock_purchase(
            self.database_path, 1, 10, 1.25, supplier="Example Supplier"
        )
        history = fetch_inventory_history(self.database_path, product_id=1)
        self.assertEqual(len(history), 2)
        self.assertEqual(int(history.iloc[-1]["purchase_id"]), purchase_id)
        self.assertEqual(history.iloc[-1]["purchased_quantity"], 10)
        current_stock = fetch_inventory(self.database_path).set_index("product_id")
        self.assertEqual(
            int(current_stock.loc[1, "stock_quantity"]), self.initial_stock + 8
        )

    def test_invalid_stock_updates_do_not_change_database(self) -> None:
        starting_stock = int(
            fetch_inventory(self.database_path)
            .set_index("product_id")
            .loc[1, "stock_quantity"]
        )
        with self.assertRaises(ValueError):
            update_inventory_stock(self.database_path, 1, -1)
        with self.assertRaises(ValueError):
            record_inventory_change(self.database_path, 1, -1000, "Invalid decrement")
        ending_stock = int(
            fetch_inventory(self.database_path)
            .set_index("product_id")
            .loc[1, "stock_quantity"]
        )
        self.assertEqual(ending_stock, starting_stock)

    def test_sales_refresh_preserves_existing_stock_and_history(self) -> None:
        update_inventory_stock(self.database_path, 1, 150, "Preservation test")
        products, _inventory, refreshed_sales = generate_synthetic_data()
        refreshed_sales = refreshed_sales.copy()
        refreshed_sales["sale_date"] = (
            pd.to_datetime(refreshed_sales["sale_date"]) + pd.Timedelta(days=1)
        ).dt.strftime("%Y-%m-%d")
        refresh_sales(self.database_path, products, refreshed_sales)

        current_stock = fetch_inventory(self.database_path).set_index("product_id")
        self.assertEqual(int(current_stock.loc[1, "stock_quantity"]), 150)
        self.assertEqual(len(fetch_inventory_history(self.database_path, product_id=1)), 1)
        with sqlite3.connect(self.database_path) as connection:
            dates = connection.execute(
                "SELECT COUNT(DISTINCT sale_date), MAX(sale_date) FROM sales"
            ).fetchone()
        self.assertEqual(dates, (180, (pd.to_datetime(refreshed_sales["sale_date"]).max()).strftime("%Y-%m-%d")))


if __name__ == "__main__":
    unittest.main()
