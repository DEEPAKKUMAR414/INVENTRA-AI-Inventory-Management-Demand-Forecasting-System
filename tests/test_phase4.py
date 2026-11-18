"""Smoke tests for the Streamlit frontend data adapters and exports."""

from pathlib import Path
from io import BytesIO
import unittest
from xml.etree import ElementTree

import pandas as pd
from streamlit.testing.v1 import AppTest

import app
from inventory_system.forecasting import forecast_demand
from inventory_system.restocking import generate_restock_recommendations


class Phase4ApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.products, cls.inventory, cls.historical = app.load_dashboard_data()

    def test_application_imports_and_has_navigation_sections(self) -> None:
        self.assertEqual(len(app.NAVIGATION), 7)
        self.assertIn("Overview Dashboard", app.NAVIGATION)
        self.assertIn("Reports and Downloads", app.NAVIGATION)
        self.assertTrue(callable(app.render_app))

    def test_dashboard_brand_and_local_illustration_render(self) -> None:
        self.assertTrue(app.BRAND_NAME)
        self.assertTrue(app.HERO_IMAGE_PATH.is_file())
        ElementTree.parse(app.HERO_IMAGE_PATH)

        rendered = AppTest.from_file(str(Path(app.__file__)), default_timeout=120).run()
        self.assertFalse(rendered.exception)
        markdown = "\n".join(element.value for element in rendered.get("markdown"))
        self.assertIn(app.BRAND_NAME, markdown)
        self.assertIn("AI SYSTEM ONLINE", markdown)

    def test_dashboard_and_inventory_data_load_from_sqlite(self) -> None:
        self.assertEqual(len(self.products), 10)
        self.assertEqual(len(self.inventory), 10)
        self.assertFalse(self.historical.empty)
        self.assertEqual(
            int(self.historical["units_sold"].sum()),
            int(app.fetch_sales(app.DATABASE_PATH)["units_sold"].sum()),
        )

    def test_forecast_and_restock_data_load_using_existing_model(self) -> None:
        model = app.load_model_for_app()
        forecast = forecast_demand(
            model, self.historical, self.products, horizon=7, product_id=1
        )
        self.assertEqual(len(forecast), 7)
        recommendations = generate_restock_recommendations(
            model, self.historical, self.products, self.inventory
        )
        self.assertEqual(len(recommendations), len(self.inventory))
        self.assertTrue(
            set(recommendations["stock_status"]).issubset(
                {"CRITICAL", "LOW STOCK", "HEALTHY"}
            )
        )

    def test_cached_recommendations_refresh_after_stock_changes(self) -> None:
        app._cached_recommendations.clear()
        try:
            model = app.load_model_for_app()
            before = app.live_recommendations(
                model, self.historical, self.products, self.inventory
            )
            updated_inventory = self.inventory.copy()
            product_id = int(updated_inventory.iloc[0]["product_id"])
            updated_inventory.loc[
                updated_inventory["product_id"] == product_id, "stock_quantity"
            ] += 1

            after = app.live_recommendations(
                model, self.historical, self.products, updated_inventory
            )

            before_stock = before.loc[
                before["product_id"] == product_id, "current_stock"
            ].iloc[0]
            after_stock = after.loc[
                after["product_id"] == product_id, "current_stock"
            ].iloc[0]
            self.assertEqual(after_stock, before_stock + 1)
        finally:
            app._cached_recommendations.clear()

    def test_csv_export_serializes_current_data(self) -> None:
        payload = app._csv_bytes(self.historical)
        exported = pd.read_csv(BytesIO(payload))
        self.assertEqual(len(exported), len(self.historical))
        self.assertIn("units_sold", exported.columns)

    def test_all_download_reports_contain_live_records(self) -> None:
        model = app.load_model_for_app()
        reports, summary = app.build_download_reports(
            self.products, self.inventory, self.historical, model
        )
        report_by_filename = {filename: frame for _label, filename, frame in reports}

        inventory_csv = pd.read_csv(BytesIO(app._csv_bytes(report_by_filename["inventory_report.csv"])))
        sales_csv = pd.read_csv(BytesIO(app._csv_bytes(report_by_filename["sales_history.csv"])))
        forecast_7 = pd.read_csv(BytesIO(app._csv_bytes(report_by_filename["forecast_7_day.csv"])))
        forecast_30 = pd.read_csv(BytesIO(app._csv_bytes(report_by_filename["forecast_30_day.csv"])))
        restock_csv = pd.read_csv(
            BytesIO(app._csv_bytes(report_by_filename["restock_recommendations.csv"]))
        )

        self.assertEqual(len(inventory_csv), len(self.products))
        self.assertEqual(
            int(inventory_csv["Current Stock"].sum()),
            int(self.inventory["stock_quantity"].sum()),
        )
        self.assertEqual(len(sales_csv), len(self.historical))
        self.assertEqual(int(sales_csv["Units Sold"].sum()), int(self.historical["units_sold"].sum()))
        self.assertEqual(len(forecast_7), len(self.products) * 7)
        self.assertEqual(len(forecast_30), len(self.products) * 30)
        latest_date = self.historical["sale_date"].max()
        self.assertGreater(pd.to_datetime(forecast_7["forecast_date"]).min(), latest_date)
        self.assertGreater(pd.to_datetime(forecast_30["forecast_date"]).min(), latest_date)
        self.assertEqual(len(restock_csv), len(self.inventory))
        self.assertTrue((restock_csv["Current Stock"] >= 0).all())
        summary_values = summary.set_index("Metric")["Value"]
        self.assertEqual(summary_values["Total products"], len(self.products))
        self.assertEqual(summary_values["Total available stock"], int(self.inventory["stock_quantity"].sum()))

    def test_existing_database_operations_remain_available(self) -> None:
        self.assertTrue(callable(app.update_inventory_stock))
        self.assertTrue(callable(app.record_inventory_change))
        self.assertTrue(callable(app.record_stock_purchase))
        self.assertTrue(callable(app.fetch_inventory_history))


if __name__ == "__main__":
    unittest.main()
