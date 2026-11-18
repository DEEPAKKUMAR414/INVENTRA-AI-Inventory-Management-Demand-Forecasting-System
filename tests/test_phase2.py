"""Tests for Phase 2 feature engineering, forecasting, and evaluation."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from inventory_system.evaluation import evaluate_models
from inventory_system.forecasting import (
    build_model,
    forecast_demand,
    load_model,
    total_predicted_demand,
    train_model,
)
from inventory_system.preprocessing import (
    FEATURE_COLUMNS,
    chronological_split,
    create_features,
    load_historical_data,
)
from inventory_system.data_generation import generate_synthetic_data


ROOT = Path(__file__).resolve().parents[1]
SALES_CSV = ROOT / "data" / "synthetic_sales.csv"
DATABASE = ROOT / "database" / "inventory.db"


class Phase2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.historical = load_historical_data(SALES_CSV, DATABASE)
        cls.products, _inventory, _sales = generate_synthetic_data()
        cls.features = create_features(cls.historical)
        cls.train, cls.test = chronological_split(cls.features)

    def test_features_use_only_prior_observations(self) -> None:
        history = self.historical.copy()
        features = create_features(history)
        product_id = 1
        target_date = features.loc[
            features["product_id"].eq(product_id), "sale_date"
        ].iloc[30]
        before = features.loc[
            (features["product_id"] == product_id)
            & (features["sale_date"] == target_date),
            ["lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_14"],
        ].reset_index(drop=True)

        history.loc[
            (history["product_id"] == product_id) & (history["sale_date"] > target_date),
            "units_sold",
        ] += 1000
        changed_features = create_features(history)
        after = changed_features.loc[
            (changed_features["product_id"] == product_id)
            & (changed_features["sale_date"] == target_date),
            ["lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_14"],
        ].reset_index(drop=True)
        pd.testing.assert_frame_equal(before, after)

    def test_split_is_chronological_for_each_product(self) -> None:
        for product_id in self.features["product_id"].unique():
            train_dates = self.train.loc[
                self.train["product_id"] == product_id, "sale_date"
            ]
            test_dates = self.test.loc[self.test["product_id"] == product_id, "sale_date"]
            self.assertLess(train_dates.max(), test_dates.min())

    def test_model_trains_and_saved_model_loads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.joblib"
            model, train, test = train_model(self.features, path)
            loaded = load_model(path)
            self.assertIsInstance(loaded, Pipeline)
            self.assertEqual(len(train), len(self.train))
            self.assertEqual(len(test), len(self.test))
            self.assertEqual(len(loaded.predict(test.loc[:, FEATURE_COLUMNS])), len(test))

    def test_random_forest_requires_at_least_100_estimators(self) -> None:
        with self.assertRaises(ValueError):
            build_model(n_estimators=99)

    def test_forecast_all_products_for_7_and_30_days(self) -> None:
        model = build_model()
        model.fit(self.train.loc[:, FEATURE_COLUMNS], self.train["units_sold"])

        for horizon, expected_rows in ((7, 70), (30, 300)):
            forecast = forecast_demand(
                model, self.historical, self.products, horizon=horizon
            )
            self.assertEqual(len(forecast), expected_rows)
            self.assertEqual(forecast.groupby("product_id").size().to_dict(), {
                product_id: horizon for product_id in self.products["product_id"]
            })
            self.assertTrue((forecast["predicted_units"] >= 0).all())
            expected_last_date = self.historical["sale_date"].max() + pd.Timedelta(
                days=horizon
            )
            self.assertEqual(forecast["forecast_date"].max(), expected_last_date)
            for _product_id, product_forecast in forecast.groupby("product_id"):
                product_history = self.historical.loc[
                    self.historical["product_id"] == _product_id, "sale_date"
                ]
                expected_dates = pd.date_range(
                    product_history.max() + pd.Timedelta(days=1),
                    periods=horizon,
                    freq="D",
                )
                self.assertEqual(product_forecast["forecast_date"].tolist(), list(expected_dates))
            self.assertAlmostEqual(
                total_predicted_demand(forecast),
                float(forecast["predicted_units"].sum()),
            )

    def test_individual_product_forecast_dates_and_names(self) -> None:
        model = build_model()
        model.fit(self.train.loc[:, FEATURE_COLUMNS], self.train["units_sold"])
        result = forecast_demand(
            model, self.historical, self.products, horizon=7, product_id=3
        )
        last_date = self.historical.loc[
            self.historical["product_id"] == 3, "sale_date"
        ].max()
        self.assertEqual(len(result), 7)
        self.assertEqual(result["product_name"].unique().tolist(), ["Free-Range Eggs"])
        self.assertEqual(
            result["forecast_date"].tolist(),
            list(pd.date_range(last_date + pd.Timedelta(days=1), periods=7, freq="D")),
        )

    def test_multi_product_forecast_batches_products_per_recursive_day(self) -> None:
        class IncrementModel:
            def __init__(self) -> None:
                self.batch_sizes: list[int] = []

            def predict(self, features: pd.DataFrame) -> np.ndarray:
                self.batch_sizes.append(len(features))
                return features["lag_1"].to_numpy() + 1

        selected_products = self.products.iloc[[1, 0]].reset_index(drop=True)
        selected_history = self.historical.loc[
            self.historical["product_id"].isin(selected_products["product_id"])
        ]
        model = IncrementModel()

        result = forecast_demand(
            model, selected_history, selected_products, horizon=3
        )

        self.assertEqual(model.batch_sizes, [2, 2, 2])
        self.assertEqual(
            result["product_id"].tolist(),
            [
                int(selected_products.iloc[0]["product_id"]),
            ] * 3
            + [int(selected_products.iloc[1]["product_id"])] * 3,
        )
        for product in selected_products.itertuples(index=False):
            last_sales = selected_history.loc[
                selected_history["product_id"] == product.product_id
            ].sort_values("sale_date")["units_sold"]
            expected = [float(last_sales.iloc[-1] + step) for step in (1, 2, 3)]
            actual = result.loc[
                result["product_id"] == product.product_id, "predicted_units"
            ].tolist()
            self.assertEqual(actual, expected)

    def test_evaluation_metrics_and_same_records_for_baseline(self) -> None:
        class FixedPredictionModel:
            def predict(self, _features: pd.DataFrame) -> np.ndarray:
                return np.full(len(_features), 10.0)

        test = self.test.copy()
        test.loc[:, "units_sold"] = 10
        test.loc[:, "rolling_mean_7"] = 10
        comparison, summary, product_summary = evaluate_models(
            FixedPredictionModel(), test  # type: ignore[arg-type]
        )
        self.assertEqual(len(comparison), len(test))
        self.assertTrue((comparison["actual_units"] == 10).all())
        self.assertTrue((summary[["mae", "rmse"]] == 0).all().all())
        self.assertTrue((summary["r2"] == 1).all())
        np.testing.assert_array_equal(
            comparison["moving_average_prediction"].to_numpy(),
            test["rolling_mean_7"].reset_index(drop=True).to_numpy(),
        )
        self.assertEqual(set(product_summary["model"]), {
            "Random Forest",
            "Moving Average (7-day)",
        })

    def test_mae_rmse_and_r2_calculated_correctly(self) -> None:
        class FixedPredictionModel:
            def predict(self, features: pd.DataFrame) -> np.ndarray:
                return np.zeros(len(features))

        test = self.test.iloc[:3].copy()
        test.loc[:, "units_sold"] = [1, 2, 3]
        test.loc[:, "rolling_mean_7"] = [1, 2, 3]
        _comparison, summary, _product_summary = evaluate_models(
            FixedPredictionModel(), test  # type: ignore[arg-type]
        )
        random_forest = summary.loc[summary["model"] == "Random Forest"].iloc[0]
        self.assertAlmostEqual(random_forest["mae"], 2.0)
        self.assertAlmostEqual(random_forest["rmse"], np.sqrt(14 / 3))
        self.assertAlmostEqual(random_forest["r2"], -6.0)


if __name__ == "__main__":
    unittest.main()
