"""Chronological one-step-ahead model and moving-average evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from inventory_system.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN


EVALUATION_NOTE = (
    "Evaluation uses one-step-ahead predictions based only on information "
    "available before each test date. Future recursive forecasting is implemented separately."
)


def _metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def evaluate_models(
    model: Pipeline, test_records: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare RF and 7-day moving-average on identical chronological test rows."""
    if test_records.empty:
        raise ValueError("Cannot evaluate models on an empty test dataset.")
    needed = set(FEATURE_COLUMNS) | {TARGET_COLUMN, "sale_date", "product_name"}
    missing = needed.difference(test_records.columns)
    if missing:
        raise ValueError(f"Test records are missing columns: {sorted(missing)}")

    actual = test_records[TARGET_COLUMN].astype(float).reset_index(drop=True)
    rf_predictions = np.maximum(
        0.0, model.predict(test_records.loc[:, FEATURE_COLUMNS].reset_index(drop=True))
    )
    moving_average = test_records["rolling_mean_7"].astype(float).to_numpy()
    comparison = test_records.loc[
        :, ["product_id", "product_name", "sale_date", TARGET_COLUMN, "rolling_mean_7"]
    ].reset_index(drop=True)
    comparison = comparison.rename(
        columns={TARGET_COLUMN: "actual_units", "rolling_mean_7": "moving_average_prediction"}
    )
    comparison["random_forest_prediction"] = rf_predictions
    comparison["random_forest_residual"] = (
        comparison["actual_units"] - comparison["random_forest_prediction"]
    )
    comparison["moving_average_residual"] = (
        comparison["actual_units"] - comparison["moving_average_prediction"]
    )

    performance_rows: list[dict[str, float | int | str]] = []
    for model_name, predictions in (
        ("Random Forest", pd.Series(rf_predictions)),
        ("Moving Average (7-day)", pd.Series(moving_average)),
    ):
        performance_rows.append(
            {"model": model_name, **_metrics(actual, predictions)}
        )
    performance = pd.DataFrame(performance_rows, columns=["model", "mae", "rmse", "r2"])

    product_rows: list[dict[str, float | int | str]] = []
    for (product_id, product_name), group in comparison.groupby(
        ["product_id", "product_name"], sort=True
    ):
        group_actual = group["actual_units"]
        for model_name, prediction_column in (
            ("Random Forest", "random_forest_prediction"),
            ("Moving Average (7-day)", "moving_average_prediction"),
        ):
            product_rows.append(
                {
                    "product_id": int(product_id),
                    "product_name": str(product_name),
                    "model": model_name,
                    **_metrics(group_actual, group[prediction_column]),
                }
            )
    product_performance = pd.DataFrame(
        product_rows,
        columns=["product_id", "product_name", "model", "mae", "rmse", "r2"],
    )
    return comparison, performance, product_performance
