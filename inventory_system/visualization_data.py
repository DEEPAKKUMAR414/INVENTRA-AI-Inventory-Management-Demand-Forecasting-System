"""Clean DataFrame adapters for future dashboard charts and tables."""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from inventory_system.evaluation import evaluate_models
from inventory_system.forecasting import forecast_demand


def historical_sales_data(historical: pd.DataFrame) -> pd.DataFrame:
    """Return normalized historical daily-sales columns for visualizations."""
    columns = ["product_id", "product_name", "sale_date", "units_sold"]
    missing = set(columns).difference(historical.columns)
    if missing:
        raise ValueError(f"Historical data is missing columns: {sorted(missing)}")
    result = historical.loc[:, columns].copy()
    result["sale_date"] = pd.to_datetime(result["sale_date"], errors="raise")
    return result.sort_values(["sale_date", "product_id"]).reset_index(drop=True)


def actual_vs_predicted_data(comparison: pd.DataFrame) -> pd.DataFrame:
    """Return actual, model predictions, and residuals for evaluation charts."""
    columns = [
        "product_id",
        "product_name",
        "sale_date",
        "actual_units",
        "random_forest_prediction",
        "moving_average_prediction",
        "random_forest_residual",
        "moving_average_residual",
    ]
    missing = set(columns).difference(comparison.columns)
    if missing:
        raise ValueError(f"Prediction comparison is missing columns: {sorted(missing)}")
    return comparison.loc[:, columns].copy()


def next_7_day_forecast(
    model: Pipeline,
    historical: pd.DataFrame,
    products: pd.DataFrame,
    product_id: int | None = None,
) -> pd.DataFrame:
    """Return forecast rows for the next seven days."""
    return forecast_demand(model, historical, products, horizon=7, product_id=product_id)


def next_30_day_forecast(
    model: Pipeline,
    historical: pd.DataFrame,
    products: pd.DataFrame,
    product_id: int | None = None,
) -> pd.DataFrame:
    """Return forecast rows for the next 30 days."""
    return forecast_demand(model, historical, products, horizon=30, product_id=product_id)


def product_evaluation_data(
    model: Pipeline, test_records: pd.DataFrame
) -> pd.DataFrame:
    """Return long-form per-product metric rows for evaluation charts."""
    _comparison, _performance, product_performance = evaluate_models(model, test_records)
    return product_performance
