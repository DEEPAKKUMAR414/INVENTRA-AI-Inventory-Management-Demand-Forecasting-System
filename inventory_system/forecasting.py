"""Global Random Forest training, persistence, and recursive demand forecasts."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from inventory_system.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, chronological_split


NUMERIC_FEATURES = tuple(column for column in FEATURE_COLUMNS if column != "product_id")
MODEL_FILENAME = "random_forest_demand.joblib"


def build_model(random_state: int = 42, n_estimators: int = 200) -> Pipeline:
    """Create the global preprocessing and RandomForestRegressor pipeline."""
    if n_estimators < 100:
        raise ValueError("n_estimators must be at least 100.")
    feature_transformer = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", list(NUMERIC_FEATURES)),
            ("product", OneHotEncoder(handle_unknown="ignore"), ["product_id"]),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("features", feature_transformer),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=-1,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )


def train_model(
    features: pd.DataFrame,
    model_path: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
    """Fit on each product's earlier dates and save the fitted pipeline."""
    train, test = chronological_split(features, test_size=test_size)
    model = build_model(random_state=random_state, n_estimators=n_estimators)
    model.fit(train.loc[:, FEATURE_COLUMNS], train[TARGET_COLUMN])

    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(model, destination)
    except OSError as exc:
        raise RuntimeError(f"Could not save trained model to {destination}") from exc
    return model, train, test


def load_model(model_path: str | Path) -> Pipeline:
    """Load a persisted scikit-learn pipeline, reporting missing/corrupt artifacts."""
    source = Path(model_path)
    if not source.is_file():
        raise FileNotFoundError(f"Trained model not found: {source}")
    try:
        model = joblib.load(source)
    except (OSError, ValueError, EOFError) as exc:
        raise RuntimeError(f"Could not load trained model from {source}") from exc
    if not isinstance(model, Pipeline):
        raise TypeError(f"Model artifact at {source} is not a scikit-learn Pipeline.")
    return model


def forecast_demand(
    model: Pipeline,
    historical_data: pd.DataFrame,
    products: pd.DataFrame,
    horizon: int,
    product_id: int | None = None,
) -> pd.DataFrame:
    """Recursively forecast future daily units for one product or all products."""
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer.")
    required_history = {"product_id", "sale_date", "units_sold"}
    if required_history.difference(historical_data.columns):
        raise ValueError(
            f"Historical data is missing columns: "
            f"{sorted(required_history.difference(historical_data.columns))}"
        )
    if {"product_id", "product_name"}.difference(products.columns):
        raise ValueError("Products must contain product_id and product_name columns.")
    if products["product_id"].duplicated().any() or products["product_name"].isna().any():
        raise ValueError("Products must have unique IDs and non-missing names.")

    selected_products = products.copy()
    if product_id is not None:
        selected_products = selected_products.loc[
            selected_products["product_id"] == product_id
        ]
        if selected_products.empty:
            raise ValueError(f"Unknown product_id: {product_id}")

    data = historical_data.copy()
    data["sale_date"] = pd.to_datetime(data["sale_date"], errors="coerce")
    if data[["product_id", "sale_date", "units_sold"]].isna().any().any():
        raise ValueError("Historical data has missing product IDs, dates, or sales.")
    if (data["units_sold"] < 0).any():
        raise ValueError("Historical sales must be non-negative.")

    product_states: list[tuple[int, str, pd.Timestamp, list[float]]] = []
    product_records: list[list[dict[str, int | str | pd.Timestamp | float]]] = []
    for product in selected_products.itertuples(index=False):
        product_history = data.loc[data["product_id"] == product.product_id].sort_values(
            "sale_date"
        )
        if len(product_history) < 14:
            raise ValueError(
                f"Product {product.product_id} needs at least 14 historical days."
            )
        observed_dates = product_history["sale_date"].diff().dropna()
        if not observed_dates.eq(pd.Timedelta(days=1)).all():
            raise ValueError(
                f"Product {product.product_id} history must have consecutive daily dates."
            )

        values = product_history["units_sold"].astype(float).tolist()
        last_date = product_history["sale_date"].iloc[-1]
        product_states.append(
            (int(product.product_id), str(product.product_name), last_date, values)
        )
        product_records.append([])

    for step in range(1, horizon + 1):
        feature_rows: list[dict[str, int | float]] = []
        forecast_dates: list[pd.Timestamp] = []
        for product_id, _product_name, last_date, values in product_states:
            forecast_date = last_date + pd.Timedelta(days=step)
            feature_rows.append(
                {
                    "lag_1": values[-1],
                    "lag_7": values[-7],
                    "lag_14": values[-14],
                    "rolling_mean_7": float(np.mean(values[-7:])),
                    "rolling_mean_14": float(np.mean(values[-14:])),
                    "day_of_week": forecast_date.dayofweek,
                    "month": forecast_date.month,
                    "product_id": product_id,
                }
            )
            forecast_dates.append(forecast_date)

        predictions = model.predict(
            pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
        )
        for state, forecast_date, raw_prediction, records in zip(
            product_states, forecast_dates, predictions, product_records
        ):
            product_id, product_name, _last_date, values = state
            prediction = max(0.0, float(raw_prediction))
            values.append(prediction)
            records.append(
                {
                    "product_id": product_id,
                    "product_name": product_name,
                    "forecast_date": forecast_date,
                    "predicted_units": prediction,
                }
            )

    records = [record for product_forecast in product_records for record in product_forecast]
    result = pd.DataFrame(
        records, columns=["product_id", "product_name", "forecast_date", "predicted_units"]
    )
    result.attrs["total_predicted_demand"] = float(result["predicted_units"].sum())
    return result


def total_predicted_demand(forecast: pd.DataFrame) -> float:
    """Return total predicted demand represented by a forecast DataFrame."""
    if "predicted_units" not in forecast.columns:
        raise ValueError("Forecast data must contain predicted_units.")
    return float(forecast["predicted_units"].sum())
