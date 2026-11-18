"""Historical data loading, validation, and leakage-safe feature engineering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from inventory_system.database import fetch_products


FEATURE_COLUMNS = (
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14",
    "day_of_week",
    "month",
    "product_id",
)
TARGET_COLUMN = "units_sold"
HISTORY_COLUMNS = ("product_id", "sale_date", "units_sold")


def load_historical_data(
    sales_csv_path: str | Path, database_path: str | Path
) -> pd.DataFrame:
    """Load daily sales from CSV, attach SQLite product details, and validate."""
    csv_path = Path(sales_csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Sales CSV does not exist: {csv_path}")

    try:
        sales = pd.read_csv(csv_path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read sales CSV at {csv_path}") from exc

    required = set(HISTORY_COLUMNS)
    missing_columns = required.difference(sales.columns)
    if missing_columns:
        raise ValueError(f"Sales CSV is missing columns: {sorted(missing_columns)}")
    if sales.empty:
        raise ValueError("Sales CSV must contain at least one record.")

    sales = sales.loc[:, HISTORY_COLUMNS].copy()
    if sales.isna().any().any():
        missing = sales.columns[sales.isna().any()].tolist()
        raise ValueError(f"Sales data contains missing values in: {missing}")

    numeric_ids = pd.to_numeric(sales["product_id"], errors="coerce")
    numeric_units = pd.to_numeric(sales["units_sold"], errors="coerce")
    if numeric_ids.isna().any() or (numeric_ids % 1 != 0).any():
        raise ValueError("product_id values must be non-missing integers.")
    if numeric_units.isna().any() or not np.isfinite(numeric_units).all():
        raise ValueError("units_sold values must be finite numbers.")
    if (numeric_units < 0).any():
        raise ValueError("units_sold values must be non-negative.")
    if (numeric_units % 1 != 0).any():
        raise ValueError("units_sold values must be whole numbers.")

    sales["product_id"] = numeric_ids.astype("int64")
    sales["units_sold"] = numeric_units.astype("int64")
    sales["sale_date"] = pd.to_datetime(sales["sale_date"], errors="coerce")
    if sales["sale_date"].isna().any():
        raise ValueError("Sales data contains invalid sale_date values.")
    if sales.duplicated(["product_id", "sale_date"]).any():
        raise ValueError("Sales data contains duplicate product/date records.")

    products = fetch_products(database_path)
    if products.empty:
        raise ValueError(f"No products were found in SQLite database: {database_path}")
    if products["product_id"].isna().any() or products["product_id"].duplicated().any():
        raise ValueError("SQLite products must have unique, non-missing product IDs.")

    merged = sales.merge(products, on="product_id", how="left", validate="many_to_one")
    if merged["product_name"].isna().any():
        unknown = sorted(merged.loc[merged["product_name"].isna(), "product_id"].unique())
        raise ValueError(f"Sales contain product IDs absent from SQLite: {unknown}")

    merged = merged.sort_values(["product_id", "sale_date"]).reset_index(drop=True)
    for product_id, group in merged.groupby("product_id", sort=False):
        date_gaps = group["sale_date"].diff().dropna()
        if not date_gaps.eq(pd.Timedelta(days=1)).all():
            raise ValueError(
                f"Product {product_id} must have one sales record per consecutive day."
            )
    return merged


def create_features(historical_data: pd.DataFrame) -> pd.DataFrame:
    """Build per-product one-step features using past observations only."""
    required = {"product_id", "sale_date", "units_sold", "product_name"}
    missing_columns = required.difference(historical_data.columns)
    if missing_columns:
        raise ValueError(f"Historical data is missing columns: {sorted(missing_columns)}")
    if historical_data.empty:
        raise ValueError("Historical data must contain at least one record.")

    data = historical_data.copy()
    data["sale_date"] = pd.to_datetime(data["sale_date"], errors="coerce")
    if data["sale_date"].isna().any() or data.isna().any().any():
        raise ValueError("Historical data contains missing values or invalid dates.")
    if (data["units_sold"] < 0).any():
        raise ValueError("Historical units_sold values must be non-negative.")
    if data.duplicated(["product_id", "sale_date"]).any():
        raise ValueError("Historical data contains duplicate product/date records.")
    data = data.sort_values(["product_id", "sale_date"]).reset_index(drop=True)

    feature_frames: list[pd.DataFrame] = []
    for _product_id, group in data.groupby("product_id", sort=False):
        group = group.copy()
        prior_sales = group["units_sold"].shift(1)
        group["lag_1"] = group["units_sold"].shift(1)
        group["lag_7"] = group["units_sold"].shift(7)
        group["lag_14"] = group["units_sold"].shift(14)
        group["rolling_mean_7"] = prior_sales.rolling(window=7, min_periods=7).mean()
        group["rolling_mean_14"] = prior_sales.rolling(window=14, min_periods=14).mean()
        group["day_of_week"] = group["sale_date"].dt.dayofweek
        group["month"] = group["sale_date"].dt.month
        feature_frames.append(group)

    features = pd.concat(feature_frames, ignore_index=True)
    features = features.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    features["product_id"] = features["product_id"].astype("int64")
    features["day_of_week"] = features["day_of_week"].astype("int64")
    features["month"] = features["month"].astype("int64")
    return features


def chronological_split(
    features: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each product's examples by time, preserving chronological order."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")
    if features.empty:
        raise ValueError("Cannot split an empty feature dataset.")
    if features["sale_date"].isna().any():
        raise ValueError("Feature data contains invalid sale dates.")

    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _product_id, group in features.sort_values(
        ["product_id", "sale_date"]
    ).groupby("product_id", sort=False):
        if len(group) < 2:
            raise ValueError("Each product needs at least two feature rows to split.")
        test_count = max(1, int(np.ceil(len(group) * test_size)))
        if test_count >= len(group):
            raise ValueError("test_size leaves no training records for a product.")
        train_parts.append(group.iloc[:-test_count])
        test_parts.append(group.iloc[-test_count:])

    train = pd.concat(train_parts).sort_values(["product_id", "sale_date"]).reset_index(drop=True)
    test = pd.concat(test_parts).sort_values(["product_id", "sale_date"]).reset_index(drop=True)
    for product_id in features["product_id"].unique():
        product_train = train.loc[train["product_id"] == product_id, "sale_date"]
        product_test = test.loc[test["product_id"] == product_id, "sale_date"]
        if product_train.max() >= product_test.min():
            raise ValueError(f"Chronological split failed for product {product_id}.")
    return train, test
