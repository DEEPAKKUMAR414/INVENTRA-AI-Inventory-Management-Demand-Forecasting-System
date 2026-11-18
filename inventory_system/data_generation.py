"""Reproducible synthetic retail products, inventory, and daily sales data."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


PRODUCTS = (
    ("SKU-1001", "Whole Milk", "Dairy", 52, 1.15, 2.49, 2),
    ("SKU-1002", "Sourdough Bread", "Bakery", 31, 1.80, 4.49, 3),
    ("SKU-1003", "Free-Range Eggs", "Dairy", 24, 2.10, 4.99, 4),
    ("SKU-1004", "Arabica Coffee", "Beverages", 18, 5.25, 10.99, 7),
    ("SKU-1005", "Basmati Rice", "Pantry", 15, 4.10, 8.49, 8),
    ("SKU-1006", "Olive Oil", "Pantry", 12, 6.40, 12.99, 10),
    ("SKU-1007", "Dark Chocolate", "Snacks", 38, 1.25, 3.49, 5),
    ("SKU-1008", "Orange Juice", "Beverages", 27, 1.55, 3.79, 3),
    ("SKU-1009", "Pasta", "Pantry", 21, 0.95, 2.29, 6),
    ("SKU-1010", "Greek Yogurt", "Dairy", 34, 0.90, 2.19, 4),
)

PRODUCT_COLUMNS = ("product_id", "sku", "product_name", "category")
INVENTORY_COLUMNS = (
    "product_id",
    "stock_quantity",
    "unit_cost",
    "selling_price",
    "supplier_lead_time_days",
)
SALES_COLUMNS = ("product_id", "sale_date", "units_sold")

_WEEKDAY_MULTIPLIERS = np.array([0.88, 0.91, 0.96, 1.02, 1.12, 1.28, 1.18])


def generate_synthetic_data(
    seed: int = 42,
    start_date: str | None = None,
    days: int = 180,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return reproducible data ending yesterday unless a start date is supplied."""
    if days <= 0:
        raise ValueError("days must be a positive integer.")
    if seed < 0:
        raise ValueError("seed must be a non-negative integer.")

    if start_date is None:
        final_date = date.today() - timedelta(days=1)
        start_date = (final_date - timedelta(days=days - 1)).isoformat()

    try:
        dates = pd.date_range(start=pd.Timestamp(start_date), periods=days, freq="D")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid start_date: {start_date!r}") from exc

    rng = np.random.default_rng(seed)
    products = pd.DataFrame(
        [
            {
                "product_id": product_id,
                "sku": sku,
                "product_name": name,
                "category": category,
            }
            for product_id, (sku, name, category, *_rest) in enumerate(PRODUCTS, start=1)
        ],
        columns=PRODUCT_COLUMNS,
    )

    inventory_records: list[dict[str, int | float]] = []
    sales_records: list[dict[str, int | str]] = []
    elapsed_days = np.arange(days)

    for product_id, product in enumerate(PRODUCTS, start=1):
        _sku, _name, _category, base_demand, unit_cost, selling_price, lead_time = product
        stock_quantity = int(rng.integers(max(12, base_demand), max(40, base_demand * 5)))
        inventory_records.append(
            {
                "product_id": product_id,
                "stock_quantity": stock_quantity,
                "unit_cost": unit_cost,
                "selling_price": selling_price,
                "supplier_lead_time_days": lead_time,
            }
        )

        weekday_effect = _WEEKDAY_MULTIPLIERS[dates.dayofweek]
        trend = 1.0 + 0.0008 * elapsed_days
        seasonal_effect = 1.0 + 0.12 * np.sin(
            2 * np.pi * elapsed_days / 45 + product_id * 0.7
        )
        expected_sales = base_demand * weekday_effect * trend * seasonal_effect
        daily_sales = rng.poisson(expected_sales)

        sales_records.extend(
            {
                "product_id": product_id,
                "sale_date": sale_date.strftime("%Y-%m-%d"),
                "units_sold": int(units_sold),
            }
            for sale_date, units_sold in zip(dates, daily_sales)
        )

    inventory = pd.DataFrame(inventory_records, columns=INVENTORY_COLUMNS)
    sales = pd.DataFrame(sales_records, columns=SALES_COLUMNS)
    validate_sales_data(sales, expected_products=len(PRODUCTS), expected_days=days)
    return products, inventory, sales


def validate_sales_data(
    sales: pd.DataFrame,
    expected_products: int | None = None,
    expected_days: int | None = None,
) -> None:
    """Raise ValueError when sales data violates the Phase 1 quality contract."""
    missing_columns = set(SALES_COLUMNS).difference(sales.columns)
    if missing_columns:
        raise ValueError(f"Sales data is missing columns: {sorted(missing_columns)}")
    if sales.empty:
        raise ValueError("Sales data must contain at least one row.")
    if sales["product_id"].isna().any():
        raise ValueError("Sales data contains a missing product_id.")
    if sales["units_sold"].isna().any() or (sales["units_sold"] < 0).any():
        raise ValueError("Sales data must have non-negative units_sold values.")

    parsed_dates = pd.to_datetime(sales["sale_date"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("Sales data contains an invalid sale_date.")
    if sales.duplicated(["product_id", "sale_date"]).any():
        raise ValueError("Sales data contains duplicate product/date records.")

    product_count = sales["product_id"].nunique()
    day_count = parsed_dates.nunique()
    if expected_products is not None and product_count != expected_products:
        raise ValueError(
            f"Expected {expected_products} products in sales, found {product_count}."
        )
    if expected_days is not None and day_count != expected_days:
        raise ValueError(f"Expected {expected_days} sales dates, found {day_count}.")
    if len(sales) != product_count * day_count:
        raise ValueError("Sales data must have one row per product for every date.")


def write_sales_csv(sales: pd.DataFrame, output_path: str | Path) -> Path:
    """Validate and write sales data as a CSV, creating its parent directory."""
    validate_sales_data(sales)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sales.to_csv(destination, index=False)
    return destination
