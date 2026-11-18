"""Forecast-driven supplier lead-time demand and inventory recommendations."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from inventory_system.forecasting import forecast_demand


RECOMMENDATION_COLUMNS = (
    "product_id",
    "product_name",
    "current_stock",
    "predicted_lead_time_demand",
    "average_daily_demand",
    "demand_standard_deviation",
    "safety_stock",
    "reorder_point",
    "recommended_order_quantity",
    "supplier_lead_time_days",
    "stock_status",
    "explanation",
)
_INVENTORY_COLUMNS = ("product_id", "stock_quantity", "supplier_lead_time_days")


def classify_stock_status(current_stock: float, reorder_point: float) -> str:
    """Classify stock against the calculated reorder point and 20% buffer."""
    if not math.isfinite(current_stock) or not math.isfinite(reorder_point):
        raise ValueError("Stock and reorder point must be finite numbers.")
    if current_stock < 0 or reorder_point < 0:
        raise ValueError("Stock and reorder point must be non-negative.")
    if current_stock < reorder_point:
        return "CRITICAL"
    if current_stock <= reorder_point * 1.2:
        return "LOW STOCK"
    return "HEALTHY"


def generate_restock_recommendations(
    model: Pipeline,
    historical_data: pd.DataFrame,
    products: pd.DataFrame,
    inventory: pd.DataFrame,
    z_score: float = 1.65,
) -> pd.DataFrame:
    """Calculate per-product restocking recommendations from recursive ML forecasts."""
    if isinstance(z_score, bool) or not isinstance(z_score, (int, float)):
        raise ValueError("z_score must be a positive finite number.")
    if not math.isfinite(z_score) or z_score <= 0:
        raise ValueError("z_score must be a positive finite number.")
    missing_inventory = set(_INVENTORY_COLUMNS).difference(inventory.columns)
    if missing_inventory:
        raise ValueError(
            f"Inventory is missing columns: {sorted(missing_inventory)}"
        )
    if inventory.empty:
        raise ValueError("Inventory must contain at least one product.")
    if inventory["product_id"].isna().any() or inventory["product_id"].duplicated().any():
        raise ValueError("Inventory must have unique, non-missing product IDs.")

    inventory_data = inventory.loc[:, _INVENTORY_COLUMNS].copy()
    for column in ("stock_quantity", "supplier_lead_time_days"):
        numeric_values = pd.to_numeric(inventory_data[column], errors="coerce")
        if (
            numeric_values.isna().any()
            or not np.isfinite(numeric_values).all()
            or (numeric_values % 1 != 0).any()
        ):
            raise ValueError(f"{column} values must be finite integers.")
        if column == "stock_quantity" and (numeric_values < 0).any():
            raise ValueError("stock_quantity values must be non-negative.")
        if column == "supplier_lead_time_days" and (numeric_values <= 0).any():
            raise ValueError("supplier_lead_time_days values must be positive.")
        inventory_data[column] = numeric_values.astype("int64")

    if products["product_id"].isna().any() or products["product_id"].duplicated().any():
        raise ValueError("Products must have unique, non-missing product IDs.")
    if {"product_id", "product_name"}.difference(products.columns):
        raise ValueError("Products must contain product_id and product_name columns.")
    if not set(inventory_data["product_id"]).issubset(set(products["product_id"])):
        raise ValueError("Inventory contains product IDs absent from product records.")

    for product_id, group in historical_data.groupby("product_id"):
        if group["units_sold"].isna().any() or (group["units_sold"] < 0).any():
            raise ValueError(f"Product {product_id} has invalid historical demand.")

    max_lead_time = int(inventory_data["supplier_lead_time_days"].max())
    forecasts = forecast_demand(
        model, historical_data, products, horizon=max_lead_time
    )
    recommendations: list[dict[str, int | float | str]] = []
    product_details = products.set_index("product_id")["product_name"]
    historical = historical_data.copy()
    historical["sale_date"] = pd.to_datetime(historical["sale_date"], errors="coerce")
    if historical["sale_date"].isna().any():
        raise ValueError("Historical data contains invalid sale dates.")

    for row in inventory_data.itertuples(index=False):
        product_id = int(row.product_id)
        stock = int(row.stock_quantity)
        lead_time = int(row.supplier_lead_time_days)
        product_history = historical.loc[historical["product_id"] == product_id]
        if product_history.empty:
            raise ValueError(f"No historical demand exists for product_id {product_id}.")
        product_forecast = forecasts.loc[forecasts["product_id"] == product_id].sort_values(
            "forecast_date"
        )
        if len(product_forecast) < lead_time:
            raise ValueError(
                f"Forecast covers {len(product_forecast)} days for product {product_id}; "
                f"{lead_time} days are required."
            )
        lead_forecast = product_forecast.iloc[:lead_time]
        if len(lead_forecast) != lead_time or lead_forecast["predicted_units"].isna().any():
            raise ValueError(
                f"Forecast coverage is incomplete for product_id {product_id}."
            )

        daily_demand = pd.to_numeric(product_history["units_sold"], errors="coerce")
        if daily_demand.isna().any() or not np.isfinite(daily_demand).all():
            raise ValueError(f"Product {product_id} has non-finite historical demand.")
        average_demand = float(daily_demand.mean())
        demand_std = float(daily_demand.std(ddof=1)) if len(daily_demand) > 1 else 0.0
        lead_demand = float(lead_forecast["predicted_units"].sum())
        safety_stock = float(z_score) * demand_std * math.sqrt(lead_time)
        reorder_point = average_demand * lead_time + safety_stock
        order_quantity = max(0, math.ceil(lead_demand + safety_stock - stock))
        status = classify_stock_status(stock, reorder_point)
        name = str(product_details.loc[row.product_id])

        explanation = (
            f"{status}: {lead_demand:.1f} units forecast over the {lead_time}-day lead time; "
            f"safety stock is {safety_stock:.1f} units (Z={float(z_score):.2f}). "
            f"Reorder point is {reorder_point:.1f} units versus {stock} currently on hand. "
            f"Order {order_quantity} units to cover forecast lead-time demand plus safety "
            "stock after accounting for current stock."
        )
        recommendations.append(
            {
                "product_id": product_id,
                "product_name": name,
                "current_stock": stock,
                "predicted_lead_time_demand": lead_demand,
                "average_daily_demand": average_demand,
                "demand_standard_deviation": demand_std,
                "safety_stock": safety_stock,
                "reorder_point": reorder_point,
                "recommended_order_quantity": order_quantity,
                "supplier_lead_time_days": lead_time,
                "stock_status": status,
                "explanation": explanation,
            }
        )

    return pd.DataFrame(recommendations, columns=RECOMMENDATION_COLUMNS).sort_values(
        "product_id"
    ).reset_index(drop=True)
