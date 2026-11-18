"""Generate recursive future demand forecasts from the trained model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inventory_system.database import fetch_products
from inventory_system.forecasting import (
    MODEL_FILENAME,
    forecast_demand,
    load_model,
    total_predicted_demand,
)
from inventory_system.preprocessing import load_historical_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, choices=(7, 30), default=7)
    parser.add_argument("--product-id", type=int, default=None)
    parser.add_argument(
        "--sales", type=Path, default=PROJECT_ROOT / "data" / "synthetic_sales.csv"
    )
    parser.add_argument(
        "--database", type=Path, default=PROJECT_ROOT / "database" / "inventory.db"
    )
    parser.add_argument(
        "--model", type=Path, default=PROJECT_ROOT / "models" / MODEL_FILENAME
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    model = load_model(args.model)
    historical = load_historical_data(args.sales, args.database)
    products = fetch_products(args.database)
    forecast = forecast_demand(
        model,
        historical,
        products,
        horizon=args.days,
        product_id=args.product_id,
    )

    output_path = args.output or (
        PROJECT_ROOT / "data" / f"forecast_{args.days}_day.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(output_path, index=False)
    print(forecast.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"\nTotal predicted demand ({args.days} days): {total_predicted_demand(forecast):.2f}")
    print(f"Saved forecast: {output_path.resolve()}")


if __name__ == "__main__":
    main()
