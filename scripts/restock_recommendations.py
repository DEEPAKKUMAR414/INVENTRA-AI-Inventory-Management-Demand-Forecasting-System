"""Generate and save data-driven inventory restocking recommendations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inventory_system.database import fetch_inventory, fetch_products
from inventory_system.forecasting import MODEL_FILENAME, load_model
from inventory_system.preprocessing import load_historical_data
from inventory_system.restocking import generate_restock_recommendations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sales", type=Path, default=PROJECT_ROOT / "data" / "synthetic_sales.csv"
    )
    parser.add_argument(
        "--database", type=Path, default=PROJECT_ROOT / "database" / "inventory.db"
    )
    parser.add_argument(
        "--model", type=Path, default=PROJECT_ROOT / "models" / MODEL_FILENAME
    )
    parser.add_argument("--z-score", type=float, default=1.65)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "restock_recommendations.csv",
    )
    args = parser.parse_args()

    model = load_model(args.model)
    historical = load_historical_data(args.sales, args.database)
    products = fetch_products(args.database)
    inventory = fetch_inventory(args.database)
    recommendations = generate_restock_recommendations(
        model, historical, products, inventory, z_score=args.z_score
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(args.output, index=False)
    print(recommendations.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print("\nStock status distribution:")
    print(recommendations["stock_status"].value_counts().to_string())
    print(f"\nSaved recommendations: {args.output.resolve()}")


if __name__ == "__main__":
    main()
