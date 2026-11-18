"""Train, evaluate, and persist the global Random Forest demand model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inventory_system.evaluation import EVALUATION_NOTE, evaluate_models
from inventory_system.forecasting import MODEL_FILENAME, train_model
from inventory_system.preprocessing import create_features, load_historical_data


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
    args = parser.parse_args()

    historical = load_historical_data(args.sales, args.database)
    features = create_features(historical)
    model, train_records, test_records = train_model(features, args.model)
    comparison, performance, _product_performance = evaluate_models(model, test_records)

    print(f"Training records: {len(train_records)}")
    print(f"Chronological test records: {len(test_records)}")
    print(f"Saved model: {args.model.resolve()}")
    print("\nPerformance on the same chronological test records:")
    print(performance.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\n{EVALUATION_NOTE}")
    print(f"Actual-vs-predicted rows: {len(comparison)}")


if __name__ == "__main__":
    main()
