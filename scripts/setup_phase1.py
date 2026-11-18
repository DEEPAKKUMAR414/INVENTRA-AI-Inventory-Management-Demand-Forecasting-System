"""Generate Phase 1 data files and initialize the SQLite database."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inventory_system.data_generation import generate_synthetic_data, write_sales_csv
from inventory_system.database import initialize_database, refresh_sales


def main() -> None:
    data_dir = PROJECT_ROOT / "data"
    database_path = PROJECT_ROOT / "database" / "inventory.db"

    products, inventory, sales = generate_synthetic_data()
    csv_path = write_sales_csv(sales, data_dir / "synthetic_sales.csv")
    if database_path.exists():
        refresh_sales(database_path, products, sales)
        print("Refreshed sales while preserving existing product and inventory records.")
    else:
        initialize_database(database_path, products, inventory, sales)

    print(f"Generated {len(products)} products and {len(inventory)} inventory records.")
    print(f"Generated {len(sales)} daily sales records at {csv_path}.")
    print(f"SQLite database is ready at {database_path}.")


if __name__ == "__main__":
    main()
