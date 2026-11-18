"""SQLite schema creation, seeding, and query helpers."""

from __future__ import annotations

import sqlite3
import math
from pathlib import Path
from typing import Any

import pandas as pd

from inventory_system.data_generation import (
    INVENTORY_COLUMNS,
    PRODUCT_COLUMNS,
    SALES_COLUMNS,
    validate_sales_data,
)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY,
    sku TEXT NOT NULL UNIQUE,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS inventory (
    product_id INTEGER PRIMARY KEY,
    stock_quantity INTEGER NOT NULL CHECK (stock_quantity >= 0),
    unit_cost REAL NOT NULL CHECK (unit_cost >= 0),
    selling_price REAL NOT NULL CHECK (selling_price >= 0),
    supplier_lead_time_days INTEGER NOT NULL CHECK (supplier_lead_time_days > 0),
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE TABLE IF NOT EXISTS sales (
    sale_id INTEGER PRIMARY KEY,
    product_id INTEGER NOT NULL,
    sale_date TEXT NOT NULL,
    units_sold INTEGER NOT NULL CHECK (units_sold >= 0),
    UNIQUE (product_id, sale_date),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE INDEX IF NOT EXISTS idx_sales_sale_date ON sales(sale_date);

CREATE TABLE IF NOT EXISTS inventory_changes (
    change_id INTEGER PRIMARY KEY,
    product_id INTEGER NOT NULL,
    previous_stock INTEGER NOT NULL CHECK (previous_stock >= 0),
    new_stock INTEGER NOT NULL CHECK (new_stock >= 0),
    quantity_change INTEGER NOT NULL,
    change_type TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE TABLE IF NOT EXISTS stock_purchases (
    purchase_id INTEGER PRIMARY KEY,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    unit_cost REAL NOT NULL CHECK (unit_cost >= 0),
    supplier TEXT,
    change_id INTEGER NOT NULL,
    purchased_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(product_id),
    FOREIGN KEY (change_id) REFERENCES inventory_changes(change_id)
);
"""


def _validate_columns(frame: pd.DataFrame, required: tuple[str, ...], label: str) -> None:
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"{label} data is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{label} data must contain at least one row.")


def _validate_seed_data(
    products: pd.DataFrame, inventory: pd.DataFrame, sales: pd.DataFrame
) -> None:
    _validate_columns(products, PRODUCT_COLUMNS, "Products")
    _validate_columns(inventory, INVENTORY_COLUMNS, "Inventory")
    validate_sales_data(sales)

    for label, frame in (("Products", products), ("Inventory", inventory)):
        if frame["product_id"].isna().any():
            raise ValueError(f"{label} data contains a missing product_id.")
        if frame["product_id"].duplicated().any():
            raise ValueError(f"{label} data contains duplicate product_id values.")

    product_ids = set(products["product_id"])
    if set(inventory["product_id"]) != product_ids:
        raise ValueError("Inventory product IDs must match the products table.")
    if not set(sales["product_id"]).issubset(product_ids):
        raise ValueError("Sales contains a product_id not present in products.")
    if (inventory["stock_quantity"] < 0).any():
        raise ValueError("Inventory stock_quantity values must be non-negative.")
    if (inventory["unit_cost"] < 0).any() or (inventory["selling_price"] < 0).any():
        raise ValueError("Inventory costs and prices must be non-negative.")
    if (inventory["supplier_lead_time_days"] <= 0).any():
        raise ValueError("Supplier lead times must be positive.")


def initialize_database(
    database_path: str | Path,
    products: pd.DataFrame,
    inventory: pd.DataFrame,
    sales: pd.DataFrame,
) -> Path:
    """Create the schema and insert validated product, inventory, and sales rows."""
    _validate_seed_data(products, inventory, sales)
    destination = Path(database_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(destination)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(_SCHEMA)
        with connection:
            connection.executemany(
                """
                INSERT INTO products (product_id, sku, product_name, category)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(product_id) DO UPDATE SET
                    sku = excluded.sku,
                    product_name = excluded.product_name,
                    category = excluded.category
                """,
                products.loc[:, PRODUCT_COLUMNS].itertuples(index=False, name=None),
            )
            connection.executemany(
                """
                INSERT INTO inventory (
                    product_id, stock_quantity, unit_cost, selling_price,
                    supplier_lead_time_days
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(product_id) DO UPDATE SET
                    stock_quantity = excluded.stock_quantity,
                    unit_cost = excluded.unit_cost,
                    selling_price = excluded.selling_price,
                    supplier_lead_time_days = excluded.supplier_lead_time_days,
                    updated_at = CURRENT_TIMESTAMP
                """,
                inventory.loc[:, INVENTORY_COLUMNS].itertuples(index=False, name=None),
            )
            connection.executemany(
                """
                INSERT INTO sales (product_id, sale_date, units_sold)
                VALUES (?, ?, ?)
                ON CONFLICT(product_id, sale_date) DO UPDATE SET
                    units_sold = excluded.units_sold
                """,
                sales.loc[:, SALES_COLUMNS].itertuples(index=False, name=None),
            )
    except sqlite3.Error as exc:
        raise RuntimeError(f"Failed to initialize SQLite database at {destination}") from exc
    finally:
        if connection is not None:
            connection.close()

    return destination


def refresh_sales(
    database_path: str | Path, products: pd.DataFrame, sales: pd.DataFrame
) -> Path:
    """Replace historical sales while preserving existing product and stock rows."""
    _validate_columns(products, PRODUCT_COLUMNS, "Products")
    validate_sales_data(sales)
    if products["product_id"].isna().any() or products["product_id"].duplicated().any():
        raise ValueError("Products must have unique, non-missing product IDs.")
    if not set(sales["product_id"]).issubset(set(products["product_id"])):
        raise ValueError("Sales contains a product_id not present in products.")

    destination = Path(database_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(destination)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(_SCHEMA)
        with connection:
            connection.executemany(
                """
                INSERT INTO products (product_id, sku, product_name, category)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(product_id) DO NOTHING
                """,
                products.loc[:, PRODUCT_COLUMNS].itertuples(index=False, name=None),
            )
            known_ids = {
                row[0]
                for row in connection.execute("SELECT product_id FROM products").fetchall()
            }
            if not set(sales["product_id"]).issubset(known_ids):
                raise ValueError("Sales contains product IDs absent from SQLite products.")
            connection.execute("DELETE FROM sales")
            connection.executemany(
                "INSERT INTO sales (product_id, sale_date, units_sold) VALUES (?, ?, ?)",
                sales.loc[:, SALES_COLUMNS].itertuples(index=False, name=None),
            )
    except sqlite3.Error as exc:
        raise RuntimeError(f"Failed to refresh sales in SQLite database at {destination}") from exc
    finally:
        if connection is not None:
            connection.close()
    return destination


def _validate_integer(value: int, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer greater than or equal to {minimum}.")
    return value


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(_SCHEMA)


def _apply_inventory_change(
    connection: sqlite3.Connection,
    product_id: int,
    new_stock: int,
    change_type: str,
    reason: str,
) -> int:
    row = connection.execute(
        "SELECT stock_quantity FROM inventory WHERE product_id = ?", (product_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"No inventory record exists for product_id {product_id}.")
    previous_stock = int(row[0])
    connection.execute(
        """
        UPDATE inventory
        SET stock_quantity = ?, updated_at = CURRENT_TIMESTAMP
        WHERE product_id = ?
        """,
        (new_stock, product_id),
    )
    cursor = connection.execute(
        """
        INSERT INTO inventory_changes (
            product_id, previous_stock, new_stock, quantity_change, change_type, reason
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            product_id,
            previous_stock,
            new_stock,
            new_stock - previous_stock,
            change_type,
            reason,
        ),
    )
    return int(cursor.lastrowid)


def update_inventory_stock(
    database_path: str | Path,
    product_id: int,
    stock_quantity: int,
    reason: str = "Manual inventory update",
) -> None:
    """Set stock to an absolute quantity and record the change atomically."""
    _validate_integer(product_id, "product_id", minimum=1)
    _validate_integer(stock_quantity, "stock_quantity")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string.")
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(database_path)
        _ensure_schema(connection)
        with connection:
            _apply_inventory_change(
                connection, product_id, stock_quantity, "adjustment", reason.strip()
            )
    except sqlite3.Error as exc:
        raise RuntimeError(f"Failed to update inventory for product_id {product_id}") from exc
    finally:
        if connection is not None:
            connection.close()


def record_inventory_change(
    database_path: str | Path,
    product_id: int,
    quantity_change: int,
    reason: str,
    change_type: str = "adjustment",
) -> None:
    """Apply and audit a signed stock change in a single database transaction."""
    _validate_integer(product_id, "product_id", minimum=1)
    if isinstance(quantity_change, bool) or not isinstance(quantity_change, int):
        raise ValueError("quantity_change must be an integer.")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string.")
    if not isinstance(change_type, str) or not change_type.strip():
        raise ValueError("change_type must be a non-empty string.")
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(database_path)
        _ensure_schema(connection)
        with connection:
            row = connection.execute(
                "SELECT stock_quantity FROM inventory WHERE product_id = ?", (product_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"No inventory record exists for product_id {product_id}.")
            new_stock = int(row[0]) + quantity_change
            if new_stock < 0:
                raise ValueError("Inventory stock cannot become negative.")
            _apply_inventory_change(
                connection,
                product_id,
                new_stock,
                change_type.strip(),
                reason.strip(),
            )
    except sqlite3.Error as exc:
        raise RuntimeError(f"Failed to record inventory change for product_id {product_id}") from exc
    finally:
        if connection is not None:
            connection.close()


def record_stock_purchase(
    database_path: str | Path,
    product_id: int,
    quantity: int,
    unit_cost: float,
    supplier: str | None = None,
    reason: str = "Stock purchase received",
) -> int:
    """Record a received purchase and stock increase atomically; return its ID."""
    _validate_integer(product_id, "product_id", minimum=1)
    _validate_integer(quantity, "quantity", minimum=1)
    if isinstance(unit_cost, bool) or not isinstance(unit_cost, (int, float)):
        raise ValueError("unit_cost must be a non-negative finite number.")
    if not math.isfinite(unit_cost) or unit_cost < 0:
        raise ValueError("unit_cost must be a non-negative finite number.")
    if supplier is not None and (not isinstance(supplier, str) or not supplier.strip()):
        raise ValueError("supplier must be None or a non-empty string.")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string.")
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(database_path)
        _ensure_schema(connection)
        with connection:
            row = connection.execute(
                """
                SELECT stock_quantity FROM inventory WHERE product_id = ?
                """,
                (product_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"No inventory record exists for product_id {product_id}.")
            new_stock = int(row[0]) + quantity
            change_id = _apply_inventory_change(
                connection,
                product_id,
                new_stock,
                "purchase",
                reason.strip(),
            )
            cursor = connection.execute(
                """
                INSERT INTO stock_purchases (
                    product_id, quantity, unit_cost, supplier, change_id
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    product_id,
                    quantity,
                    float(unit_cost),
                    supplier.strip() if supplier is not None else None,
                    change_id,
                ),
            )
            purchase_id = int(cursor.lastrowid)
    except sqlite3.Error as exc:
        raise RuntimeError(f"Failed to record stock purchase for product_id {product_id}") from exc
    finally:
        if connection is not None:
            connection.close()
    return purchase_id


def fetch_inventory(database_path: str | Path) -> pd.DataFrame:
    """Load current inventory, including supplier lead time and unit cost."""
    try:
        with sqlite3.connect(database_path) as connection:
            return pd.read_sql_query(
                """
                SELECT product_id, stock_quantity, unit_cost, selling_price,
                       supplier_lead_time_days, updated_at
                FROM inventory
                ORDER BY product_id
                """,
                connection,
            )
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        raise RuntimeError(f"Failed to read inventory from database at {database_path}") from exc


def fetch_inventory_history(
    database_path: str | Path, product_id: int | None = None
) -> pd.DataFrame:
    """Retrieve audited inventory changes, including linked purchase details."""
    if product_id is not None:
        _validate_integer(product_id, "product_id", minimum=1)
    query = """
        SELECT changes.change_id, changes.product_id, changes.previous_stock,
               changes.new_stock, changes.quantity_change, changes.change_type,
               changes.reason, changes.created_at, purchases.purchase_id,
               purchases.quantity AS purchased_quantity, purchases.unit_cost,
               purchases.supplier, purchases.purchased_at
        FROM inventory_changes AS changes
        LEFT JOIN stock_purchases AS purchases ON purchases.change_id = changes.change_id
    """
    parameters: tuple[int, ...] = ()
    if product_id is not None:
        query += " WHERE changes.product_id = ?"
        parameters = (product_id,)
    query += " ORDER BY changes.created_at, changes.change_id"
    try:
        connection = sqlite3.connect(database_path)
        _ensure_schema(connection)
        try:
            return pd.read_sql_query(query, connection, params=parameters)
        finally:
            connection.close()
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        raise RuntimeError(f"Failed to retrieve inventory history from {database_path}") from exc


def fetch_table_counts(database_path: str | Path) -> dict[str, int]:
    """Return row counts for the Phase 1 tables."""
    counts: dict[str, int] = {}
    try:
        with sqlite3.connect(database_path) as connection:
            for table in ("products", "inventory", "sales"):
                row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                if row is None:
                    raise RuntimeError(f"Could not count rows in {table}.")
                counts[table] = int(row[0])
    except sqlite3.Error as exc:
        raise RuntimeError(f"Failed to read database at {database_path}") from exc
    return counts


def fetch_table_columns(database_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Return SQLite column metadata for the Phase 1 tables."""
    schema: dict[str, list[dict[str, Any]]] = {}
    try:
        with sqlite3.connect(database_path) as connection:
            for table in ("products", "inventory", "sales"):
                rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
                schema[table] = [
                    {"name": row[1], "type": row[2], "not_null": bool(row[3]), "primary_key": bool(row[5])}
                    for row in rows
                ]
    except sqlite3.Error as exc:
        raise RuntimeError(f"Failed to inspect database at {database_path}") from exc
    return schema


def fetch_sales(database_path: str | Path) -> pd.DataFrame:
    """Load historical sales ordered by date and product ID."""
    try:
        with sqlite3.connect(database_path) as connection:
            return pd.read_sql_query(
                """
                SELECT product_id, sale_date, units_sold
                FROM sales
                ORDER BY sale_date, product_id
                """,
                connection,
            )
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        raise RuntimeError(f"Failed to read sales from database at {database_path}") from exc


def fetch_products(database_path: str | Path) -> pd.DataFrame:
    """Load product IDs and descriptive information from SQLite."""
    try:
        with sqlite3.connect(database_path) as connection:
            return pd.read_sql_query(
                """
                SELECT product_id, sku, product_name, category
                FROM products
                ORDER BY product_id
                """,
                connection,
            )
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        raise RuntimeError(f"Failed to read products from database at {database_path}") from exc
