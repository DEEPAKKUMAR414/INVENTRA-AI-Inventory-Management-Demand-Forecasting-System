"""Streamlit frontend for the AI inventory and demand forecasting project."""

from __future__ import annotations

from html import escape
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from inventory_system.database import (
    fetch_inventory,
    fetch_inventory_history,
    fetch_products,
    fetch_sales,
    record_inventory_change,
    record_stock_purchase,
    update_inventory_stock,
)
from inventory_system.evaluation import EVALUATION_NOTE, evaluate_models
from inventory_system.forecasting import (
    MODEL_FILENAME,
    forecast_demand,
    load_model,
    total_predicted_demand,
)
from inventory_system.preprocessing import (
    chronological_split,
    create_features,
    load_historical_data,
)
from inventory_system.restocking import generate_restock_recommendations


PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_PATH = PROJECT_ROOT / "database" / "inventory.db"
SALES_CSV_PATH = PROJECT_ROOT / "data" / "synthetic_sales.csv"
MODEL_PATH = PROJECT_ROOT / "models" / MODEL_FILENAME
BRAND_NAME = "INVENTRA"
HERO_IMAGE_PATH = PROJECT_ROOT / "assets" / "stockpilot_dashboard.svg"
NAVIGATION = (
    "Overview Dashboard",
    "Inventory Management",
    "Demand Forecasting",
    "Restocking Recommendations",
    "Sales Analytics",
    "Model Performance",
    "Reports and Downloads",
)
STATUS_COLORS = {"CRITICAL": "#ff657a", "LOW STOCK": "#f7b955", "HEALTHY": "#43d5a4"}
CHART_COLORS = ["#55d6e8", "#9d83ff", "#49b8ff", "#f7b955", "#43d5a4"]
KPI_ICON_PATHS = {
    "products": '<path d="m12 3 8 4.5v9L12 21l-8-4.5v-9L12 3Z"/><path d="m4.3 7.7 7.7 4.4 7.7-4.4M12 12v9"/>',
    "stock": '<path d="m4 7 8-4 8 4-8 4-8-4Z"/><path d="M4 7v10l8 4 8-4V7M12 11v10"/>',
    "sales": '<path d="M4 19V5M4 19h17"/><path d="m7 15 4-4 3 2 5-6"/>',
    "critical": '<path d="m12 3 9 17H3l9-17Z"/><path d="M12 9v5m0 3h.01"/>',
    "low_stock": '<path d="M5 18h14M7 15V9a5 5 0 0 1 10 0v6l2 3H5l2-3Z"/><path d="M10 21h4"/>',
    "healthy": '<circle cx="12" cy="12" r="9"/><path d="m8 12 2.5 2.5L16 9"/>',
    "purchase": '<path d="M3 4h2l2.2 11.2a2 2 0 0 0 2 1.6h8.5a2 2 0 0 0 2-1.6L21 8H6"/><circle cx="10" cy="20" r="1"/><circle cx="18" cy="20" r="1"/>',
}
SETUP_INSTRUCTIONS = (
    "Run the project setup, training, and report scripts from the project directory:\n\n"
    "1. `python .\\scripts\\setup_phase1.py`\n"
    "2. `python .\\scripts\\train_model.py`\n"
    "3. `python .\\scripts\\forecast_demand.py --days 7` and `--days 30`\n"
    "4. `python .\\scripts\\restock_recommendations.py`"
)


@st.cache_resource(show_spinner="Loading trained Random Forest model...")
def get_model(model_path: str, modified_time: float):
    """Cache only the immutable model, invalidating the cache when it changes."""
    del modified_time
    return load_model(model_path)


def load_dashboard_data(
    database_path: str | Path = DATABASE_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read fresh product, inventory, and sales records from SQLite."""
    products = fetch_products(database_path)
    inventory = fetch_inventory(database_path)
    sales = fetch_sales(database_path)
    if products.empty:
        raise ValueError("The database contains no products. Initialize the project data.")
    if inventory.empty:
        raise ValueError("The database contains no inventory records.")
    if sales.empty:
        raise ValueError("The database contains no sales history.")
    sales["sale_date"] = pd.to_datetime(sales["sale_date"], errors="raise")
    historical = sales.merge(
        products.loc[:, ["product_id", "product_name", "category", "sku"]],
        on="product_id",
        how="left",
        validate="many_to_one",
    )
    if historical["product_name"].isna().any():
        raise ValueError("Sales data references product IDs missing from the products table.")
    return products, inventory, historical.sort_values(
        ["sale_date", "product_id"]
    ).reset_index(drop=True)


def load_model_for_app(model_path: str | Path = MODEL_PATH):
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained model is missing: {path}")
    return get_model(str(path), path.stat().st_mtime)


def live_recommendations(
    model,
    historical: pd.DataFrame,
    products: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Recalculate recommendations using the latest supplied inventory snapshot."""
    return _cached_recommendations(
        model,
        joblib.hash(model),
        historical,
        products,
        inventory,
    )


@st.cache_data(
    ttl=900,
    max_entries=16,
    show_spinner="Calculating current restocking recommendations...",
)
def _cached_recommendations(
    _model,
    model_signature: str,
    historical: pd.DataFrame,
    products: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Cache forecasts and recommendations only for identical live inputs."""
    del model_signature
    return generate_restock_recommendations(_model, historical, products, inventory)


def forecast_for_app(
    model,
    historical: pd.DataFrame,
    products: pd.DataFrame,
    horizon: int,
    product_id: int | None = None,
) -> pd.DataFrame:
    """Reuse recursive forecasts while model, history, and selection are unchanged."""
    return _cached_forecast(
        model,
        joblib.hash(model),
        historical,
        products,
        horizon,
        product_id,
    )


@st.cache_data(
    ttl=900,
    max_entries=32,
    show_spinner="Generating demand forecast...",
)
def _cached_forecast(
    _model,
    model_signature: str,
    historical: pd.DataFrame,
    products: pd.DataFrame,
    horizon: int,
    product_id: int | None,
) -> pd.DataFrame:
    del model_signature
    return forecast_demand(_model, historical, products, horizon, product_id)


def prepare_evaluation():
    historical = load_historical_data(SALES_CSV_PATH, DATABASE_PATH)
    features = create_features(historical)
    _train, test = chronological_split(features)
    model = load_model_for_app()
    return evaluate_models(model, test)


def _show_setup_error(message: str) -> None:
    st.error(message)
    st.info(SETUP_INSTRUCTIONS)


def _format_table(frame: pd.DataFrame, column_config: dict | None = None) -> None:
    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        column_config=column_config,
    )


def _render_kpi_cards(cards: list[tuple[str, str, str, str]]) -> None:
    """Render compact, responsive KPI cards using values from application data."""
    if len(cards) == 7:
        rows = [
            (cards[:4], [1, 1, 1, 1], slice(None)),
            (cards[4:], [0.5, 1, 1, 1, 0.5], slice(1, 4)),
        ]
    else:
        rows = [
            (
                cards[offset : offset + 4],
                [1] * min(4, len(cards) - offset),
                slice(None),
            )
            for offset in range(0, len(cards), 4)
        ]

    card_index = 0
    for group, column_weights, column_slice in rows:
        columns = st.columns(column_weights)[column_slice]
        for column, (label, value, description, icon) in zip(columns, group):
            icon_markup = KPI_ICON_PATHS.get(icon)
            icon_html = (
                f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                f'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" '
                f'aria-hidden="true">{icon_markup}</svg>'
                if icon_markup
                else escape(icon)
            )
            with column:
                st.markdown(
                    f"""
                    <div class="glass-card kpi-card kpi-accent-{card_index % 4}">
                        <div class="kpi-topline"><span class="kpi-icon">{icon_html}</span>
                        <span class="kpi-label">{escape(label)}</span></div>
                        <div class="kpi-value">{escape(value)}</div>
                        <div class="kpi-description">{escape(description)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            card_index += 1


def _render_page_header(
    title: str,
    subtitle: str,
    eyebrow: str = "INVENTORY INTELLIGENCE",
    system_status: str | None = None,
) -> None:
    status_markup = (
        f'<div class="system-status"><span aria-hidden="true"></span>'
        f'{escape(system_status)}</div>'
        if system_status
        else ""
    )
    st.markdown(
        f"""
        <div class="page-hero">
            <div class="page-hero-top"><div class="page-eyebrow">{escape(eyebrow)}</div>{status_markup}</div>
            <h1>{escape(title)}</h1>
            <p>{escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _style_figure(figure: go.Figure) -> go.Figure:
    """Apply a consistent high-contrast dark chart theme to Plotly figures."""
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(9,16,34,0.2)",
        font={"color": "#d9e5f6", "family": "Inter, Segoe UI, sans-serif", "size": 12},
        title={"font": {"color": "#f3f6ff", "size": 17}},
        legend={
            "bgcolor": "rgba(12,18,37,0.48)",
            "bordercolor": "rgba(255,255,255,0.08)",
            "borderwidth": 1,
        },
        xaxis={
            "gridcolor": "rgba(174,194,225,0.10)",
            "linecolor": "rgba(174,194,225,0.18)",
            "zerolinecolor": "rgba(174,194,225,0.12)",
        },
        yaxis={
            "gridcolor": "rgba(174,194,225,0.10)",
            "linecolor": "rgba(174,194,225,0.18)",
            "zerolinecolor": "rgba(174,194,225,0.12)",
        },
        margin={"l": 22, "r": 18, "t": 58, "b": 24},
        hoverlabel={
            "bgcolor": "#18223e",
            "bordercolor": "rgba(134,157,205,0.32)",
            "font": {"color": "#f4f7ff"},
        },
    )
    return figure


def _rename_inventory(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={
            "product_name": "Product",
            "category": "Category",
            "sku": "SKU",
            "stock_quantity": "Current Stock",
            "unit_cost": "Unit Cost",
            "selling_price": "Selling Price",
            "supplier_lead_time_days": "Lead Time (days)",
            "updated_at": "Last Updated",
        }
    )


def _rename_recommendations(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={
            "product_name": "Product",
            "current_stock": "Current Stock",
            "predicted_lead_time_demand": "Lead-Time Demand",
            "average_daily_demand": "Avg. Daily Demand",
            "demand_standard_deviation": "Demand Std. Dev.",
            "safety_stock": "Safety Stock",
            "reorder_point": "Reorder Point",
            "recommended_order_quantity": "Recommended Order",
            "supplier_lead_time_days": "Lead Time (days)",
            "stock_status": "Status",
            "explanation": "Recommendation Explanation",
        }
    )


def render_overview(
    products: pd.DataFrame,
    inventory: pd.DataFrame,
    historical: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> None:
    hero_copy, hero_art = st.columns([1.45, 1], vertical_alignment="center")
    with hero_copy:
        st.markdown(
            f"""
            <div class="page-hero overview-hero">
                <div class="page-hero-top">
                    <div class="page-eyebrow">WELCOME BACK · LIVE OPERATIONS</div>
                    <div class="system-status"><span aria-hidden="true"></span>AI SYSTEM ONLINE · LIVE DATA</div>
                </div>
                <div class="overview-brand">{escape(BRAND_NAME)}</div>
                <h1>Inventory at a glance</h1>
                <p>AI-powered inventory intelligence and demand forecasting.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hero_art:
        with st.container(border=True):
            st.image(
                HERO_IMAGE_PATH,
                width="stretch",
                alt="Illustration of inventory boxes alongside demand and stock status panels.",
            )
    critical_count = int((recommendations["stock_status"] == "CRITICAL").sum())
    low_count = int((recommendations["stock_status"] == "LOW STOCK").sum())
    healthy_count = int((recommendations["stock_status"] == "HEALTHY").sum())
    total_stock = int(inventory["stock_quantity"].sum())
    total_sales = int(historical["units_sold"].sum())
    total_orders = int(recommendations["recommended_order_quantity"].sum())

    _render_kpi_cards(
        [
            ("Total products", f"{len(products):,}", "Active catalog items", "products"),
            ("Available stock", f"{total_stock:,}", "Units currently on hand", "stock"),
            ("Historical sales", f"{total_sales:,}", "Units in recorded sales", "sales"),
            ("Critical stock", f"{critical_count:,}", "Below calculated reorder point", "critical"),
            ("Low stock", f"{low_count:,}", "Within 20% above reorder point", "low_stock"),
            ("Healthy stock", f"{healthy_count:,}", "Above the low-stock buffer", "healthy"),
            ("Recommended purchase", f"{total_orders:,}", "Units suggested to order", "purchase"),
        ]
    )

    product_inventory = inventory.merge(
        products.loc[:, ["product_id", "product_name"]],
        on="product_id",
        validate="one_to_one",
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Stock quantity by product")
        chart = px.bar(
            product_inventory.sort_values("stock_quantity"),
            x="stock_quantity",
            y="product_name",
            orientation="h",
            labels={"stock_quantity": "Units in stock", "product_name": "Product"},
            color="stock_quantity",
            color_continuous_scale=[[0, "#705de5"], [1, "#55d6e8"]],
        )
        st.plotly_chart(_style_figure(chart), width="stretch")
    with right:
        st.subheader("Stock status")
        status_counts = (
            recommendations["stock_status"]
            .value_counts()
            .reindex(["CRITICAL", "LOW STOCK", "HEALTHY"], fill_value=0)
            .rename_axis("Status")
            .reset_index(name="Products")
        )
        chart = px.pie(
            status_counts,
            names="Status",
            values="Products",
            color="Status",
            color_discrete_map=STATUS_COLORS,
            hole=0.48,
        )
        st.plotly_chart(_style_figure(chart), width="stretch")

    daily = historical.groupby("sale_date", as_index=False)["units_sold"].sum()
    sales_chart = px.area(
        daily,
        x="sale_date",
        y="units_sold",
        labels={"sale_date": "Date", "units_sold": "Units sold"},
        title="Daily sales trend",
        color_discrete_sequence=["#55d6e8"],
    )
    sales_chart.update_traces(line={"width": 2}, fillcolor="rgba(85,214,232,0.12)")
    st.plotly_chart(_style_figure(sales_chart), width="stretch")

    top = (
        historical.groupby(["product_id", "product_name"], as_index=False)["units_sold"]
        .sum()
        .sort_values("units_sold", ascending=False)
        .head(5)
    )
    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.subheader("Top-selling products")
        st.plotly_chart(
            _style_figure(px.bar(
                top, x="product_name", y="units_sold",
                labels={"product_name": "Product", "units_sold": "Units sold"},
                color_discrete_sequence=["#9d83ff"],
            )),
            width="stretch",
        )
    with bottom_right:
        st.subheader("Products requiring immediate restocking")
        urgent = recommendations.loc[
            recommendations["stock_status"] == "CRITICAL"
        ].sort_values("recommended_order_quantity", ascending=False)
        if urgent.empty:
            st.success("No products are currently below their reorder point.")
        else:
            _format_table(
                _rename_recommendations(
                    urgent.loc[
                        :, [
                            "product_name",
                            "current_stock",
                            "reorder_point",
                            "recommended_order_quantity",
                            "stock_status",
                        ]
                    ]
                )
            )


def _inventory_details(
    database_path: Path, products: pd.DataFrame, inventory: pd.DataFrame
) -> pd.DataFrame:
    details = inventory.merge(products, on="product_id", validate="one_to_one")
    return details.sort_values("product_name").reset_index(drop=True)


def render_inventory_management() -> None:
    _render_page_header(
        "Inventory management",
        "Search the catalog, adjust on-hand quantities, and record received goods.",
    )
    try:
        products, inventory, _historical = load_dashboard_data()
        details = _inventory_details(DATABASE_PATH, products, inventory)
        recommendations = live_recommendations(
            load_model_for_app(), _historical, products, inventory
        )
        details = details.merge(
            recommendations.loc[:, ["product_id", "stock_status"]],
            on="product_id",
            validate="one_to_one",
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        _show_setup_error(str(exc))
        return

    status_counts = recommendations["stock_status"].value_counts()
    _render_kpi_cards(
        [
            ("Catalog", f"{len(details):,}", "Products with inventory records", "▦"),
            ("Units on hand", f"{int(inventory['stock_quantity'].sum()):,}", "Persisted in SQLite", "↗"),
            ("Critical", f"{int(status_counts.get('CRITICAL', 0)):,}", "Below reorder point", "!"),
            ("Healthy", f"{int(status_counts.get('HEALTHY', 0)):,}", "Above stock buffer", "✓"),
        ]
    )
    search = st.text_input("Search product, category, or SKU", key="inventory_search")
    selected_statuses = st.multiselect(
        "Stock status",
        ["CRITICAL", "LOW STOCK", "HEALTHY"],
        default=["CRITICAL", "LOW STOCK", "HEALTHY"],
        key="inventory_status_filter",
    )
    filtered = details
    filtered = filtered.loc[filtered["stock_status"].isin(selected_statuses)]
    if search.strip():
        query = search.strip().casefold()
        mask = (
            details["product_name"].str.casefold().str.contains(query, na=False)
            | details["category"].str.casefold().str.contains(query, na=False)
            | details["sku"].str.casefold().str.contains(query, na=False)
        )
        filtered = filtered.loc[mask]
    _format_table(
        _rename_inventory(
            filtered.loc[
                :,
                [
                    "product_name",
                    "category",
                    "sku",
                    "stock_quantity",
                    "unit_cost",
                    "selling_price",
                    "supplier_lead_time_days",
                    "stock_status",
                    "updated_at",
                ],
            ]
        ),
        {
            "Unit Cost": st.column_config.NumberColumn(format="$%.2f"),
            "Selling Price": st.column_config.NumberColumn(format="$%.2f"),
            "stock_status": st.column_config.TextColumn("Stock Status"),
        },
    )

    names = details.set_index("product_id")["product_name"].to_dict()
    selected_id = st.selectbox(
        "Product details",
        options=list(names),
        format_func=lambda value: names[value],
        key="inventory_product",
    )
    selected = details.loc[details["product_id"] == selected_id].iloc[0]
    st.info(
        f"**{selected['product_name']}** · {selected['category']} · SKU {selected['sku']} · "
        f"{int(selected['stock_quantity'])} in stock · {int(selected['supplier_lead_time_days'])} "
        "day supplier lead time"
    )

    adjustment_col, change_col = st.columns(2)
    with adjustment_col:
        st.subheader("Manual stock count")
        with st.form("absolute_stock_form"):
            new_stock = st.number_input(
                "New stock quantity",
                min_value=0,
                step=1,
                value=int(selected["stock_quantity"]),
            )
            reason = st.text_input("Adjustment reason", value="Manual stock count")
            submitted = st.form_submit_button("Save stock level")
        if submitted:
            try:
                update_inventory_stock(
                    DATABASE_PATH, int(selected_id), int(new_stock), reason
                )
                st.success("Stock level saved and inventory history recorded.")
                st.rerun()
            except (ValueError, RuntimeError) as exc:
                st.error(f"Could not update stock: {exc}")

    with change_col:
        st.subheader("Record stock adjustment")
        with st.form("stock_change_form"):
            direction = st.selectbox("Adjustment", ("Increase", "Decrease"))
            quantity = st.number_input("Adjustment quantity", min_value=1, step=1)
            change_reason = st.text_input("Reason", value="Inventory adjustment")
            submitted_change = st.form_submit_button("Apply adjustment")
        if submitted_change:
            delta = int(quantity) if direction == "Increase" else -int(quantity)
            try:
                record_inventory_change(
                    DATABASE_PATH, int(selected_id), delta, change_reason
                )
                st.success("Stock adjustment applied and recorded.")
                st.rerun()
            except (ValueError, RuntimeError) as exc:
                st.error(f"Could not apply adjustment: {exc}")

    st.subheader("Receive stock purchase")
    with st.form("inventory_purchase_form"):
        purchase_quantity = st.number_input(
            "Quantity received", min_value=1, step=1, key="inventory_purchase_quantity"
        )
        supplier = st.text_input("Supplier (optional)")
        confirm = st.checkbox("Confirm that these goods have been received")
        received = st.form_submit_button("Record received purchase")
    if received:
        if not confirm:
            st.error("Confirm receipt before recording a purchase.")
        else:
            try:
                purchase_id = record_stock_purchase(
                    DATABASE_PATH,
                    int(selected_id),
                    int(purchase_quantity),
                    float(selected["unit_cost"]),
                    supplier=supplier.strip() or None,
                )
                st.success(f"Purchase #{purchase_id} recorded and stock updated.")
                st.rerun()
            except (ValueError, RuntimeError) as exc:
                st.error(f"Could not record purchase: {exc}")

    history = fetch_inventory_history(DATABASE_PATH, int(selected_id))
    st.subheader("Inventory history")
    if history.empty:
        st.caption("No inventory changes have been recorded for this product.")
    else:
        _format_table(history)


def render_demand_forecasting(products: pd.DataFrame, historical: pd.DataFrame) -> None:
    _render_page_header(
        "AI demand forecasting",
        "Explore historical demand and model-generated daily outlooks for any product.",
        "PREDICTIVE ANALYTICS",
    )
    product_names = products.set_index("product_id")["product_name"].to_dict()
    control_col, horizon_col = st.columns(2)
    with control_col:
        product_id = st.selectbox(
            "Product",
            options=list(product_names),
            format_func=lambda value: product_names[value],
            key="forecast_product",
        )
    with horizon_col:
        horizon = st.selectbox("Forecast duration", (7, 30), format_func=lambda n: f"{n} days")

    try:
        model = load_model_for_app()
        forecast = forecast_for_app(
            model, historical, products, int(horizon), int(product_id)
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        _show_setup_error(str(exc))
        return

    product_history = historical.loc[
        historical["product_id"] == product_id, ["sale_date", "units_sold"]
    ].sort_values("sale_date")
    total = total_predicted_demand(forecast)
    _render_kpi_cards(
        [
            ("Forecast window", f"{horizon} days", "Recursive model horizon", "◷"),
            ("Predicted demand", f"{total:,.1f}", "Total forecast units", "↗"),
            ("Daily average", f"{total / horizon:,.1f}", "Predicted units per day", "∑"),
        ]
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=product_history["sale_date"],
            y=product_history["units_sold"],
            mode="lines",
            name="Historical sales",
            line={"color": "#288f82", "width": 2},
        )
    )
    forecast_start = forecast["forecast_date"].min()
    fig.add_vline(
        x=forecast_start,
        line_width=1,
        line_dash="dot",
        line_color="#f7b955",
        annotation_text="Forecast begins",
        annotation_position="top",
        annotation_font_color="#f7c96d",
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["forecast_date"],
            y=forecast["predicted_units"],
            mode="lines+markers",
            name="ML forecast",
            line={"color": "#ed9c47", "width": 3, "dash": "dash"},
        )
    )
    fig.update_layout(
        title=f"{product_names[product_id]} · history and {horizon}-day forecast",
        xaxis_title="Date",
        yaxis_title="Units",
        hovermode="x unified",
    )
    st.plotly_chart(_style_figure(fig), width="stretch")
    st.subheader("Forecast detail")
    _format_table(
        forecast.rename(
            columns={
                "forecast_date": "Forecast Date",
                "predicted_units": "Predicted Units",
            }
        ).loc[:, ["Forecast Date", "Predicted Units"]]
    )


def render_restocking(
    products: pd.DataFrame,
    inventory: pd.DataFrame,
    historical: pd.DataFrame,
) -> None:
    _render_page_header(
        "Restocking intelligence",
        "Forecast-led purchasing guidance matched to current stock and supplier lead times.",
        "DECISION SUPPORT",
    )
    try:
        model = load_model_for_app()
        recommendations = live_recommendations(model, historical, products, inventory)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        _show_setup_error(str(exc))
        return

    status_counts = recommendations["stock_status"].value_counts()
    _render_kpi_cards(
        [
            ("Critical", f"{int(status_counts.get('CRITICAL', 0)):,}", "Below reorder point", "!"),
            ("Low stock", f"{int(status_counts.get('LOW STOCK', 0)):,}", "Within 20% above reorder point", "△"),
            ("Healthy", f"{int(status_counts.get('HEALTHY', 0)):,}", "Above the low-stock buffer", "✓"),
            (
                "Purchase quantity",
                f"{int(recommendations['recommended_order_quantity'].sum()):,}",
                "Units recommended across catalog",
                "↗",
            ),
        ]
    )

    status_filter = st.multiselect(
        "Filter by stock status",
        ["CRITICAL", "LOW STOCK", "HEALTHY"],
        default=["CRITICAL", "LOW STOCK", "HEALTHY"],
    )
    search = st.text_input("Filter by product name", key="restock_search")
    filtered = recommendations.loc[recommendations["stock_status"].isin(status_filter)]
    if search.strip():
        filtered = filtered.loc[
            filtered["product_name"].str.contains(search.strip(), case=False, na=False)
        ]
    visible = _rename_recommendations(filtered)
    _format_table(
        visible.loc[
            :,
            [
                "Product",
                "Current Stock",
                "Lead-Time Demand",
                "Safety Stock",
                "Reorder Point",
                "Recommended Order",
                "Lead Time (days)",
                "Status",
            ],
        ],
        {
            "Status": st.column_config.TextColumn(
                "Status",
                help="Calculated from current stock relative to the reorder point.",
            ),
            "Recommended Order": st.column_config.NumberColumn(format="%d"),
        },
    )

    names = products.set_index("product_id")["product_name"].to_dict()
    selected_id = st.selectbox(
        "Recommendation details",
        options=list(names),
        format_func=lambda value: names[value],
        key="restock_product",
    )
    selected = recommendations.loc[
        recommendations["product_id"] == selected_id
    ].iloc[0]
    st.markdown(
        f"<div class='glass-card recommendation-detail status-{selected['stock_status'].lower().replace(' ', '-')}'"
        f"><div class='status-badge'>{escape(str(selected['stock_status']))}</div>"
        f"<p>{escape(str(selected['explanation']))}</p></div>",
        unsafe_allow_html=True,
    )
    _render_kpi_cards(
        [
            ("Current stock", f"{int(selected['current_stock']):,}", "Units on hand", "▦"),
            ("Lead-time demand", f"{selected['predicted_lead_time_demand']:.1f}", "Forecast units", "↗"),
            ("Safety stock", f"{selected['safety_stock']:.1f}", "Demand variability buffer", "◈"),
            ("Reorder point", f"{selected['reorder_point']:.1f}", "Calculated stock threshold", "⌁"),
            (
                "Recommended order",
                f"{int(selected['recommended_order_quantity']):,}",
                f"Units · {int(selected['supplier_lead_time_days'])} day lead time",
                "＋",
            ),
        ]
    )
    with st.expander("How this recommendation is calculated"):
        st.write(
            f"- Recursive model demand over the supplier's {int(selected['supplier_lead_time_days'])}-day "
            f"lead time: **{selected['predicted_lead_time_demand']:.2f} units**."
        )
        st.write(
            f"- Safety stock: **1.65 × {selected['demand_standard_deviation']:.2f} × "
            f"√{int(selected['supplier_lead_time_days'])} = {selected['safety_stock']:.2f} units**."
        )
        st.write(
            f"- Reorder point: **{selected['average_daily_demand']:.2f} × "
            f"{int(selected['supplier_lead_time_days'])} + "
            f"{selected['safety_stock']:.2f} = {selected['reorder_point']:.2f} units**."
        )
        st.write(
            f"- Recommended quantity: **max(0, forecast demand + safety stock - stock) = "
            f"{int(selected['recommended_order_quantity'])} units** (rounded up to a whole unit)."
        )

    st.subheader("Receive a recommended purchase")
    product_inventory = inventory.set_index("product_id")
    with st.form("recommendation_purchase_form"):
        receive_quantity = st.number_input(
            "Received quantity",
            min_value=1,
            value=max(1, int(selected["recommended_order_quantity"])),
            step=1,
        )
        supplier = st.text_input("Supplier (optional)", key="restock_supplier")
        confirm = st.checkbox(
            f"Confirm {int(receive_quantity)} units of {names[selected_id]} were received",
            key="restock_received_confirm",
        )
        submitted = st.form_submit_button("Record received purchase")
    if submitted:
        if not confirm:
            st.error("Confirm receipt before recording a purchase.")
        else:
            try:
                purchase_id = record_stock_purchase(
                    DATABASE_PATH,
                    int(selected_id),
                    int(receive_quantity),
                    float(product_inventory.loc[selected_id, "unit_cost"]),
                    supplier=supplier.strip() or None,
                )
                st.success(
                    f"Purchase #{purchase_id} was recorded. Recommendations will use "
                    "the updated stock on the next refresh."
                )
                st.rerun()
            except (ValueError, RuntimeError) as exc:
                st.error(f"Could not record purchase: {exc}")


def render_sales_analytics(historical: pd.DataFrame) -> None:
    _render_page_header(
        "Sales analytics",
        "Understand sales momentum, product performance, and category contribution.",
        "HISTORICAL PERFORMANCE",
    )
    daily = (
        historical.groupby("sale_date", as_index=False)["units_sold"]
        .sum()
        .sort_values("sale_date")
    )
    _render_kpi_cards(
        [
            ("Units sold", f"{int(historical['units_sold'].sum()):,}", "Across loaded history", "↗"),
            ("Sales days", f"{historical['sale_date'].nunique():,}", "Distinct recorded dates", "◷"),
            ("Products", f"{historical['product_id'].nunique():,}", "Products with sales", "▦"),
        ]
    )
    daily_chart = px.area(
        daily,
        x="sale_date",
        y="units_sold",
        markers=False,
        title="Daily sales",
        color_discrete_sequence=["#55d6e8"],
    )
    daily_chart.update_traces(line={"width": 2}, fillcolor="rgba(85,214,232,0.12)")
    st.plotly_chart(_style_figure(daily_chart), width="stretch")

    sales = historical.copy()
    sales["week"] = sales["sale_date"].dt.to_period("W").dt.start_time
    sales["month"] = sales["sale_date"].dt.to_period("M").dt.start_time
    weekly = sales.groupby("week", as_index=False)["units_sold"].sum()
    monthly = sales.groupby("month", as_index=False)["units_sold"].sum()
    weekly_col, monthly_col = st.columns(2)
    with weekly_col:
        st.subheader("Weekly sales")
        st.plotly_chart(
            _style_figure(px.bar(
                weekly, x="week", y="units_sold", labels={"week": "Week"},
                color_discrete_sequence=["#9d83ff"],
            )),
            width="stretch",
        )
    with monthly_col:
        st.subheader("Monthly sales")
        _format_table(monthly.rename(columns={"month": "Month", "units_sold": "Units Sold"}))
        st.plotly_chart(
            _style_figure(px.bar(
                monthly, x="month", y="units_sold", labels={"month": "Month"},
                color_discrete_sequence=["#55d6e8"],
            )),
            width="stretch",
        )

    product_sales = (
        sales.groupby(["product_id", "product_name"], as_index=False)["units_sold"]
        .sum()
        .sort_values("units_sold", ascending=False)
    )
    top_col, low_col = st.columns(2)
    with top_col:
        st.subheader("Top-selling products")
        _format_table(
            product_sales.head(5).rename(
                columns={"product_name": "Product", "units_sold": "Units Sold"}
            )
        )
    with low_col:
        st.subheader("Lowest-selling products")
        _format_table(
            product_sales.tail(5).sort_values("units_sold").rename(
                columns={"product_name": "Product", "units_sold": "Units Sold"}
            )
        )

    category_sales = (
        sales.groupby("category", as_index=False)["units_sold"]
        .sum()
        .sort_values("units_sold", ascending=False)
    )
    st.subheader("Sales by product category")
    st.plotly_chart(
        _style_figure(px.bar(
            category_sales, x="category", y="units_sold", color="category",
            color_discrete_sequence=CHART_COLORS,
        )),
        width="stretch",
    )
    st.subheader("Sales quantity comparison")
    st.plotly_chart(
        _style_figure(px.bar(
            product_sales,
            x="product_name",
            y="units_sold",
            color="units_sold",
            color_continuous_scale=[[0, "#705de5"], [1, "#55d6e8"]],
            labels={"product_name": "Product", "units_sold": "Historical units sold"},
        )),
        width="stretch",
    )


def render_model_performance() -> None:
    _render_page_header(
        "Model performance",
        "Monitor chronological holdout performance against a transparent baseline.",
        "MODEL MONITORING",
    )
    try:
        comparison, performance, product_performance = prepare_evaluation()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        _show_setup_error(str(exc))
        return

    rf = performance.loc[performance["model"] == "Random Forest"].iloc[0]
    baseline = performance.loc[
        performance["model"] == "Moving Average (7-day)"
    ].iloc[0]
    _render_kpi_cards(
        [
            ("Random Forest · MAE", f"{rf['mae']:.4f}", "Average absolute error", "RF"),
            ("Random Forest · RMSE", f"{rf['rmse']:.4f}", "Large misses weighted more", "RF"),
            ("Random Forest · R²", f"{rf['r2']:.4f}", "Explained variance", "RF"),
            ("Moving Average · MAE", f"{baseline['mae']:.4f}", "7-day baseline", "MA"),
            ("Moving Average · RMSE", f"{baseline['rmse']:.4f}", "7-day baseline", "MA"),
            ("Moving Average · R²", f"{baseline['r2']:.4f}", "7-day baseline", "MA"),
        ]
    )
    better_scores = {
        "Random Forest": int(rf["mae"] < baseline["mae"])
        + int(rf["rmse"] < baseline["rmse"])
        + int(rf["r2"] > baseline["r2"]),
        "Moving Average": int(baseline["mae"] < rf["mae"])
        + int(baseline["rmse"] < rf["rmse"])
        + int(baseline["r2"] > rf["r2"]),
    }
    winner = max(better_scores, key=better_scores.get)
    st.success(f"{winner} leads on {better_scores[winner]} of the 3 displayed metrics.")

    st.plotly_chart(
        _style_figure(px.bar(
            performance.melt(id_vars="model", var_name="Metric", value_name="Score"),
            x="Metric",
            y="Score",
            color="model",
            barmode="group",
            title="Random Forest vs. 7-day moving average",
            color_discrete_sequence=["#55d6e8", "#9d83ff"],
        )),
        width="stretch",
    )
    st.subheader("Actual and predicted demand")
    product_names = sorted(comparison["product_name"].unique().tolist())
    selected_name = st.selectbox("Product", product_names, key="evaluation_product")
    product_comparison = comparison.loc[
        comparison["product_name"] == selected_name
    ].sort_values("sale_date")
    fig = go.Figure()
    for column, label in (
        ("actual_units", "Actual"),
        ("random_forest_prediction", "Random Forest"),
        ("moving_average_prediction", "Moving Average"),
    ):
        fig.add_trace(
            go.Scatter(
                x=product_comparison["sale_date"],
                y=product_comparison[column],
                mode="lines",
                name=label,
                line={"color": {
                    "Actual": "#eef3ff",
                    "Random Forest": "#55d6e8",
                    "Moving Average": "#9d83ff",
                }[label]},
            )
        )
    st.plotly_chart(_style_figure(fig), width="stretch")

    st.subheader("Product-wise model performance")
    _format_table(product_performance.rename(
        columns={
            "product_name": "Product",
            "model": "Model",
            "mae": "MAE",
            "rmse": "RMSE",
            "r2": "R²",
        }
    ))
    st.subheader("Model and metric guide")
    st.markdown(
        """
        - **Random Forest** combines many decision trees to learn relationships
          between product, calendar, and recent-sales features.
        - **Moving Average** is a simple seven-day recent-demand baseline; it
          shows whether the trained model improves on a transparent heuristic.
        - **MAE** is the average absolute error in units. Lower is better.
        - **RMSE** is the square root of the average squared error and penalizes
          larger misses more strongly. Lower is better.
        - **R²** measures explained variation relative to predicting the test
          mean; values closer to 1 are better and negative values are possible.
        - **Chronological splitting** trains on earlier dates and tests on later
          dates, preserving the real direction of time rather than mixing future
          observations into training.
        - **One-step-ahead evaluation** predicts each test date from preceding
          actual history. **Recursive forecasting** predicts future dates using
          earlier model predictions as later lag inputs, so it is a separate
          operation and is not represented by these holdout metrics.
        """
    )
    st.info(EVALUATION_NOTE)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def build_download_reports(
    products: pd.DataFrame,
    inventory: pd.DataFrame,
    historical: pd.DataFrame,
    model,
) -> tuple[list[tuple[str, str, pd.DataFrame]], pd.DataFrame]:
    """Build every downloadable report from the current database snapshot."""
    recommendations = live_recommendations(model, historical, products, inventory)
    forecast_7 = forecast_for_app(model, historical, products, horizon=7)
    forecast_30 = forecast_for_app(model, historical, products, horizon=30)

    inventory_report = inventory.merge(
        products, on="product_id", validate="one_to_one"
    ).rename(
        columns={
            "product_name": "Product",
            "category": "Category",
            "sku": "SKU",
            "stock_quantity": "Current Stock",
            "unit_cost": "Unit Cost",
            "selling_price": "Selling Price",
            "supplier_lead_time_days": "Lead Time (days)",
        }
    )
    sales_report = historical.rename(
        columns={
            "product_name": "Product",
            "category": "Category",
            "sku": "SKU",
            "sale_date": "Sale Date",
            "units_sold": "Units Sold",
        }
    )
    restock_report = _rename_recommendations(recommendations)
    reports = [
        ("Complete inventory report", "inventory_report.csv", inventory_report),
        ("Sales history", "sales_history.csv", sales_report),
        ("7-day forecast", "forecast_7_day.csv", forecast_7),
        ("30-day forecast", "forecast_30_day.csv", forecast_30),
        ("Restocking recommendations", "restock_recommendations.csv", restock_report),
    ]
    summary = pd.DataFrame(
        [
            {"Metric": "Total products", "Value": int(len(products))},
            {"Metric": "Total available stock", "Value": int(inventory["stock_quantity"].sum())},
            {"Metric": "Total historical sales", "Value": int(historical["units_sold"].sum())},
            {
                "Metric": "Critical products",
                "Value": int((recommendations["stock_status"] == "CRITICAL").sum()),
            },
            {
                "Metric": "Recommended purchase quantity",
                "Value": int(recommendations["recommended_order_quantity"].sum()),
            },
        ]
    )
    return reports, summary


def render_reports(
    products: pd.DataFrame,
    inventory: pd.DataFrame,
    historical: pd.DataFrame,
) -> None:
    _render_page_header(
        "Reports & downloads",
        "Export current inventory, sales, forecasts, and restocking decisions.",
        "DATA EXPORTS",
    )
    try:
        model = load_model_for_app()
        reports, summary = build_download_reports(
            products, inventory, historical, model
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        _show_setup_error(str(exc))
        return

    descriptions = {
        "inventory_report.csv": "Current product details, pricing, supplier lead time, and on-hand stock.",
        "sales_history.csv": "Daily product-level sales records loaded from SQLite.",
        "forecast_7_day.csv": "Recursive model output for each product over the next seven days.",
        "forecast_30_day.csv": "Recursive model output for each product over the next thirty days.",
        "restock_recommendations.csv": "Current stock status and model-driven lead-time purchase quantities.",
    }
    for offset in range(0, len(reports), 2):
        columns = st.columns(2)
        for column, (label, filename, frame) in zip(columns, reports[offset : offset + 2]):
            with column:
                with st.container(border=True):
                    st.markdown(f"#### {label}")
                    st.caption(descriptions[filename])
                    st.download_button(
                        f"Download {label}",
                        data=_csv_bytes(frame),
                        file_name=filename,
                        mime="text/csv",
                        key=f"download_{filename}",
                        width="stretch",
                    )

    st.subheader("Summary report")
    _format_table(summary)
    with st.container(border=True):
        st.markdown("#### Executive summary")
        st.caption("A compact, current snapshot of catalog, stock, sales, and purchase needs.")
        st.download_button(
            "Download summary report",
            data=_csv_bytes(summary),
            file_name="inventory_summary.csv",
            mime="text/csv",
            key="download_summary",
            width="stretch",
        )


def render_app() -> None:
    st.set_page_config(
        page_title=f"{BRAND_NAME} · Inventory Intelligence",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
            --page-bg: #080d1b;
            --panel: rgba(17, 25, 47, 0.74);
            --panel-strong: rgba(21, 31, 57, 0.92);
            --line: rgba(185, 204, 245, 0.13);
            --muted: #9eacc6;
            --text: #edf2ff;
            --cyan: #55d6e8;
            --purple: #9d83ff;
        }
        html, body, [class*="css"] {
            font-family: Inter, "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .stApp {
            color: var(--text);
            background:
                radial-gradient(ellipse at 84% 3%, rgba(100, 79, 190, 0.20), transparent 36rem),
                radial-gradient(ellipse at 8% 30%, rgba(28, 132, 175, 0.13), transparent 34rem),
                radial-gradient(ellipse at 78% 78%, rgba(54, 74, 152, 0.10), transparent 38rem),
                var(--page-bg);
        }
        [data-testid="stAppViewContainer"] {
            background: transparent;
        }
        [data-testid="stHeader"] {
            background: rgba(8, 13, 27, 0.56);
            border-bottom: 1px solid rgba(185, 204, 245, 0.06);
        }
        [data-testid="stMain"] > div {
            padding-top: 1.65rem;
        }
        .block-container {
            max-width: 1540px;
            padding-left: clamp(1rem, 3vw, 2.6rem);
            padding-right: clamp(1rem, 3vw, 2.6rem);
            padding-bottom: 3rem;
        }
        h1, h2, h3, h4 {
            color: #f3f6ff !important;
            letter-spacing: -0.035em;
        }
        h2, h3 {
            margin-top: 1.15rem !important;
        }
        p, label, [data-testid="stMarkdownContainer"] {
            color: #d7e1f3;
        }
        [data-testid="stCaptionContainer"], .stCaption {
            color: var(--muted) !important;
        }
        .page-hero {
            position: relative;
            overflow: hidden;
            padding: clamp(1rem, 2.1vw, 1.45rem) clamp(1.2rem, 2.6vw, 1.85rem);
            margin: 0 0 1.05rem;
            border: 1px solid rgba(189, 205, 244, 0.16);
            border-radius: 19px;
            background:
                radial-gradient(ellipse at 88% 0%, rgba(113, 92, 224, 0.18), transparent 42%),
                linear-gradient(118deg, rgba(20, 37, 66, 0.77), rgba(15, 23, 45, 0.68));
            box-shadow: 0 14px 40px rgba(0, 0, 0, 0.20),
                inset 0 1px rgba(255,255,255,0.075), inset 0 0 24px rgba(127,151,226,0.025);
            -webkit-backdrop-filter: blur(15px) saturate(125%);
            backdrop-filter: blur(15px) saturate(125%);
        }
        .page-hero-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            min-height: 1.25rem;
        }
        .page-hero h1 {
            margin: 0.36rem 0 0.24rem;
            font-size: clamp(1.65rem, 2.7vw, 2.3rem);
            line-height: 1.12;
        }
        .page-hero p {
            margin: 0;
            color: #aebbd3;
            font-size: 0.96rem;
        }
        .overview-hero {
            display: flex;
            min-height: 100%;
            flex-direction: column;
            justify-content: center;
            padding: clamp(1.25rem, 2.4vw, 2rem);
            margin: 0;
        }
        .overview-brand {
            margin-top: 1.05rem;
            color: #c7baff;
            font-size: 0.78rem;
            font-weight: 780;
            letter-spacing: 0.13em;
            text-transform: uppercase;
        }
        [data-testid="stImage"] {
            overflow: hidden;
            border-radius: 12px;
        }
        [data-testid="stImage"] img {
            display: block;
            width: 100%;
            height: auto;
        }
        .page-eyebrow {
            color: var(--cyan);
            font-size: 0.68rem;
            font-weight: 750;
            letter-spacing: 0.14em;
        }
        .system-status {
            display: inline-flex;
            flex: 0 0 auto;
            align-items: center;
            gap: 0.45rem;
            padding: 0.32rem 0.62rem;
            border: 1px solid rgba(104, 227, 180, 0.2);
            border-radius: 999px;
            color: #b7e9d8;
            background: rgba(32, 127, 101, 0.11);
            font-size: 0.65rem;
            font-weight: 700;
            letter-spacing: 0.055em;
        }
        .system-status > span {
            width: 0.42rem;
            height: 0.42rem;
            border-radius: 50%;
            background: #68e3b4;
            box-shadow: 0 0 9px rgba(104, 227, 180, 0.42);
        }
        .glass-card, [data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid rgba(185, 204, 245, 0.15);
            border-radius: 17px;
            background:
                linear-gradient(145deg, rgba(25, 36, 64, 0.77), rgba(12, 19, 38, 0.73));
            box-shadow: 0 12px 30px rgba(0,0,0,0.19),
                inset 0 1px rgba(255,255,255,0.07), inset 0 0 22px rgba(140,160,230,0.025);
            -webkit-backdrop-filter: blur(15px) saturate(125%);
            backdrop-filter: blur(15px) saturate(125%);
        }
        .kpi-card {
            position: relative;
            display: flex;
            flex-direction: column;
            min-height: 142px;
            height: 142px;
            overflow: hidden;
            padding: 0.95rem 1rem 0.9rem;
            margin-bottom: 0.75rem;
            transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
        }
        .kpi-card:hover {
            transform: translateY(-2px);
            border-color: rgba(110, 211, 240, 0.35);
            box-shadow: 0 16px 34px rgba(0,0,0,0.24),
                0 0 18px rgba(91,177,255,0.075), inset 0 1px rgba(255,255,255,0.09);
        }
        .kpi-card::after {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 3px;
            height: 100%;
            background: linear-gradient(180deg, #55d6e8, #9276ff);
            opacity: 0.9;
        }
        .kpi-accent-1::after { background: linear-gradient(180deg, #9d83ff, #55a9ff); }
        .kpi-accent-2::after { background: linear-gradient(180deg, #55d6e8, #3fa6c7); }
        .kpi-accent-3::after { background: linear-gradient(180deg, #f7b955, #ff7b96); }
        .kpi-topline {
            display: flex;
            align-items: center;
            gap: 0.62rem;
            min-height: 2rem;
            color: #9faccc;
            font-size: 0.82rem;
            font-weight: 650;
        }
        .kpi-icon {
            display: inline-grid;
            flex: 0 0 2rem;
            place-items: center;
            width: 2rem;
            height: 2rem;
            border: 1px solid rgba(145, 159, 255, 0.23);
            border-radius: 10px;
            color: var(--cyan);
            background: linear-gradient(145deg, rgba(108, 108, 211, 0.17), rgba(46, 151, 185, 0.08));
            box-shadow: inset 0 1px rgba(255,255,255,0.07);
        }
        .kpi-icon svg {
            width: 1.08rem;
            height: 1.08rem;
        }
        .kpi-label {
            display: -webkit-box;
            overflow: hidden;
            -webkit-box-orient: vertical;
            -webkit-line-clamp: 2;
            line-height: 1.25rem;
        }
        .kpi-value {
            margin-top: 0.58rem;
            color: #f4f7ff;
            font-size: clamp(1.55rem, 2.2vw, 2.1rem);
            font-weight: 740;
            line-height: 1;
            letter-spacing: -0.045em;
        }
        .kpi-description {
            margin-top: auto;
            padding-top: 0.42rem;
            color: #a1afc8;
            font-size: 0.72rem;
            line-height: 1.25;
        }
        .recommendation-detail {
            padding: 1.1rem 1.25rem;
            margin: 1rem 0;
        }
        .recommendation-detail p {
            margin: 0.6rem 0 0;
            color: #d9e3f5;
        }
        .status-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.28rem 0.72rem;
            border: 1px solid currentColor;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.08em;
        }
        .status-critical .status-badge { color: #ff8291; background: rgba(255,101,122,0.11); }
        .status-low-stock .status-badge { color: #ffd17b; background: rgba(247,185,85,0.1); }
        .status-healthy .status-badge { color: #68e3b4; background: rgba(67,213,164,0.1); }
        [data-testid="stVerticalBlockBorderWrapper"] {
            padding: 1rem 1.1rem;
        }
        [data-testid="stMetric"] {
            padding: 0.9rem 1rem;
            border: 1px solid var(--line);
            border-radius: 15px;
            background: var(--panel);
            backdrop-filter: blur(10px);
        }
        [data-testid="stMetricLabel"] { color: #a8b5ce !important; }
        [data-testid="stMetricValue"] { color: #f3f6ff !important; }
        [data-testid="stPlotlyChart"] {
            overflow: hidden;
            border: 1px solid rgba(185,204,245,0.13);
            border-radius: 17px;
            background: linear-gradient(145deg, rgba(20,30,53,0.54), rgba(11,17,34,0.47));
            box-shadow: 0 10px 28px rgba(0,0,0,0.16),
                inset 0 1px rgba(255,255,255,0.045);
            -webkit-backdrop-filter: blur(13px) saturate(120%);
            backdrop-filter: blur(13px) saturate(120%);
            transition: border-color 180ms ease, box-shadow 180ms ease;
        }
        [data-testid="stPlotlyChart"]:hover {
            border-color: rgba(125, 163, 225, 0.22);
            box-shadow: 0 13px 30px rgba(0,0,0,0.19),
                inset 0 1px rgba(255,255,255,0.06);
        }
        [data-testid="stDataFrame"] {
            overflow: hidden;
            border: 1px solid var(--line);
            border-radius: 15px;
            background: rgba(12,18,35,0.48);
        }
        [data-testid="stForm"] {
            padding: 1.1rem;
            border: 1px solid var(--line);
            border-radius: 17px;
            background: linear-gradient(145deg, rgba(24,34,60,0.72), rgba(13,20,39,0.78));
            box-shadow: 0 12px 30px rgba(0,0,0,0.16);
        }
        input, textarea, [data-baseweb="select"] > div {
            border-color: rgba(180,198,236,0.2) !important;
            border-radius: 10px !important;
            background-color: rgba(10,16,32,0.75) !important;
        }
        [data-testid="stButton"] button, [data-testid="stDownloadButton"] button,
        [data-testid="stFormSubmitButton"] button {
            min-height: 2.55rem;
            border: 1px solid rgba(126,161,250,0.34);
            border-radius: 10px;
            color: #f5f7ff;
            background: linear-gradient(105deg, rgba(111,83,224,0.92), rgba(47,154,190,0.9));
            font-weight: 680;
            box-shadow: 0 7px 18px rgba(51,79,174,0.18);
            transition: filter 140ms ease, transform 140ms ease, box-shadow 140ms ease;
        }
        [data-testid="stButton"] button:hover, [data-testid="stDownloadButton"] button:hover,
        [data-testid="stFormSubmitButton"] button:hover {
            filter: brightness(1.12);
            transform: translateY(-1px);
            box-shadow: 0 9px 22px rgba(51,115,204,0.28);
        }
        [data-testid="stSidebar"] {
            background:
                radial-gradient(ellipse at 14% 0%, rgba(103,73,186,0.22), transparent 23rem),
                linear-gradient(180deg, rgba(11,17,34,0.96), rgba(9,15,29,0.98));
            border-right: 1px solid rgba(185,204,245,0.1);
        }
        [data-testid="stSidebar"] > div {
            background: transparent;
        }
        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.7rem 0.4rem 1.1rem;
            margin-bottom: 0.8rem;
            border-bottom: 1px solid rgba(185,204,245,0.12);
        }
        .brand-icon {
            display: grid;
            place-items: center;
            width: 2.75rem;
            height: 2.75rem;
            border: 1px solid rgba(151,133,255,0.38);
            border-radius: 13px;
            background: linear-gradient(145deg, rgba(131,91,255,0.26), rgba(55,195,220,0.16));
            box-shadow: 0 0 20px rgba(104,100,235,0.12);
        }
        .brand-name { color: #f3f6ff; font-size: 1.02rem; font-weight: 760; }
        .brand-caption { margin-top: 0.18rem; color: #98a8c5; font-size: 0.72rem; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            min-height: 2.6rem;
            padding: 0.46rem 0.7rem;
            margin: 0.15rem 0;
            border: 1px solid transparent;
            border-radius: 10px;
            color: #aebbd2 !important;
            transition: background 140ms ease, color 140ms ease, border-color 140ms ease;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
            border-color: rgba(128,142,255,0.28);
            color: #f2f4ff !important;
            background: linear-gradient(95deg, rgba(117,84,230,0.28), rgba(54,174,201,0.13));
            box-shadow: inset 3px 0 #8e7aff;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            border-color: rgba(165,184,232,0.16);
            background-color: rgba(138,158,209,0.08);
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label p {
            color: inherit !important;
            font-weight: 590;
        }
        .sidebar-footer {
            padding: 0.8rem 0.45rem;
            margin-top: 1.3rem;
            border-top: 1px solid rgba(185,204,245,0.11);
            color: #8392ad;
            font-size: 0.7rem;
            line-height: 1.6;
        }
        [data-testid="stAlert"] {
            border-radius: 13px;
            border-color: rgba(145,165,210,0.22);
            background: rgba(20,30,53,0.78);
        }
        [data-testid="stExpander"] {
            border-color: var(--line);
            border-radius: 13px;
            background: rgba(17,25,47,0.48);
        }
        @media (max-width: 760px) {
            .block-container { padding: 1rem 0.8rem 2rem; }
            .page-hero { padding: 1.05rem 1rem; border-radius: 16px; }
            .overview-hero { min-height: auto; }
            .page-hero-top { align-items: flex-start; }
            .system-status { max-width: 48%; white-space: normal; line-height: 1.2; }
            .kpi-card { min-height: 138px; height: 138px; padding: 0.85rem; }
            .kpi-topline { gap: 0.48rem; font-size: 0.77rem; }
            .kpi-icon { flex-basis: 1.85rem; width: 1.85rem; height: 1.85rem; }
            [data-testid="stDataFrame"] { overflow-x: auto; }
        }
        @media (max-width: 430px) {
            .page-hero-top { flex-wrap: wrap; }
            .system-status { max-width: 100%; }
            .kpi-card { padding: 0.78rem; }
            .kpi-label { font-size: 0.74rem; }
            .kpi-description { font-size: 0.68rem; }
        }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after { transition-duration: 0.01ms !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-brand">
                <div class="brand-icon" role="img" aria-label="{escape(BRAND_NAME)} logo">
                    <svg width="25" height="25" viewBox="0 0 24 24" fill="none"
                         xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                        <path d="M4 8.5 12 4l8 4.5v8L12 21l-8-4.5v-8Z"
                              stroke="#79DDF0" stroke-width="1.5" stroke-linejoin="round"/>
                        <path d="m4.5 8.8 7.5 4.3 7.5-4.3M12 13v7.5"
                              stroke="#A994FF" stroke-width="1.5" stroke-linecap="round"/>
                        <path d="m15.7 5.9 1.1-2m2.1 3.1 2-.8"
                              stroke="#79DDF0" stroke-width="1.4" stroke-linecap="round"/>
                    </svg>
                </div>
                <div>
                    <div class="brand-name">{escape(BRAND_NAME)}</div>
                    <div class="brand-caption">Intelligent operations</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        page = st.radio("Navigate", NAVIGATION, label_visibility="collapsed")
        st.divider()
        st.markdown(
            '<div class="sidebar-footer">AI-POWERED INVENTORY SYSTEM<br>'
            'Local analytics · SQLite · Random Forest</div>',
            unsafe_allow_html=True,
        )

    try:
        products, inventory, historical = load_dashboard_data()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        _show_setup_error(str(exc))
        return

    if page == "Inventory Management":
        render_inventory_management()
    elif page == "Demand Forecasting":
        render_demand_forecasting(products, historical)
    elif page == "Restocking Recommendations":
        render_restocking(products, inventory, historical)
    elif page == "Sales Analytics":
        render_sales_analytics(historical)
    elif page == "Model Performance":
        render_model_performance()
    elif page == "Reports and Downloads":
        render_reports(products, inventory, historical)
    else:
        try:
            recommendations = live_recommendations(
                load_model_for_app(), historical, products, inventory
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            _show_setup_error(str(exc))
            return
        render_overview(products, inventory, historical, recommendations)


if __name__ == "__main__":
    render_app()
