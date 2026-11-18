# Technical Reference Notes for the Formal Project Report

**Formal project title:** AI-Based Intelligent Inventory Management and Demand Forecasting System
**Application brand:** INVENTRA
**Purpose of this document:** Source-grounded notes for preparing the formal college project report. These notes describe the implementation inspected in this workspace; they are not claims of production deployment or real-world forecasting accuracy.

## 1. Project overview and problem statement

Retailers need to maintain enough stock to meet customer demand without tying up excessive money in inventory. Manual stock checks, fixed reorder rules, and spreadsheet-based sales review can make it difficult to anticipate changes in demand or account for different supplier lead times.

This project combines a local product and inventory register, daily sales history, a trained machine-learning model, and forecast-based replenishment calculations. The application displays current records from SQLite, forecasts product demand, compares the model with a simple baseline, and presents purchase recommendations. The source sales data and initial stock are synthetic; the project has not established accuracy on real retail operations.

## 2. Objectives and scope

### Implemented objectives

- Maintain product, inventory, sales, inventory-change, and received-purchase records in SQLite.
- Generate reproducible synthetic daily sales for ten products.
- Engineer lagged and calendar features and train a shared Random Forest regression pipeline.
- Produce recursive daily forecasts for a selected product or the full catalog, including 7- and 30-day horizons.
- Evaluate the model against a seven-day moving-average baseline using MAE, RMSE, and R².
- Calculate stock status and purchase recommendations using forecasts, historical demand variation, and supplier lead time.
- Provide a seven-page Streamlit interface and CSV downloads sourced from the current application data.

### Scope observed in code

The application is a local, single-database demonstration. It has no user authentication, cloud service, external API, point-of-sale integration, multi-location inventory, automated supplier ordering, or automated model retraining workflow. Those are not represented as implemented features.

## 3. Existing system and proposed system

### Existing-system comparison

The codebase does not identify a particular retailer, deployed inventory product, or measured manual workflow as its existing system. A formal report may use a general comparison—manual or spreadsheet-based tracking can require staff to review sales and stock separately, and fixed reorder rules may not incorporate a demand forecast—but should not present that comparison as a measured fact about a named organization.

**Needs confirmation:** Supply the actual existing process or system at the intended organization, if the college requires a case-specific comparison. Otherwise, describe the manual/spreadsheet workflow as a general problem context, not as a verified local baseline.

### Proposed system

The implemented proposal is a local application that combines a SQLite-backed catalog and transaction history with a saved Random Forest pipeline. It displays historical sales, future forecasts, model evaluation, and replenishment guidance in Streamlit. Stock changes and received purchases are persisted and audited in the same SQLite database.

## 4. Technology stack and purpose

Versions below were queried from the project `.venv` during this inspection. `requirements.txt` lists package names without version pins, so these versions describe the inspected environment, not a guaranteed installation lock.

| Technology | Inspected version / source | Purpose in this project |
|---|---:|---|
| Python | 3.10.0 | Application, data processing, database access, model training, scripts, and tests. |
| Streamlit | 1.65.0 | Seven-page interactive dashboard, widgets, forms, charts, and CSV downloads. |
| Pandas | 2.3.3 | Tabular sales, inventory, feature, evaluation, and report data. |
| NumPy | 2.2.6 | Reproducible random-number generation, numerical features, and forecast calculations. |
| scikit-learn | 1.7.2 | Preprocessing pipeline, Random Forest regressor, and evaluation metrics. |
| Plotly | 7.1.0 | Interactive dashboard charts. |
| Matplotlib | 3.10.9 | Listed as a dependency; no direct Matplotlib import was found in the project Python source inspected. |
| Joblib | 1.6.0 | Persisting and loading the fitted scikit-learn pipeline. |
| SQLite / `sqlite3` | Python standard-library interface | Local relational storage and transactional inventory operations. |
| `unittest` | Python standard library | Automated unit and application-data tests. |

The application has no external API dependency. Plotly, Streamlit, and the other listed packages are used locally.

## 5. System architecture and data flow

1. `scripts/setup_phase1.py` calls `generate_synthetic_data()`, writes `data/synthetic_sales.csv`, and initializes the SQLite database if it does not exist. When the database already exists, it calls `refresh_sales()` so the sales history can be replaced while existing product and inventory records are preserved.
2. `inventory_system.preprocessing.load_historical_data()` reads and validates the sales CSV, obtains product details from SQLite, checks product references and daily date continuity, and returns product-level history.
3. `create_features()` constructs past-sales and calendar inputs. `chronological_split()` divides each product's rows by date.
4. `scripts/train_model.py` trains the scikit-learn pipeline using the earlier portion of each product's data, evaluates it on the later portion, and saves the fitted pipeline under `models/random_forest_demand.joblib`.
5. Forecast and restocking scripts load that saved artifact and use the persisted product and sales data. The Streamlit application instead loads current product, inventory, and sales records from SQLite and merges product details into historical sales.
6. `forecast_demand()` iteratively predicts future dates. Restocking uses those recursive predictions over each product's supplier lead time and combines them with current stock and historical demand statistics.
7. `app.py` routes navigation, renders charts/tables/forms, and produces reports. Inventory adjustments and received purchases are written to SQLite and represented in the audit tables.

The app caches the immutable loaded model and caches forecast/recommendation results for matching inputs for up to 15 minutes. Inventory is read from SQLite on app reruns; a changed inventory input therefore changes the recommendation cache key. The helper module `inventory_system.visualization_data` provides normalized DataFrame adapters, but `app.py` builds its current page data and visualizations directly rather than importing that module.

## 6. Database schema and relationships

The schema is declared in `inventory_system/database.py`. Foreign keys are enabled by the database write/setup helpers; the database uses checks and uniqueness constraints for important fields.

| Table | Main columns and constraints | Relationship / role |
|---|---|---|
| `products` | `product_id` primary key; `sku` unique and non-null; `product_name`; `category` | Product master. |
| `inventory` | `product_id` primary key and FK to `products`; `stock_quantity` non-negative; `unit_cost` and `selling_price` non-negative; positive `supplier_lead_time_days`; `updated_at` | One inventory row per product. |
| `sales` | `sale_id` primary key; `product_id` FK to `products`; `sale_date`; non-negative `units_sold`; unique `(product_id, sale_date)` | Daily sales history; an index is created on `sale_date`. |
| `inventory_changes` | `change_id` primary key; `product_id` FK to `products`; previous/new stock; signed quantity change; change type, reason, timestamp | Audit record for stock updates and purchase receipts. |
| `stock_purchases` | `purchase_id` primary key; `product_id` FK to `products`; positive quantity; unit cost, optional supplier, timestamp; `change_id` FK to `inventory_changes` | Purchase-receipt details linked to the inventory change that applied it. |

The product master has one-to-many relationships with sales and inventory-change records. Inventory is one-to-one with products through its primary-key foreign key. A stock purchase references both a product and its corresponding inventory-change record. The current database inspected contained 10 products, 10 inventory rows, and 1,800 sales rows. At inspection time it had no inventory-change or purchase records; test transactions were not left in this database.

## 7. Dataset description and preprocessing

### Synthetic dataset

`inventory_system.data_generation.generate_synthetic_data()` creates:

- 10 product records, with product IDs 1–10, SKU, product name, and category.
- 10 initial inventory records with generated stock quantity, unit cost, selling price, and supplier lead time.
- 180 daily sales rows per product, or 1,800 product-date rows total.
- A default fixed random seed of 42. The default date range ends yesterday and spans 180 consecutive dates.

The inspected CSV and database sales history covered **2026-05-15 through 2026-11-10**. A later rerun of the generator will use a later date range because the default endpoint is relative to the run date.

Daily sales use a product-specific base demand, weekday multipliers, a small positive linear trend, a 45-day sinusoidal seasonal factor, and Poisson sampling. The default weekday multiplier array is `[0.88, 0.91, 0.96, 1.02, 1.12, 1.28, 1.18]` for Monday through Sunday. Each product has a distinct base demand and the same general weekly pattern, with a product-specific seasonal phase. Zero daily units are possible; negative sales are not.

`validate_sales_data()` checks required columns, non-empty data, non-missing product IDs, non-negative units, parseable dates, unique product/date pairs, and—when expected counts are supplied—complete product-by-date coverage. `load_historical_data()` further checks integer-valued IDs/units, known product IDs, duplicate records, and consecutive daily dates for each product. The generator and loader validate but do not perform a separate statistical outlier-removal or missing-value imputation stage.

### Feature preparation

`create_features()` builds features separately for each product:

- `lag_1`, `lag_7`, and `lag_14`
- `rolling_mean_7` and `rolling_mean_14`, calculated from shifted prior sales
- `day_of_week`, `month`, and `product_id`

Rows without enough preceding history for these fields are dropped. Because the rolling windows use shifted sales, the current row's target is not included in its own features. For the inspected 1,800-row dataset, feature creation yielded 1,660 rows: 166 per product after the initial 14 days. The per-product chronological split produced 1,320 training rows and 340 test rows (132 train and 34 test per product with the default 20% setting).

## 8. Machine-learning algorithm and training process

The forecasting task is supervised regression: estimate the next day's non-negative unit demand from recent sales, calendar fields, and product identity.

`scripts/train_model.py` loads and validates the history, generates features, chronologically splits each product's feature rows, trains the pipeline, and evaluates the trained model. Training does not randomly shuffle rows. The model pipeline is serialized using Joblib, and forecasting uses the saved artifact through `load_model()`.

The evaluation is one-step-ahead on chronological holdout rows. For each test date, lag and rolling features are derived from observations preceding that date. These holdout metrics are distinct from the separate multi-day recursive future forecast.

## 9. Random Forest model configuration

`inventory_system.forecasting.build_model()` returns a scikit-learn `Pipeline`:

1. A `ColumnTransformer` passes numeric features through unchanged and applies `OneHotEncoder(handle_unknown="ignore")` to `product_id`.
2. `RandomForestRegressor` uses `n_estimators=200`, `random_state=42`, `n_jobs=-1`, and `min_samples_leaf=2`.

The default model builder rejects fewer than 100 trees. No extra scaling, feature selection, or hyperparameter search is performed by the inspected training script. The product ID is treated as a categorical feature in the pipeline.

## 10. Forecasting methodology, including recursive forecasting

`forecast_demand()` supports a positive integer horizon and can forecast one selected product or all products. Each product requires at least 14 consecutive daily historical rows. For each forecast day it calculates lag and rolling features from that product's current history, adds the forecast date's weekday and month, and calls the saved model.

The predicted value is clipped to a minimum of zero. It is then appended to that product's history and becomes input for later forecast days. Thus, for day two and later, some lag and rolling features contain prior model predictions rather than actual observed sales; this is recursive forecasting. For all-product predictions, the implementation batches products together for each recursive day while maintaining separate product histories.

Forecast dates begin one day after the last available sales date and advance daily. The Streamlit page lets the user choose a product and either 7 or 30 days, displays historical sales beside predicted units, and marks the forecast start. The saved model—not a hand-entered demonstration series—is used to generate the predictions.

## 11. Model evaluation methodology and actual results

`evaluate_models()` compares Random Forest predictions with `rolling_mean_7`, which is the seven-day moving-average baseline already present in the same test feature rows. Both models are evaluated on the same chronological holdout observations. The model's predictions are clipped at zero. The library metrics are:

- **MAE:** mean absolute error, measured in units; lower is better.
- **RMSE:** square root of mean squared error, measured in units and more sensitive to large misses; lower is better.
- **R²:** coefficient of determination; closer to 1 indicates better fit relative to the test-set mean, and a negative result is possible.

The current saved Random Forest model was loaded and evaluated against the current 340 chronological test rows during this inspection:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 5.132230 | 6.596231 | 0.818965 |
| Moving Average (7-day) | 5.933613 | 7.757651 | 0.749602 |

These are results on the synthetic holdout dataset only. They are not evidence of real-world retail accuracy or a guarantee of future performance. The current model-performance page also presents product-wise metrics, actual-versus-predicted charts, and a comparison across the three metrics.

## 12. Inventory management functionality

The Inventory Management page reads product, inventory, and sales data from SQLite; shows catalog, stock, critical-status, and healthy-status summaries; and allows product search and stock-status filtering. It offers:

- Setting an absolute stock count with a reason.
- Applying a signed stock increase or decrease with a reason.
- Recording a received purchase, optionally with a supplier, after an explicit receipt confirmation.
- Viewing the selected product's inventory-change history and any linked purchase information.

Stock values cannot be set below zero. A received purchase increases the stock quantity and records both a `purchase` inventory change and a linked `stock_purchases` row. Recommendations use the latest inventory snapshot when the app refreshes.

## 13. Safety stock, reorder point, and purchase quantity

`generate_restock_recommendations()` validates the inventory inputs and computes a recursive forecast long enough to cover the longest supplier lead time. For each product it uses that product's first `supplier_lead_time_days` predictions.

Let:

- \(L\) = supplier lead time in days
- \(\bar d\) = mean of the product's historical daily sales
- \(s_d\) = sample standard deviation (`ddof=1`) of historical daily sales
- \(F_L\) = sum of recursive predicted demand over the next \(L\) days
- \(S\) = current stock
- \(z\) = safety factor, default 1.65

The implementation calculates:

- **Safety stock:** \(z \times s_d \times \sqrt{L}\)
- **Reorder point:** \(\bar d \times L + \text{safety stock}\)
- **Recommended purchase quantity:** \(\max(0,\lceil F_L + \text{safety stock} - S\rceil)\)

Stock status is **CRITICAL** when current stock is below the reorder point; **LOW STOCK** when it is at or below 1.2 times the reorder point after not being classified critical; and **HEALTHY** when it is above that buffer. Lead time affects the forecast horizon, reorder threshold, and safety-stock calculation. The recommendation is decision support; the app records a purchase only after the user confirms receipt.

## 14. Major Python modules and functions

### `inventory_system/data_generation.py`

- `generate_synthetic_data()` — creates product, initial inventory, and daily sales DataFrames using the seed and date options.
- `validate_sales_data()` — enforces sales quality and optional dimension checks.
- `write_sales_csv()` — validates and writes the sales data, creating parent directories as needed.

### `inventory_system/database.py`

- `initialize_database()` — creates the schema and upserts validated product, inventory, and sales records.
- `refresh_sales()` — inserts missing product rows if necessary and replaces sales while preserving existing inventory rows.
- `update_inventory_stock()` — sets absolute stock and writes an audit change atomically.
- `record_inventory_change()` — applies and audits a signed stock change in a transaction.
- `record_stock_purchase()` — increments stock and inserts linked change and purchase records in one transaction.
- `fetch_products()`, `fetch_inventory()`, and `fetch_sales()` — load their corresponding records into DataFrames.
- `fetch_inventory_history()` — returns stock changes and optional linked purchase fields.
- `fetch_table_counts()` and `fetch_table_columns()` — expose database inspection metadata.
- Private helpers validate seed data and values, ensure schema/foreign-key settings, and apply an inventory change.

### `inventory_system/preprocessing.py`

- `load_historical_data()` — reads the sales CSV, validates values/dates/references, attaches SQLite product details, and checks daily continuity.
- `create_features()` — creates prior-observation lag, rolling-mean, and calendar features.
- `chronological_split()` — separates each product's rows by time and checks that its training dates precede its test dates.

### `inventory_system/forecasting.py`

- `build_model()` — constructs the preprocessing and Random Forest pipeline.
- `train_model()` — splits, fits, saves, and returns the model and train/test records.
- `load_model()` — loads and type-checks the saved scikit-learn pipeline.
- `forecast_demand()` — creates one-product or multi-product recursive forecasts.
- `total_predicted_demand()` — sums the `predicted_units` column.

### `inventory_system/evaluation.py`

- `_metrics()` — computes MAE, RMSE, and R².
- `evaluate_models()` — produces row-level actual/predicted comparisons, overall Random Forest versus moving-average metrics, and per-product metrics.
- `EVALUATION_NOTE` — describes the chronological one-step-ahead evaluation distinction.

### `inventory_system/restocking.py`

- `classify_stock_status()` — classifies stock relative to reorder point and its 20% buffer.
- `generate_restock_recommendations()` — validates inputs, obtains recursive lead-time predictions, calculates stock statistics/formulas, and returns one recommendation row per inventory product.

### `inventory_system/visualization_data.py`

- `historical_sales_data()` and `actual_vs_predicted_data()` — return validated, normalized chart/table DataFrames.
- `next_7_day_forecast()` and `next_30_day_forecast()` — convenience wrappers around `forecast_demand()`.
- `product_evaluation_data()` — returns per-product evaluation metrics.

### `app.py`

- `load_dashboard_data()` — fetches live product, inventory, and sales rows and merges product metadata into history.
- `load_model_for_app()` / `get_model()` — validate and cache the saved model.
- `live_recommendations()` / `_cached_recommendations()` — calculate or reuse recommendations for identical model and data inputs.
- `forecast_for_app()` / `_cached_forecast()` — calculate or reuse forecast results for identical model, data, horizon, and product selection.
- `prepare_evaluation()` — creates features, splits chronologically, loads the saved model, and evaluates it.
- `render_overview()`, `render_inventory_management()`, `render_demand_forecasting()`, `render_restocking()`, `render_sales_analytics()`, `render_model_performance()`, and `render_reports()` — implement the respective views.
- `build_download_reports()` and `_csv_bytes()` — construct report DataFrames from the current data and serialize them for download.
- `render_app()` — sets up styling/sidebar routing, loads current data, and selects the requested page.

### `scripts/`

- `setup_phase1.py` — generates synthetic data and initializes or refreshes the database.
- `train_model.py` — trains, evaluates, and saves the model; accepts optional sales, database, and model paths.
- `forecast_demand.py` — generates a 7- or 30-day forecast for one or all products and saves CSV output.
- `restock_recommendations.py` — generates and saves recommendations, with a configurable z-score.

## 15. Streamlit application pages

Navigation is implemented in `app.py` using a sidebar radio control and retains seven options:

1. **Overview** — current database-driven product, stock, historical-sales, critical-stock, and recommended-purchase KPIs; stock-status and sales visualizations.
2. **Inventory Management** — searchable/filterable inventory, manual stock setting, signed adjustment, received-purchase form, and audit history.
3. **Demand Forecasting** — product and 7-/30-day horizon selectors, historical-versus-predicted chart, forecast start marker, summary values, and forecast table.
4. **Restocking Recommendations** — critical/low/healthy counts, total recommended quantity, filters, product-level recommendations, calculated details, and purchase receipt form.
5. **Sales Analytics** — daily, weekly, and monthly sales views; top/lowest-selling products; category and product comparisons.
6. **Model Performance** — Random Forest and Moving Average metric cards/comparison, selected-product actual/predicted chart, per-product results, and metric explanations.
7. **Reports and Downloads** — CSV exports for inventory, sales, 7-day forecast, 30-day forecast, recommendations, and a summary report.

The application brand in the sidebar is INVENTRA. The page metrics and reports are computed from loaded database records, forecasts, or evaluation outputs rather than fixed demonstration values. The underlying sales records remain synthetic.

## 16. Database operations and transaction handling

The database helpers validate values before writes, enable SQLite foreign-key checks on their write connections, use parameterized SQL, group related writes in a connection transaction, and close opened connections in `finally` blocks.

For purchase receipt, `record_stock_purchase()` reads current stock, calculates the new quantity, calls the shared stock-change helper, inserts a `stock_purchases` row linked to that change, and commits the work as one transaction. If a database error occurs, it is surfaced as a `RuntimeError`; invalid values and missing inventory are reported as `ValueError`. This prevents a successful stock increment from being recorded without its corresponding purchase/audit event.

`initialize_database()` upserts initial inventory values if called directly. The normal setup script avoids that behavior for an existing database by calling `refresh_sales()`, which preserves current inventory and history while replacing sales rows. This distinction matters if documenting how a database reset or setup operation affects stock.

## 17. Testing methodology and actual results

The project uses `unittest` tests in `tests/test_phase1.py` through `tests/test_phase4.py`. Coverage includes:

- Synthetic dataset dimensions, reproducibility, weekly variation, and invalid sales.
- SQLite schema/round-trip behavior, foreign-key constraints, and invalid stock.
- Feature leakage prevention, chronological splitting, model save/load, forecast dates/horizons, batched recursive forecasts, and metric calculations.
- Inventory update persistence, validation, audit history, transactional purchase receipts, stock-status thresholds, lead times, safety-stock formulas, and use of recursive forecasts.
- Application imports/navigation, live database loading, forecast/recommendation availability, report contents, CSV serialization, and recommendation cache freshness after stock changes.

The exact command run during this inspection was:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v
```

**Result:** 34 tests ran in 15.203 seconds; all passed. Streamlit emitted `No runtime found` / `missing ScriptRunContext` warnings in bare unit-test execution; the tests completed successfully. The suite includes tests that exercise database operations using test databases; the project demonstration database was inspected read-only and remained without inventory-change or purchase rows.

This test result does not constitute a separate browser-based visual test or a production concurrency/security test. **Needs confirmation** if the formal report requires screenshots, a live browser demonstration, or a separately recorded manual end-to-end purchase exercise.

## 18. Limitations and future scope

### Verified limitations / boundaries

- Sales and initial inventory are generated synthetic inputs; no real point-of-sale or supplier dataset is connected.
- The dataset contains ten products and 180 days, which is limited evidence for broader retail seasonality or generalization.
- The features are based on product identity, recent sales, weekday, and month. Promotions, holidays, prices, weather, stockouts, and supplier constraints are not model inputs in the inspected implementation.
- Multi-day recursive prediction feeds model outputs back into later predictions, so errors can compound with forecast horizon.
- Evaluation is one chronological holdout on the synthetic dataset. It does not establish real-world accuracy or compare performance across stores or time periods.
- Safety stock uses a fixed default z-score of 1.65 and a historical standard deviation formula; the code does not estimate a service level from actual retailer policy.
- The interface is a local Streamlit app backed by a local SQLite database. Authentication, multi-user access control, cloud deployment, and automated supplier transmission are not implemented.

### Potential future work (not implemented)

- Replace synthetic records with validated point-of-sale, inventory, supplier, and promotion data.
- Evaluate with rolling-origin backtesting, multiple product/store groups, and operational service-level measures.
- Consider promotions, holidays, prices, stockouts, product lifecycle, and external demand drivers.
- Add uncertainty intervals, forecast monitoring, drift checks, and a controlled retraining process.
- Incorporate supplier minimum order quantities, pack sizes, budgets, and actual ordering constraints.
- Consider multi-location support, access control, backups, and deployment only after requirements and security review.

## Items to supply or confirm manually for the formal college report

- Student name(s), roll number(s), registration number(s), degree/program, year, semester, and department.
- College/university name, address, logo usage requirements, and academic year.
- Project guide/supervisor name, designation, department, and any co-guide details.
- Submission date and any required certificate, declaration, acknowledgement, abstract, or formatting template.
- Whether the college expects a named organization/current-system comparison; if so, provide the real workflow and evidence.
- Whether the college requires browser screenshots or a live demonstration record. The 34-test result above is automated test evidence, not a claim that browser testing was performed.
