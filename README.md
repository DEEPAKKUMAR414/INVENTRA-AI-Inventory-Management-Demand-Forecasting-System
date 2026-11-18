# INVENTRA

AI-based intelligent inventory management and demand forecasting system.

## Phase 1: data foundation

Phase 1 creates reproducible synthetic retail data and a SQLite database.
The default 180-day dataset ends yesterday, using seed 42. Rerunning the data
setup refreshes sales data without resetting existing product or inventory
records.

## Phase 2: demand forecasting engine

Phase 2 adds a global Random Forest model for demand prediction without paid
APIs.

### Setup and run

From the project folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe .\scripts\setup_phase1.py
```

The script writes `data\synthetic_sales.csv` and creates
`database\inventory.db`.

Run Phase 1 tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The default dataset uses seed 42 and covers 180 daily dates ending yesterday
for 10 products. Sales vary by product, weekday, trend, seasonal cycle, and
random sampling. Inventory and product attributes are stored in the SQLite
database.

### Feature engineering

`inventory_system.preprocessing` reads historical daily sales from
`data\synthetic_sales.csv` and product descriptions from SQLite, validates
records, joins product names, and sorts by product and date. It creates
`lag_1`, `lag_7`, `lag_14`, seven- and fourteen-day rolling means,
`day_of_week`, `month`, and `product_id`. Lags shift the target before reading
prior observations; rolling features use the previous seven or fourteen
observations and exclude the current target date. The first 14 rows per
product are omitted because their full history features are unavailable.

### Random Forest architecture and training

`inventory_system.forecasting` uses a scikit-learn `Pipeline` containing a
`ColumnTransformer`, `OneHotEncoder` for categorical `product_id`, and a
`RandomForestRegressor` with 200 trees and `random_state=42`. One global model
learns across all products. The fitted pipeline is persisted with Joblib at
`models\random_forest_demand.joblib`.

Train and evaluate the model from the project directory:

```powershell
.\.venv\Scripts\python.exe .\scripts\train_model.py
```

### Chronological validation and evaluation

The split is chronological within each product: earlier records are used to
fit the model and each product's latest 20% of feature rows are held out.
There is no random train/test split. For every held-out date, the lag and
rolling features refer only to sales dated before that test date, giving
one-step-ahead predictions. The seven-day moving-average baseline uses the
same test rows and their trailing-seven-day mean feature.

The evaluation reports:

- **MAE**: average absolute error in demand units; lower is better.
- **RMSE**: square root of average squared error, which penalizes larger
  misses more strongly; lower is better.
- **R²**: proportion of target variance explained relative to predicting the
  test-set mean; values closer to 1 are better, and a negative value can be
  worse than that mean baseline.

Evaluation also returns actual-versus-predicted records with residuals and
overall and per-product metrics for the Random Forest and Moving Average.
Evaluation uses one-step-ahead predictions based only on information
available before each test date. Future recursive forecasting is implemented
separately.

### Recursive future forecasts

Forecasting is performed recursively: after each forecast step, that
predicted demand is appended to the product's history and supplies lag and
rolling inputs for the next date. No actual future sales are read. Predictions
are constrained to non-negative values, and forecast rows include product
name, date, and predicted units.

Generate and print the next seven days for all products (or pass
`--product-id 3` for one product):

```powershell
.\.venv\Scripts\python.exe .\scripts\forecast_demand.py --days 7
```

Generate a 30-day forecast:

```powershell
.\.venv\Scripts\python.exe .\scripts\forecast_demand.py --days 30
```

Both commands write model-derived rows to `data\forecast_7_day.csv` or
`data\forecast_30_day.csv` respectively. Use `--output PATH` to select another
destination. DataFrame adapters for historical sales, model evaluation, and
7-/30-day forecast data are available in `inventory_system.visualization_data`
for a future dashboard.

Run all Phase 1 and Phase 2 tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v
```

## Phase 3: intelligent inventory restocking

`inventory_system.restocking` uses recursively generated Random Forest demand
forecasts for each product's supplier lead time. It does not create fixed or
random recommendations. Historical daily demand is used to calculate the
average and sample standard deviation, while the forecast model supplies
predicted demand during the lead time.

With the default service-level assumption **Z = 1.65**, the calculations are:

- **Safety stock** = `Z × daily demand standard deviation × √lead time`
- **Reorder point** = `average daily demand × lead time + safety stock`
- **Recommended purchase quantity** =
  `ceil(max(0, forecast lead-time demand + safety stock - current stock))`

Purchase quantities are rounded up to whole units. For example, with a
10-day supplier lead time, average demand of 12 units/day, demand standard
deviation of 3 units/day, forecast lead-time demand of 125 units, and 100 units
in stock, safety stock is `1.65 × 3 × √10 ≈ 15.65`, the reorder point is
`12 × 10 + 15.65 = 135.65`, and the recommended order is
`ceil(max(0, 125 + 15.65 - 100)) = 41` units.

Stock status is derived from calculated stock and reorder point for every
product:

- **CRITICAL**: current stock is below the reorder point.
- **LOW STOCK**: current stock is at or above the reorder point and no more
  than 20% above it.
- **HEALTHY**: current stock is more than 20% above the reorder point.

If a model forecast does not cover the complete supplier lead time, the
recommendation call raises an explicit error rather than substituting a made-up
demand value. Inventory updates, signed adjustments, and received purchases
are recorded transactionally in SQLite. `inventory_changes` stores stock
before/after values and a reason; `stock_purchases` stores received purchase
quantities, cost, supplier, and their linked change. The existing inventory
stock remains persisted in the database across process restarts.

### Complete Phase 3 workflow

Run these commands from the project directory in PowerShell. Data refresh
updates the sales table and preserves current stock; training regenerates the
Joblib model, forecast commands update their CSVs, and the final command
produces recommendations from the stored model and inventory:

```powershell
.\.venv\Scripts\python.exe .\scripts\setup_phase1.py
.\.venv\Scripts\python.exe .\scripts\train_model.py
.\.venv\Scripts\python.exe .\scripts\forecast_demand.py --days 7
.\.venv\Scripts\python.exe .\scripts\forecast_demand.py --days 30
.\.venv\Scripts\python.exe .\scripts\restock_recommendations.py
.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v
```

The recommendation report is written to
`data\restock_recommendations.csv`. The database helper functions for
persisting stock updates and retrieving inventory history are in
`inventory_system.database`.

## Faculty Demonstration Guide

Suggested 6-minute demonstration. Start the app before the presentation with
the launch command at the end of this guide.

1. **Introduce the problem (30 seconds).** Explain that manual stock tracking
   can miss demand changes and supplier lead times, causing stockouts or
   excess inventory. This project combines sales history, machine learning,
   and reorder calculations to support decisions.
2. **Show the inventory dashboard (45 seconds).** Open **Overview Dashboard**.
   Point out product count, current stock, database sales totals, calculated
   stock-status counts, sales trend, and products below reorder point. Explain
   that the recommendations are recalculated from the current stock and saved
   model.
3. **Explore one product's sales (45 seconds).** Open **Demand Forecasting**,
   choose a product, and identify its historical sales trace. Note the last
   historical date and that forecast dates start the following day.
4. **Compare forecast horizons (60 seconds).** Select **7 days**, explain the
   daily predictions and total, then select **30 days** and explain that
   predictions are recursive: earlier predictions become inputs for later
   forecast days.
5. **Explain restocking (60 seconds).** Open **Restocking Recommendations**,
   select a product, and walk through predicted lead-time demand, safety stock
   (`Z = 1.65`), reorder point, current stock, and whole-unit suggested order.
   Briefly explain the CRITICAL, LOW STOCK, and HEALTHY thresholds.
6. **Demonstrate inventory receiving (60 seconds).** On the same page, enter a
   small controlled quantity as already received, check the confirmation box,
   and submit. Show that stock and the purchase/history record update, and
   that recommendations refresh using the new stock. For a disposable demo
   database, restore the original quantity after the presentation if needed.
7. **Show model evaluation (60 seconds).** Open **Model Performance**. Compare
   the actual Random Forest metrics with the Moving Average baseline. Mention
   that evaluation is one-step-ahead on later chronological test rows, unlike
   future recursive forecasting.
8. **Download a report (30 seconds).** Open **Reports and Downloads** and
   download the inventory or recommendation CSV.

### Start the demonstration

From the project directory in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Phase 4: Streamlit application

The application entry point is `app.py`. It provides sidebar navigation for
the overview, inventory management, demand forecasts, restocking
recommendations, sales analytics, model performance, and CSV reports. Charts
are interactive Plotly visualizations with a premium dark glassmorphism
interface: midnight-blue surfaces, restrained purple/cyan gradients,
translucent cards, readable KPI panels, responsive layouts, and consistent
dark charts. The dashboard is branded **INVENTRA** and includes an
original local SVG inventory-and-forecast illustration in a glass card.
Inventory, sales, and purchase history
are read from SQLite; forecasts and recommendations are calculated through
the persisted Random Forest and the existing Phase 2/3 functions. The model is
cached, and completed forecasts/recommendations are reused for identical
inputs. Inventory is fetched on every rerun, so a stock change creates a fresh
recommendation result instead of showing stale stock data.

### Installation and application startup

Create the project virtual environment (if it does not already exist) and
install dependencies into it:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

Or activate the existing environment in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Initialize or refresh the database sales records while preserving existing
stock, train the model, and start the application:

```powershell
.\.venv\Scripts\python.exe .\scripts\setup_phase1.py
.\.venv\Scripts\python.exe .\scripts\train_model.py
.\.venv\Scripts\python.exe -m streamlit run .\app.py
```

The dashboard displays setup instructions if the database, sales CSV, or
trained model is missing. To generate stand-alone forecast and restocking CSV
files before opening the dashboard, run:

```powershell
.\.venv\Scripts\python.exe .\scripts\forecast_demand.py --days 7
.\.venv\Scripts\python.exe .\scripts\forecast_demand.py --days 30
.\.venv\Scripts\python.exe .\scripts\restock_recommendations.py
```

### Project architecture

```text
app.py                              Streamlit navigation, pages, charts, exports
inventory_system/
  data_generation.py                Reproducible retail data generation
  database.py                       SQLite reads and transactional inventory writes
  preprocessing.py                  Validation and leakage-safe feature engineering
  forecasting.py                    Random Forest pipeline and recursive forecasting
  evaluation.py                     Chronological model and baseline metrics
  restocking.py                     Forecast-driven inventory recommendations
  visualization_data.py             Dashboard-ready DataFrame helpers
scripts/                            Setup, training, forecasting, restocking CLIs
tests/                              Phase 1–4 automated tests
data/                               Generated sales, forecasts, and reports
database/inventory.db               Persistent SQLite application data
models/                              Persisted Joblib Random Forest model
```

### Dashboard screenshots

Screenshots can be added here after capturing the application:

- Overview dashboard: _screenshot placeholder_
- Restocking recommendations: _screenshot placeholder_
- Forecasting and analytics: _screenshot placeholder_

Run all project tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v
```
