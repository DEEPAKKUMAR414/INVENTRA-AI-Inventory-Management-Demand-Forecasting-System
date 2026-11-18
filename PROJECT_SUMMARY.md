# AI-Based Intelligent Inventory Management and Demand Forecasting System

## Problem statement

Retailers need enough stock to meet customer demand, but excess inventory ties
up capital and insufficient inventory risks stockouts. Manual estimates may
not account consistently for recent sales patterns or supplier delivery
times. This project combines a demand forecast and inventory calculations to
provide an auditable restocking decision aid.

## Objectives

- Store products, inventory, daily sales, inventory changes, and received
  purchases in SQLite.
- Create reproducible multi-product daily sales data for development.
- Forecast product demand for 7 and 30 future days with a machine-learning
  model.
- Evaluate the model against a Moving Average baseline using chronological
  holdout data.
- Calculate supplier lead-time demand, safety stock, reorder points, and
  recommended purchase quantities.
- Provide an interactive Streamlit application for dashboards, forecasts,
  inventory workflows, analytics, model performance, and CSV reports.

## Technology stack

- Python 3.10
- Streamlit 1.65.0
- Pandas 2.3.3 and NumPy 2.2.6
- Scikit-learn 1.7.2
- Plotly 7.1.0 and Matplotlib 3.10.9
- SQLite
- Joblib 1.6.0

## System architecture

1. **Data generation and storage:** a seeded generator writes synthetic daily
   product sales to CSV and SQLite. Products and current stock are retained
   when sales are refreshed.
2. **Preprocessing:** sales are validated, joined with product names, sorted by
   product/date, and converted into lags, trailing rolling means, weekday,
   month, and product ID features.
3. **Model training:** a scikit-learn Pipeline one-hot encodes product ID and
   fits one global Random Forest regressor. The persisted model is stored as a
   Joblib artifact.
4. **Evaluation and forecasting:** chronological per-product holdout data is
   used for one-step-ahead evaluation. Separate recursive forecasts predict
   future 7- and 30-day demand.
5. **Restocking:** recursive forecasts are aggregated over each supplier's
   lead time; results are combined with historical demand variation and live
   stock to calculate recommendations.
6. **Application:** Streamlit reads live inventory and sales data from
   SQLite, displays Plotly charts, invokes the saved model and existing
   calculation modules, and provides transactional inventory/purchase actions
   and downloads.

## Dataset description

The current dataset is **synthetic**, generated with a fixed random seed for
ten retail products and 180 consecutive daily records per product. The
demonstration dataset currently runs from **2026-05-15 through 2026-11-10**
(system date at its last refresh), for 1,800 product-date sales rows. Sales are
non-negative and include product-level demand levels, weekly variations,
trend, a seasonal component, and seeded random sampling. The database also
stores product category, SKU, stock quantity, unit cost, selling price, and
supplier lead time.

The synthetic data makes development and demonstrations reproducible and does
not represent observed customer or retailer behavior.

## Machine-learning methodology

One global `RandomForestRegressor` learns across product records. A
`ColumnTransformer` passes numeric lag/rolling/calendar variables through
and applies `OneHotEncoder` to product ID. The forest uses 200 trees and
`random_state=42`.

Features include sales lags at 1, 7, and 14 days; trailing 7- and 14-day
averages; day of week; month; and product ID. Rolling features are shifted so
they use observations prior to the target date.

The latest 20% of feature rows for each product are held out chronologically.
Random train-test splitting is not used. Evaluation metrics are calculated
from one-step-ahead predictions: each test date's features use sales available
before that date. Recursive future forecasting is a separate procedure: each
predicted future day becomes input history for later predicted days.

## Actual evaluation results

Metrics below were recomputed from the current saved model and the current
chronological test set of 340 rows:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 5.132230 | 6.596231 | 0.818965 |
| Moving Average (7-day) | 5.933613 | 7.757651 | 0.749602 |

On this synthetic chronological holdout, Random Forest has lower MAE and RMSE
and higher R² than the seven-day Moving Average baseline. These figures do not
establish real-world accuracy.

## Key features

- Product and inventory data management.
- Historical daily, weekly, and monthly sales analytics.
- Product-specific 7- and 30-day recursive demand forecasts.
- Model evaluation and comparison with a Moving Average baseline.
- Lead-time-aware safety stock, reorder point, and purchase suggestions.
- CRITICAL, LOW STOCK, and HEALTHY stock classifications.
- Persistent stock adjustment and received-purchase audit history in SQLite.
- Streamlit dashboard, product filters, Plotly charts, and CSV downloads.

## Limitations

- The sales data is synthetic; real-world accuracy has not yet been
  established.
- There are only ten products and six months of daily history.
- One-step-ahead metrics do not measure error accumulation in multi-step
  recursive forecasts.
- Current features do not model promotions, changing prices, holidays,
  supplier delays, or unobserved demand during stockouts.
- Safety stock uses a fixed default Z value of 1.65 and assumes demand
  variation can be summarized by historical sample standard deviation.
- Recommendations are decision support and should be reviewed before
  purchasing.

## Future scope

- Integrate validated real point-of-sale and supplier data.
- Add promotion, holiday, price, and stockout-aware demand features.
- Compare additional forecasting approaches using rolling-origin backtests.
- Estimate forecast uncertainty and tune service-level assumptions.
- Monitor model drift and define a controlled retraining process.
- Add operational user roles and deployment only after security and data
  governance requirements are defined.

## Conclusion

The project demonstrates an end-to-end inventory analytics workflow: local
data storage, leakage-aware feature engineering, global Random Forest
forecasting, chronological model evaluation, recursive future demand
prediction, and transactional restocking workflows. Its current results are
for reproducible synthetic data only; validation with real operational data is
required before relying on its recommendations in a business.
