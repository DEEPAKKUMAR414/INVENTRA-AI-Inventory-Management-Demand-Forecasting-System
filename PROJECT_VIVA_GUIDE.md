# Project Viva Guide

## Project in simple terms

The project helps a retailer decide what stock to keep and when to order more.
It uses recorded daily sales to estimate future demand, accounts for how long a
supplier takes to deliver, and compares projected needs with current inventory.

## Likely viva questions and answers

### 1. What problem does your project solve?

It supports inventory decisions that are often made manually. It shows current
stock, forecasts near-future demand, and calculates when and how much to
reorder. The goal is to reduce stockout risk while avoiding unnecessary excess
stock.

### 2. What does intelligent inventory management mean?

It means using inventory, sales, and supplier information to support decisions
instead of relying only on fixed guesses. In this project, the system combines
current stock, historical demand, a machine-learning forecast, and lead time to
calculate recommendations.

### 3. Why is demand forecasting needed?

An order placed today may arrive only after several days. Forecasting estimates
what customers may buy during that wait so the business can plan before stock
runs out.

### 4. Why did you choose Random Forest?

Random Forest combines many decision trees and can learn nonlinear patterns
from recent sales, weekday, month, and product identity. It works for a
multi-product regression problem and does not require the project to call an
external AI service. The model is reproducible with a fixed random seed.

### 5. What is the Moving Average baseline?

It predicts the next day's demand using the average of the previous seven
days. It is a simple reference point: the more complex Random Forest should be
compared with a model that is easy to understand.

### 6. What does MAE mean?

Mean Absolute Error is the average size of prediction errors, ignoring whether
the prediction was too high or too low. It is measured in units sold, so an
MAE of about 5.13 means predictions miss actual demand by around 5.13 units on
average across the evaluated records.

### 7. What does RMSE mean?

Root Mean Squared Error squares errors before averaging and then takes the
square root. It is also measured in demand units, but it penalizes large
mistakes more than MAE.

### 8. What does R² mean?

R² measures how much variation the model explains compared with predicting the
overall test-set average. A value closer to 1 is generally better. A negative
value is possible when a model performs worse than that simple mean
prediction.

### 9. Why is the train-test split chronological?

In real forecasting, the model learns from the past and predicts the future.
Randomly mixing dates could let future periods influence training, making
evaluation unrealistically optimistic. This project holds out the latest
records separately for each product.

### 10. What is recursive forecasting?

The model predicts the first future day. That prediction is then added to the
product's history and used to form lag and rolling features for the next day.
The process repeats for each requested future date, without reading actual
future sales.

### 11. How is safety stock calculated?

The project uses `Safety Stock = Z × demand standard deviation × √lead time`.
The default Z value is 1.65. Safety stock is a buffer for variation in daily
demand while a supplier order is in transit.

### 12. What is the reorder point?

The reorder point is a stock threshold: `average daily demand × supplier lead
time + safety stock`. If available stock falls below this estimated need, the
product is marked CRITICAL.

### 13. How is recommended purchase quantity calculated?

The system sums recursive ML predictions across the supplier's lead-time days,
adds safety stock, subtracts current stock, and floors the result at zero. It
rounds a positive quantity up to a whole unit so a fractional unit is not
under-ordered.

### 14. Why does supplier lead time matter?

A longer delivery time means the business must cover more days of expected
demand. Lead time therefore changes forecast coverage, safety stock, and the
reorder point for each product.

### 15. How is SQLite used, and why is the data synthetic?

SQLite stores products, inventory, daily sales, stock changes, and received
purchases in a local relational database. It is suitable for a self-contained
college project and persists data across application restarts. Synthetic
sales provide a reproducible dataset for development and demonstrations
without exposing real customer or business records. The current data is
synthetic, so real-world forecast accuracy has not been established.

## Current evaluation results

The latest evaluation uses 340 chronological one-step-ahead test rows:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 5.1322 | 6.5962 | 0.8190 |
| 7-day Moving Average | 5.9336 | 7.7577 | 0.7496 |

These are results on the synthetic dataset, not a guarantee of real-world
performance.

## Limitations and future improvements

- Sales are synthetic and only cover ten products and 180 days.
- The evaluation is one-step-ahead and may not represent the error accumulated
  over a long recursive horizon.
- The model does not currently use promotions, price changes, holidays,
  weather, supplier delays, or stockout-censored demand.
- The default safety-stock service assumption is a fixed Z value rather than
  an empirically calibrated service level.
- Before business use, the system needs validated real sales and inventory
  data, monitoring, retraining procedures, and domain review.
- Future improvements could include real-data ingestion, promotion and
  holiday features, forecast uncertainty, configurable service levels,
  backtesting across multiple time windows, and role-based workflows.
