# Multi-Market LSTM Stock Trend Prediction — GitHub Screenshots

Copy the `screenshots/` folder into the root of your repository.

## Screenshot guide

1. `01_historical_price_chart.png` — Historical TCS closing-price chart.
2. `02_supported_stocks.png` — Stock selector showing Indian and US stock options.
3. `03_actual_vs_lstm_baseline.png` — Actual closing prices compared with LSTM predictions and the naive baseline.
4. `04_model_evaluation.png` — LSTM evaluation metrics and comparison with the naive baseline.
5. `05_apple_us_dashboard.png` — Apple (USA) dashboard showing latest close, last-session change, and historical closing prices.

## README snippet

Paste this into your main `README.md`:

```markdown
## Dashboard Screenshots

### Historical Prices
![Historical price chart](screenshots/01_historical_price_chart.png)

### Supported Stocks
![Supported stocks](screenshots/02_supported_stocks.png)

### Model Evaluation
![Model evaluation](screenshots/04_model_evaluation.png)

### Actual vs Predicted
![Actual vs LSTM vs baseline](screenshots/03_actual_vs_lstm_baseline.png)

### Apple (USA) Dashboard
![Apple stock dashboard](screenshots/05_apple_us_dashboard.png)
```

## Notes

- The dashboard is an educational forecasting demo, not financial advice.
- In the recorded TCS and AAPL tests, the LSTM's MAE was higher than the naive baseline's MAE. Report this honestly; the current model has not demonstrated better accuracy than the baseline.
- Before presenting the app as polished, fix the raw Streamlit API-method text that appeared below the evaluation section in an earlier screenshot.
