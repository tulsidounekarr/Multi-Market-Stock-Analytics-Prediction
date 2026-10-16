# Multi-Market LSTM Stock Trend Prediction

Educational LSTM forecasting project for Indian and US stocks. It downloads adjusted daily prices, trains an LSTM on the previous 60 trading days, and compares results with a naive baseline (next close = previous close).

**Not financial advice.** Stock prices are noisy, and LSTM can underperform simple baselines.

## Stocks
- India: TCS.NS, RELIANCE.NS, INFY.NS
- US: AAPL, MSFT, NVDA

## Install (Windows PowerShell)
```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lstm.txt
```

Back up the current `train_lstm.py` before replacing it with this package's version. The dashboard is named `app_lstm.py` to preserve the existing `app.py`.

## Train
```powershell
python train_lstm.py --ticker TCS.NS
python train_lstm.py --ticker AAPL
python train_lstm.py --all
```
Start with one stock; training all six takes longer.

## Dashboard
```powershell
python -m streamlit run app_lstm.py
```

## Outputs
- `data/lstm/<ticker>.csv`
- `models/lstm/<ticker>_lstm.keras`
- `outputs/lstm/<ticker>_predictions.csv`
- `outputs/lstm/<ticker>_metrics.json`

The scaler is fitted only on training-period prices. Evaluation reports MAE, RMSE and R² and compares MAE with a naive baseline. If LSTM MAE is higher than baseline MAE, it did not outperform the baseline on the held-out period.
