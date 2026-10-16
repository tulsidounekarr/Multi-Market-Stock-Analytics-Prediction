"""Multi-market LSTM stock prediction. Run: python train_lstm.py --ticker TCS.NS / --ticker AAPL / --all"""
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA_DIR, MODEL_DIR, OUTPUT_DIR = ROOT/"data/lstm", ROOT/"models/lstm", ROOT/"outputs/lstm"
STOCKS = {"TCS (India)":"TCS.NS", "Reliance (India)":"RELIANCE.NS", "Infosys (India)":"INFY.NS",
          "Apple (USA)":"AAPL", "Microsoft (USA)":"MSFT", "NVIDIA (USA)":"NVDA"}
LOOKBACK, TRAIN_RATIO, EPOCHS, BATCH_SIZE, SEED = 60, 0.80, 20, 32, 42

def safe_name(ticker):
    return re.sub(r"[^A-Za-z0-9_-]", "_", ticker)

def download_stock_data(ticker, period="5y"):
    print(f"\nDownloading {ticker} ({period})...")
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker and internet.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        raise ValueError(f"'Close' column missing for {ticker}.")
    df = df[["Close"]].dropna()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR/f"{safe_name(ticker)}.csv", index_label="Date")
    print(f"Saved {len(df)} rows to data/lstm/{safe_name(ticker)}.csv")
    return df

def prepare_sequences(df):
    prices = df[["Close"]].values.astype(np.float32)
    split_row = int(len(prices)*TRAIN_RATIO)
    if split_row <= LOOKBACK or len(prices)-split_row < 2:
        raise ValueError("Not enough data to create train/test sequences.")
    scaler = MinMaxScaler()
    scaler.fit(prices[:split_row])  # fit only on training data
    scaled = scaler.transform(prices)
    X, y, target_rows = [], [], []
    for i in range(LOOKBACK, len(scaled)):
        X.append(scaled[i-LOOKBACK:i, 0]); y.append(scaled[i, 0]); target_rows.append(i)
    X = np.asarray(X, dtype=np.float32).reshape(-1, LOOKBACK, 1)
    y, target_rows = np.asarray(y, dtype=np.float32), np.asarray(target_rows)
    train_mask, test_mask = target_rows < split_row, target_rows >= split_row
    return X[train_mask], X[test_mask], y[train_mask], y[test_mask], scaler, prices[target_rows[test_mask],0], df.index[target_rows[test_mask]]

def build_model():
    tf.keras.utils.set_random_seed(SEED)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(LOOKBACK,1)),
        tf.keras.layers.LSTM(50, return_sequences=True), tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50), tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(25, activation="relu"), tf.keras.layers.Dense(1)])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def train_one(ticker, company=None):
    df = download_stock_data(ticker)
    Xtr, Xte, ytr, yte, scaler, actual, dates = prepare_sequences(df)
    print(f"Training samples: {len(Xtr)} | Test samples: {len(Xte)}")
    model = build_model()
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    hist = model.fit(Xtr, ytr, validation_split=0.1, epochs=EPOCHS, batch_size=BATCH_SIZE,
                     shuffle=False, callbacks=[stop], verbose=1)
    pred_scaled = model.predict(Xte, verbose=0)
    pred = scaler.inverse_transform(pred_scaled).ravel()
    baseline = scaler.inverse_transform(Xte[:,-1,0].reshape(-1,1)).ravel()
    mae = float(mean_absolute_error(actual,pred))
    rmse = float(np.sqrt(mean_squared_error(actual,pred)))
    r2 = float(r2_score(actual,pred)) if len(actual)>1 else float("nan")
    bmae = float(mean_absolute_error(actual,baseline))
    brmse = float(np.sqrt(mean_squared_error(actual,baseline)))
    safe = safe_name(ticker)
    MODEL_DIR.mkdir(parents=True,exist_ok=True); OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    model.save(MODEL_DIR/f"{safe}_lstm.keras")
    pd.DataFrame({"Date":dates,"Actual_Close":actual,"LSTM_Predicted_Close":pred,
                  "Naive_Baseline_Close":baseline}).to_csv(OUTPUT_DIR/f"{safe}_predictions.csv",index=False)
    metrics = {"company":company or ticker,"ticker":ticker,"rows":len(df),"train_samples":len(Xtr),
               "test_samples":len(Xte),"mae":mae,"rmse":rmse,"r2":r2,"baseline_mae":bmae,
               "baseline_rmse":brmse,"epochs_completed":len(hist.history.get("loss",[]))}
    (OUTPUT_DIR/f"{safe}_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(f"\nResults for {ticker}: LSTM MAE={mae:.4f}; RMSE={rmse:.4f}; R²={r2:.4f}")
    print(f"Naive baseline MAE={bmae:.4f}; RMSE={brmse:.4f}")
    print(f"Saved model and results under models/lstm and outputs/lstm.")
    return metrics

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ticker", help="Yahoo ticker, e.g. TCS.NS or AAPL")
    p.add_argument("--all", action="store_true", help="Train all six example stocks")
    args=p.parse_args()
    chosen=list(STOCKS.items()) if args.all else [(next((n for n,s in STOCKS.items() if s==(args.ticker or "TCS.NS")),args.ticker or "TCS.NS"),args.ticker or "TCS.NS")]
    results=[]
    for name,ticker in chosen:
        try: results.append(train_one(ticker,name))
        except Exception as e: print(f"\nERROR for {ticker}: {type(e).__name__}: {e}")
    if results:
        OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
        (OUTPUT_DIR/"all_metrics.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
        print("\nFinished. Compare LSTM MAE with baseline MAE; lower is better.")
    else: raise SystemExit("No models trained successfully.")
if __name__=="__main__": main()
