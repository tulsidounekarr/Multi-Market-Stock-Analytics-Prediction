
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("data/tcs_stock_data.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Features use information available by the current trading day
df["Return_1d"] = df["Close"].pct_change()
df["Return_5d"] = df["Close"].pct_change(5)
df["MA_Ratio_5"] = df["Close"] / df["Close"].rolling(5).mean() - 1
df["MA_Ratio_20"] = df["Close"] / df["Close"].rolling(20).mean() - 1
df["Volatility_5d"] = df["Return_1d"].rolling(5).std()
df["Volume_Change"] = df["Volume"].pct_change()

# Next trading day's percentage return
df["Target_Return"] = df["Close"].shift(-1) / df["Close"] - 1

features = [
    "Return_1d", "Return_5d", "MA_Ratio_5",
    "MA_Ratio_20", "Volatility_5d", "Volume_Change"
]

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=features + ["Target_Return"]).reset_index(drop=True)

X = df[features]
y = df["Target_Return"]

split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

predicted_returns = model.predict(X_test)

# Convert predicted returns back to next-day prices
current_close = df.loc[X_test.index, "Close"].to_numpy()
actual_next_close = current_close * (1 + y_test.to_numpy())
predicted_next_close = current_close * (1 + predicted_returns)

mae = mean_absolute_error(actual_next_close, predicted_next_close)
rmse = np.sqrt(mean_squared_error(actual_next_close, predicted_next_close))

# Persistence baseline: tomorrow's price equals today's price
baseline_mae = mean_absolute_error(actual_next_close, current_close)

print("RETURN-BASED MODEL")
print("-" * 35)
print(f"Test records: {len(X_test)}")
print(f"MAE: ₹{mae:.2f}")
print(f"RMSE: ₹{rmse:.2f}")
print(f"Baseline MAE: ₹{baseline_mae:.2f}")

Path("models").mkdir(exist_ok=True)
joblib.dump(
    {"model": model, "features": features},
    "models/tcs_return_model.joblib"
)

results = pd.DataFrame({
    "Date": df.loc[X_test.index, "Date"].values,
    "Actual_Close": actual_next_close,
    "Predicted_Close": predicted_next_close,
    "Baseline_Close": current_close
})
results.to_csv("outputs/return_predictions.csv", index=False)

print("\nSaved model to models/tcs_return_model.joblib")
print("Saved predictions to outputs/return_predictions.csv")
