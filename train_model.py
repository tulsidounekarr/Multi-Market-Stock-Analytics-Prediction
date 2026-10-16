
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load historical stock data
df = pd.read_csv("data/tcs_stock_data.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Create features using only current and past information
df["Return_1d"] = df["Close"].pct_change()
df["Return_5d"] = df["Close"].pct_change(5)
df["MA_5"] = df["Close"].rolling(5).mean()
df["MA_20"] = df["Close"].rolling(20).mean()
df["Volatility_5d"] = df["Return_1d"].rolling(5).std()

# Target: next trading day's closing price
df["Target"] = df["Close"].shift(-1)

features = [
    "Close", "Volume", "Return_1d", "Return_5d",
    "MA_5", "MA_20", "Volatility_5d"
]

df = df.dropna().reset_index(drop=True)

X = df[features]
y = df["Target"]

# Chronological split: older data for training, newer data for testing
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Train the model
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate on unseen, later dates
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("TCS NEXT-DAY PRICE MODEL")
print("-" * 35)
print(f"Training records: {len(X_train)}")
print(f"Testing records:  {len(X_test)}")
print(f"MAE:  ₹{mae:.2f}")
print(f"RMSE: ₹{rmse:.2f}")
print(f"R²:   {r2:.4f}")

# Save model and feature information
Path("models").mkdir(exist_ok=True)
joblib.dump(
    {"model": model, "features": features},
    "models/tcs_random_forest.joblib"
)

# Save actual and predicted prices for plotting
results = pd.DataFrame({
    "Date": df.loc[X_test.index, "Date"].values,
    "Actual_Close": y_test.values,
    "Predicted_Close": predictions
})
Path("outputs").mkdir(exist_ok=True)
results.to_csv("outputs/predictions.csv", index=False)

# Compare with a simple baseline: tomorrow's price = today's price
baseline = df.loc[X_test.index, "Close"].to_numpy()
baseline_mae = mean_absolute_error(y_test, baseline)
print(f"Baseline MAE (today's close): ₹{baseline_mae:.2f}")

print("\nModel saved to models/tcs_random_forest.joblib")
print("Predictions saved to outputs/predictions.csv")
