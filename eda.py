
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load our downloaded dataset
df = pd.read_csv("data/tcs_stock_data.csv", parse_dates=["Date"])
df = df.sort_values("Date")

print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistical summary:")
print(df[["Open", "High", "Low", "Close", "Volume"]].describe())

# Calculate daily percentage returns
df["Daily_Return_%"] = df["Close"].pct_change() * 100

# Create folder for charts
Path("outputs").mkdir(exist_ok=True)

# Chart 1: Closing price over time
plt.figure(figsize=(12, 5))
plt.plot(df["Date"], df["Close"])
plt.title("TCS Historical Closing Price")
plt.xlabel("Date")
plt.ylabel("Adjusted Close Price (INR)")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/tcs_closing_price.png")
plt.show()

# Chart 2: Trading volume
plt.figure(figsize=(12, 5))
plt.plot(df["Date"], df["Volume"])
plt.title("TCS Trading Volume")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/tcs_trading_volume.png")
plt.show()

# Chart 3: Daily returns
plt.figure(figsize=(12, 5))
plt.plot(df["Date"], df["Daily_Return_%"])
plt.title("TCS Daily Percentage Returns")
plt.xlabel("Date")
plt.ylabel("Daily Return (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/tcs_daily_returns.png")
plt.show()

print("\nEDA completed! Charts saved in the outputs folder.")
