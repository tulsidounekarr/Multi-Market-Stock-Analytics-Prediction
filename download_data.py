
import yfinance as yf
import pandas as pd
from pathlib import Path

# TCS stock symbol on the Indian stock market
ticker = "TCS.NS"

print("Downloading TCS historical stock data...")

stock = yf.Ticker(ticker)
df = stock.history(period="5y", auto_adjust=True)

if df.empty:
    raise RuntimeError(
        "No stock data downloaded. Check your internet connection "
        "and try again."
    )

# Remove timezone information and clean column names
df.index = df.index.tz_localize(None)
df.index.name = "Date"

# Save the data inside our project
Path("data").mkdir(exist_ok=True)
df.to_csv("data/tcs_stock_data.csv")

print("\nDownload successful!")
print(f"Total records: {len(df)}")
print("\nFirst 5 rows:")
print(df.head())
print("\nSaved to: data/tcs_stock_data.csv")
