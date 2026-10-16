
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="TCS Stock Analytics",
    page_icon="📈",
    layout="wide"
)

st.title("📈 TCS Stock Analytics & ML Dashboard")
st.caption(
    "Historical analysis and experimental next-day price predictions. "
    "For educational purposes only."
)

data_path = Path("data/tcs_stock_data.csv")
pred_path = Path("outputs/return_predictions.csv")

if not data_path.exists() or not pred_path.exists():
    st.error("Required data files are missing. Run the download and training scripts first.")
    st.stop()

stock = pd.read_csv(data_path, parse_dates=["Date"])
pred = pd.read_csv(pred_path, parse_dates=["Date"])

stock = stock.sort_values("Date")
pred = pred.sort_values("Date")

# Sidebar filters
st.sidebar.header("Dashboard Settings")
start_date = st.sidebar.date_input(
    "Start date", stock["Date"].min().date()
)
end_date = st.sidebar.date_input(
    "End date", stock["Date"].max().date()
)

if start_date > end_date:
    st.error("Start date must be on or before end date.")
    st.stop()

filtered = stock[
    (stock["Date"].dt.date >= start_date)
    & (stock["Date"].dt.date <= end_date)
]

if filtered.empty:
    st.warning("No stock data is available for this date range.")
    st.stop()

latest = filtered.iloc[-1]

c1, c2, c3 = st.columns(3)
c1.metric("Latest Close", f"₹{latest['Close']:,.2f}")
c2.metric("Highest Close", f"₹{filtered['Close'].max():,.2f}")
c3.metric("Lowest Close", f"₹{filtered['Close'].min():,.2f}")

st.subheader("Historical Closing Price")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(filtered["Date"], filtered["Close"])
ax.set_xlabel("Date")
ax.set_ylabel("Price (INR)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("Trading Volume")
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(filtered["Date"], filtered["Volume"])
ax.set_xlabel("Date")
ax.set_ylabel("Volume")
ax.grid(True, alpha=0.3)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("Model Evaluation")
comparison = pred[
    (pred["Date"].dt.date >= start_date)
    & (pred["Date"].dt.date <= end_date)
]

if not comparison.empty:
    mae = (
        comparison["Actual_Close"] - comparison["Predicted_Close"]
    ).abs().mean()
    baseline_mae = (
        comparison["Actual_Close"] - comparison["Baseline_Close"]
    ).abs().mean()

    m1, m2 = st.columns(2)
    m1.metric("Return Model MAE", f"₹{mae:.2f}")
    m2.metric("Baseline MAE", f"₹{baseline_mae:.2f}")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        comparison["Date"], comparison["Actual_Close"],
        label="Actual", linewidth=2
    )
    ax.plot(
        comparison["Date"], comparison["Predicted_Close"],
        label="Random Forest"
    )
    ax.plot(
        comparison["Date"], comparison["Baseline_Close"],
        label="Baseline", alpha=0.8
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (INR)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(
        comparison.tail(20).sort_values("Date", ascending=False),
        use_container_width=True
    )
else:
    st.info("No model predictions exist for the selected dates.")

st.caption(
    "Historical prices and model outputs are not investment advice. "
    "Past performance does not guarantee future results."
)
