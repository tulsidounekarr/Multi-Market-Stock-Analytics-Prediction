"""Streamlit dashboard for the multi-market LSTM project."""
import json, re
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"outputs/lstm"
STOCKS={"TCS (India)":"TCS.NS","Reliance (India)":"RELIANCE.NS","Infosys (India)":"INFY.NS",
        "Apple (USA)":"AAPL","Microsoft (USA)":"MSFT","NVIDIA (USA)":"NVDA"}
def safe_name(s): return re.sub(r"[^A-Za-z0-9_-]","_",s)

st.set_page_config(page_title="Multi-Market LSTM Stock Dashboard",page_icon="📈",layout="wide")
st.title("📈 Multi-Market Stock Trend Prediction")
st.caption("Educational forecasting demo — not financial advice.")
with st.sidebar:
    company=st.selectbox("Select stock",list(STOCKS.keys()))
    ticker=STOCKS[company]
    period=st.selectbox("Chart period",["6mo","1y","2y","5y"],index=2)
    if st.button("Refresh data"): st.cache_data.clear()

@st.cache_data(ttl=3600)
def history(symbol,period):
    df=yf.download(symbol,period=period,interval="1d",auto_adjust=True,progress=False)
    if df.empty: return pd.DataFrame()
    if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
    return df.reset_index()

try: hist=history(ticker,period)
except Exception as e:
    st.error(f"Could not download data: {e}"); st.stop()
if hist.empty:
    st.warning("No data returned. Check internet connection or ticker."); st.stop()
currency="INR" if ticker.endswith(".NS") else "USD"
close=hist["Close"]
a,b,c=st.columns(3)
a.metric("Ticker",ticker); b.metric("Latest close",f"{currency} {close.iloc[-1]:,.2f}")
c.metric("Last-session change",f"{((close.iloc[-1]/close.iloc[-2]-1)*100):+.2f}%" if len(close)>1 else "N/A")
st.plotly_chart(px.line(hist,x="Date",y="Close",title=f"{ticker} historical closing price"),use_container_width=True)
safe=safe_name(ticker); mp=OUT/f"{safe}_metrics.json"; pp=OUT/f"{safe}_predictions.csv"
st.subheader("LSTM evaluation")
if mp.exists():
    m=json.loads(mp.read_text(encoding="utf-8"))
    x1,x2,x3,x4=st.columns(4)
    x1.metric("LSTM MAE",f"{m['mae']:.3f} {currency}"); x2.metric("Baseline MAE",f"{m['baseline_mae']:.3f} {currency}")
    x3.metric("LSTM RMSE",f"{m['rmse']:.3f} {currency}"); x4.metric("R²",f"{m['r2']:.3f}")
    st.success("LSTM beat the naive baseline on this test period.") if m["mae"]<m["baseline_mae"] else st.warning("LSTM did not beat the naive baseline on this test period.")
else: st.info(f"No saved metrics for {ticker}. Train it in the terminal first.")
if pp.exists():
    pred=pd.read_csv(pp,parse_dates=["Date"])
    st.plotly_chart(px.line(pred,x="Date",y=["Actual_Close","LSTM_Predicted_Close","Naive_Baseline_Close"],
                            title="Actual vs LSTM vs naive baseline"),use_container_width=True)
    st.dataframe(pred.tail(20),use_container_width=True)
st.subheader("Training command")
st.code(f"python train_lstm.py --ticker {ticker}",language="powershell")
