#!/usr/bin/env python3
"""
Streamlit Backtest UI

Run:
    streamlit run app_streamlit_backtest.py

Features:
- Upload prices CSV and headlines JSON (or point to filenames)
- Optional: use FinBERT scoring if transformers available
- Configure thresholds & sim params
- Run backtest -> show:
  - Price chart with entry/exit markers
  - Equity curve
  - Trades table + download
"""
import streamlit as st
import pandas as pd
import io
import json
import base64
from datetime import datetime, timezone
import plotly.graph_objects as go
import plotly.express as px

# import backtester functions
from backtester import load_candles_csv, load_headlines_json, BacktestParams, simulate, HF_AVAILABLE

st.set_page_config(page_title="Kalshi Sentiment Backtester", layout="wide")

st.title("Kalshi Sentiment Backtester")

with st.sidebar:
    st.header("Upload data")
    prices_file = st.file_uploader("Prices CSV (timestamp in seconds, price)", type=["csv"])
    headlines_file = st.file_uploader("Headlines JSON", type=["json"])
    st.markdown("---")
    st.header("Or sample files")
    use_sample = st.checkbox("Use sample bundled data", value=False)

st.subheader("Backtest parameters")
col1, col2, col3 = st.columns(3)
with col1:
    yes_implies_good = st.selectbox("yes_implies_good_economy", options=[True, False], index=0)
    entry_threshold = st.number_input("Entry threshold (abs)", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    hold_candles = st.number_input("Hold (candles)", min_value=1, max_value=10000, value=24)
with col2:
    position_size = st.number_input("Position size (USD)", value=1000.0, step=100.0)
    initial_capital = st.number_input("Initial capital (USD)", value=10000.0, step=100.0)
    slippage = st.number_input("Slippage fraction", min_value=0.0, max_value=0.1, value=0.0005, step=0.0001)
with col3:
    commission = st.number_input("Commission fraction per side", min_value=0.0, max_value=0.1, value=0.0, step=0.0001)
    aggregate_method = st.selectbox("Aggregate method for candle", options=["mean", "median"], index=0)
    align_method = st.selectbox("Align headlines to candle", options=["nearest", "next"], index=0)

st.markdown("---")

# Load data
if use_sample:
    st.info("Using bundled tiny sample (for demo).")
    # build tiny sample in memory
    now = int(datetime.now(tz=timezone.utc).timestamp())
    # 48 hourly candles
    times = [now + 3600 * i for i in range(48)]
    prices_df = pd.DataFrame({"timestamp": times, "price": (100 + pd.Series(range(48)).apply(lambda x: math.sin(x/5)*2 + x*0.02)).values})
    # small headlines sample:
    headlines = [
        {"timestamp": times[2], "text": "CPI rose more than expected, inflation higher"},
        {"timestamp": times[10], "text": "Inflation cooled slightly in latest report"},
        {"timestamp": times[20], "text": "Surge in prices surprises economists"},
    ]
    headlines_df = pd.DataFrame(headlines)
else:
    prices_df = None
    headlines_df = None
    if prices_file:
        try:
            prices_df = pd.read_csv(prices_file)
            if "timestamp" not in prices_df.columns:
                st.error("Prices CSV must contain 'timestamp' column (unix seconds).")
        except Exception as e:
            st.error(f"Failed to read prices CSV: {e}")
    if headlines_file:
        try:
            raw = json.load(headlines_file)
            # normalize to list
            if isinstance(raw, dict) and "headlines" in raw:
                raw = raw["headlines"]
            headlines_df = pd.json_normalize(raw)
            # ensure required columns
            if "timestamp" not in headlines_df.columns:
                st.warning("Headlines JSON does not contain explicit 'timestamp' field; trying to infer.")
        except Exception as e:
            st.error(f"Failed to read headlines JSON: {e}")

run_bt = st.button("Run backtest")

if run_bt:
    if prices_df is None or headlines_df is None:
        st.error("Please upload both prices CSV and headlines JSON (or use sample).")
    else:
        # ensure candle shape compatible with backtester
        prices_df.to_csv("tmp_prices_for_bt.csv", index=False)
        # Convert headlines_df to JSON-like list and save
        headlines_json_path = "tmp_headlines_for_bt.json"
        with open(headlines_json_path, "w", encoding="utf-8") as fh:
            # if data frame has columns text or headline/title, attempt to normalize:
            if "text" not in headlines_df.columns:
                # try 'headline' or 'title'
                if "headline" in headlines_df.columns:
                    headlines_df = headlines_df.rename(columns={"headline": "text"})
                elif "title" in headlines_df.columns:
                    headlines_df = headlines_df.rename(columns={"title": "text"})
            # ensure timestamp column is int or iso
            headlines_df = headlines_df.rename(columns={c: c for c in headlines_df.columns})
            # output list of dicts
            js_list = headlines_df.to_dict(orient="records")
            json.dump(js_list, fh, ensure_ascii=False, indent=2)

        # Import and run backtester
        from backtester import load_candles_csv, load_headlines_json, BacktestParams, simulate

        candles = load_candles_csv("tmp_prices_for_bt.csv")
        headlines = load_headlines_json(headlines_json_path)

        params = BacktestParams(
            entry_threshold=entry_threshold,
            hold_candles=int(hold_candles),
            slippage=float(slippage),
            commission=float(commission),
            initial_capital=float(initial_capital),
            position_size=float(position_size),
            aggregate_method=aggregate_method,
            align_method=align_method,
            hf_score=False
        )

        trades_df, equity_df, metrics = simulate(candles, headlines, yes_implies_good, params)

        st.success("Backtest complete")
        st.write("## Summary metrics")
        st.json(metrics)

        # Price chart + markers
        st.write("## Price chart with trades")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pd.to_datetime(candles["timestamp"], unit="s"),
                                 y=candles["price"], mode="lines", name="price"))
        # add trade markers
        if not trades_df.empty:
            for _, t in trades_df.iterrows():
                # entry and exit timestamps are present
                entry_ts = int(t["entry_ts"])
                exit_ts = int(t["exit_ts"])
                entry_idx = candles[candles["timestamp"] == entry_ts].index
                exit_idx = candles[candles["timestamp"] == exit_ts].index
                entry_price = t["entry_price"]
                exit_price = t["exit_price"]
                dir_text = t["direction"]
                color = "green" if dir_text == "LONG" else "red"
                fig.add_trace(go.Scatter(
                    x=[pd.to_datetime(entry_ts, unit="s")],
                    y=[entry_price],
                    mode="markers",
                    marker=dict(symbol="triangle-up" if dir_text == "LONG" else "triangle-down", size=12, color=color),
                    name=f"Entry {dir_text}"
                ))
                fig.add_trace(go.Scatter(
                    x=[pd.to_datetime(exit_ts, unit="s")],
                    y=[exit_price],
                    mode="markers",
                    marker=dict(symbol="x", size=10, color=color),
                    name=f"Exit {dir_text}"
                ))
        st.plotly_chart(fig, use_container_width=True)

        # Equity curve
        st.write("## Equity curve")
        eq_fig = px.line(equity_df, x=pd.to_datetime(equity_df["timestamp"], unit="s"), y="capital", labels={"x":"time","capital":"capital"})
        st.plotly_chart(eq_fig, use_container_width=True)

        # Trades table
        st.write("## Trades")
        if trades_df.empty:
            st.info("No trades executed with current params.")
        else:
            st.dataframe(trades_df)

            # download trades CSV
            buf = io.StringIO()
            trades_df.to_csv(buf, index=False)
            b64 = base64.b64encode(buf.getvalue().encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="trades.csv">Download trades.csv</a>'
            st.markdown(href, unsafe_allow_html=True)