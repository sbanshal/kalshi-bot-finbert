# app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
from transformers import pipeline

st.set_page_config(layout="wide")
st.title("Kalshi — Economics Sentiment Dashboard (Demo)")

mapping = json.load(open("market_mapping.json"))
market_id = st.selectbox("Market", list(mapping.keys()))
info = mapping[market_id]

st.write("Market info:", info.get("title",""))
# load prices CSV if exists
prices_file = st.file_uploader("Upload prices CSV (timestamp,price)", type=["csv"])
headlines_file = st.file_uploader("Upload headlines JSON (from news_ingestor)", type=["json"])

if headlines_file:
    headlines_j = json.load(headlines_file)
    hlist = headlines_j.get(market_id, [])
    dfh = pd.DataFrame(hlist)
    dfh["publishedAt"] = pd.to_datetime(dfh["publishedAt"])
    dfh = dfh.sort_values("publishedAt")
    st.write("Recent headlines:", dfh[["publishedAt","title","source"]].tail(20))

    if st.button("Score headlines (FinBERT)"):
        pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert")
        dfh["score"] = dfh["title"].apply(lambda t: (pipe(t[:512])[0]["score"] * (1 if "positive" in pipe(t[:512])[0]["label"].lower() else -1)))
        st.line_chart(dfh.set_index("publishedAt")["score"].ewm(alpha=0.5).mean())
else:
    st.info("Upload headlines JSON to score.")

if prices_file:
    dfp = pd.read_csv(prices_file, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    st.line_chart(dfp["price"].tail(500))
