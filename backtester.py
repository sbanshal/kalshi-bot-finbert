#!/usr/bin/env python3
"""
backtester.py

MVP backtester for Kalshi sentiment trading pipeline.

Features:
- Load candles CSV (timestamp (s), price)
- Load headlines JSON (list of {"timestamp": int|str, "text": str})
- Optionally score headlines via HuggingFace FinBERT pipeline (if available)
- Align headlines to candlesticks (nearest or next candle)
- Aggregate sentiment per candle (mean)
- Map aggregated sentiment -> signals using market metadata (yes_implies_good_economy)
- Simulate simple trades (entry on next candle open; fixed hold in candles or until opposite)
- Produce: trades DataFrame, equity curve, metrics summary, optional trades CSV

Usage (CLI):
    python backtester.py --prices prices_market_YYYYMMDD.csv --headlines headlines_YYYYMMDD.json --yes_implies_good_economy True

Importable functions for integration in Streamlit.
"""
import json
import math
import argparse
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone

# Optional HF FinBERT scoring (only used if user wants and transformers available)
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# ---------------------
# Utilities
# ---------------------
def parse_iso_or_ts(v):
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        pass
    try:
        # try ISO parse
        dt = pd.to_datetime(v)
        if dt.tzinfo is None:
            dt = dt.tz_localize('UTC')
        return int(dt.timestamp())
    except Exception:
        raise ValueError(f"Cannot parse timestamp: {v}")


def finbert_to_score_texts(texts: List[str], hf_pipeline=None) -> List[float]:
    """
    Convert texts -> scalar scores in [-1, 1] using FinBERT or fallback simple polarity heuristic.
    If hf_pipeline is None or transformers not available, use a tiny fallback: polarity by counting
    positive/negative words. (You can replace with your existing FinBERT pipeline.)
    """
    if HF_AVAILABLE and hf_pipeline is None:
        try:
            hf_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)
        except Exception:
            hf_pipeline = None

    if hf_pipeline:
        out = hf_pipeline(texts, truncation=True)
        scores = []
        for o in out:
            label = o.get("label", "").lower()
            # labels may be POSITIVE/NEGATIVE/NEUTRAL or similar — map to [-1,1]
            if "positive" in label or "pos" in label:
                scores.append(float(o.get("score", 0.5)))
            elif "negative" in label or "neg" in label:
                scores.append(-float(o.get("score", 0.5)))
            elif "neutral" in label:
                scores.append(0.0)
            else:
                # fallback: map score with sign guess
                scores.append(0.0)
        return scores

    # Fallback naive scoring (very small heuristic)
    pos_words = {"up", "higher", "positive", "rise", "increased", "beat", "surge"}
    neg_words = {"down", "lower", "negative", "fall", "decline", "miss", "drop"}
    scores = []
    for t in texts:
        t_l = t.lower()
        pos = sum(1 for w in pos_words if w in t_l)
        neg = sum(1 for w in neg_words if w in t_l)
        if pos + neg == 0:
            scores.append(0.0)
        else:
            scores.append((pos - neg) / max(1, pos + neg))
    return scores


# ---------------------
# Core transform / align functions
# ---------------------
def load_candles_csv(path: str, timestamp_col: str = "timestamp", price_col: str = "price") -> pd.DataFrame:
    df = pd.read_csv(path)
    if timestamp_col not in df.columns:
        raise ValueError(f"{timestamp_col} not found in CSV")
    if price_col not in df.columns:
        # attempt to find common price columns
        for candidate in ["close", "price", "last"]:
            if candidate in df.columns:
                price_col = candidate
                break
        else:
            raise ValueError(f"No price column found in CSV; looked for {price_col} or close/price/last")
    df = df[[timestamp_col, price_col]].copy()
    df["timestamp"] = df[timestamp_col].astype(int)
    df["price"] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.sort_values("timestamp").dropna(subset=["price"]).reset_index(drop=True)
    # add candle index
    df = df.reset_index().rename(columns={"index": "candle_idx"})
    return df[["candle_idx", "timestamp", "price"]]


def load_headlines_json(path: str, ts_field: str = "timestamp", text_field: str = "text") -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    # data is expected to be a list of objects; try to normalize
    if isinstance(data, dict) and "headlines" in data:
        data = data["headlines"]
    rows = []
    for item in data:
        ts = item.get(ts_field) or item.get("time") or item.get("ts") or item.get("published_at")
        if ts is None:
            continue
        ts_i = parse_iso_or_ts(ts)
        rows.append({"timestamp": ts_i, "text": item.get(text_field) or item.get("headline") or item.get("title", "")})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def align_headlines_to_candles(headlines: pd.DataFrame, candles: pd.DataFrame, method: str = "nearest") -> pd.DataFrame:
    """
    Add candle_idx to each headline:
    - method = 'nearest': find the candle with nearest timestamp
    - method = 'next': find the first candle with timestamp > headline_ts
    """
    if headlines.empty:
        return headlines.assign(candle_idx=pd.Series(dtype=int), candle_ts=pd.Series(dtype=int))
    # for efficient search, use numpy searchsorted on candle timestamps
    ts_arr = candles["timestamp"].to_numpy()
    result_idxs = []
    result_candle_ts = []
    for ts in headlines["timestamp"].to_numpy():
        pos = np.searchsorted(ts_arr, ts, side="left")
        if method == "next":
            idx = pos if pos < len(ts_arr) else len(ts_arr) - 1
        else:  # nearest
            if pos == 0:
                idx = 0
            elif pos >= len(ts_arr):
                idx = len(ts_arr) - 1
            else:
                before = pos - 1
                after = pos
                idx = before if abs(ts_arr[before] - ts) <= abs(ts_arr[after] - ts) else after
        result_idxs.append(int(candles.iloc[idx]["candle_idx"]))
        result_candle_ts.append(int(candles.iloc[idx]["timestamp"]))
    headlines = headlines.copy()
    headlines["candle_idx"] = result_idxs
    headlines["candle_ts"] = result_candle_ts
    return headlines


def aggregate_sentiment_per_candle(headlines: pd.DataFrame, method="mean") -> pd.Series:
    """
    headlines must include columns: candle_idx and sentiment
    Returns a pd.Series indexed by candle_idx with aggregated sentiment (fill 0 for no headlines).
    """
    if headlines.empty:
        return pd.Series(dtype=float)
    agg = headlines.groupby("candle_idx")["sentiment"].agg(method)
    return agg


# ---------------------
# Signal mapping & simulation
# ---------------------
@dataclass
class BacktestParams:
    entry_threshold: float = 0.2   # threshold on aggregated sentiment
    hold_candles: int = 24         # default hold length
    slippage: float = 0.0005       # fraction of price
    commission: float = 0.0        # fraction of trade value
    initial_capital: float = 10000.0
    position_size: float = 1000.0  # USD per trade (not leverage)
    aggregate_method: str = "mean"
    align_method: str = "nearest"
    hf_score: bool = False
    hf_pipeline = None


def sentiment_to_signal(agg_sent: float, yes_implies_good_economy: bool, threshold: float) -> str:
    """
    Map aggregated sentiment scalar -> one of "BUY_YES", "BUY_NO", "HOLD"
    If yes_implies_good_economy==True, positive sentiment favors YES; otherwise invert.
    """
    if math.isnan(agg_sent):
        return "HOLD"
    val = agg_sent if yes_implies_good_economy else -agg_sent
    if val >= threshold:
        return "BUY_YES"
    if val <= -threshold:
        return "BUY_NO"
    return "HOLD"


def simulate(candles: pd.DataFrame,
             headlines: pd.DataFrame,
             yes_implies_good_economy: bool,
             params: BacktestParams) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Run simulation. Returns (trades_df, equity_curve_df, metrics_dict)
    Simple rule:
      - For each candle where aggregated signal crosses threshold -> open position next candle at its price
      - Position direction: long for BUY_YES, short for BUY_NO
      - Position closed after hold_candles OR when opposite signal occurs (whichever first)
    """
    # prepare candles
    candles = candles.copy().reset_index(drop=True)
    n = candles.shape[0]
    candles_idx_map = {int(r["candle_idx"]): i for i, r in candles.reset_index().to_dict("index").items()}
    # Score headlines if needed
    hl = headlines.copy()
    if params.hf_score:
        texts = hl["text"].fillna("").astype(str).tolist()
        scores = finbert_to_score_texts(texts, hf_pipeline=params.hf_pipeline)
        hl["sentiment"] = scores
    elif "sentiment" not in hl.columns:
        # assume they provided sentiment externally; default to 0
        hl["sentiment"] = 0.0

    # align headlines
    if "candle_idx" not in hl.columns:
        hl = align_headlines_to_candles(hl, candles, method=params.align_method)

    # aggregate sentiment per candle
    agg = aggregate_sentiment_per_candle(hl, method=params.aggregate_method)
    # create agg series for all candles
    agg_full = pd.Series(0.0, index=candles["candle_idx"].tolist())
    for idx, v in agg.items():
        agg_full.loc[idx] = float(v)

    # compute signals per candle
    signals = []
    for idx in candles["candle_idx"].tolist():
        agg_sent = float(agg_full.loc[idx])
        sig = sentiment_to_signal(agg_sent, yes_implies_good_economy, params.entry_threshold)
        signals.append(sig)
    candles = candles.copy()
    candles["agg_sentiment"] = [float(agg_full.loc[idx]) for idx in candles["candle_idx"].tolist()]
    candles["signal"] = signals

    # Simulation variables
    trades = []
    capital = params.initial_capital
    equity_points = []
    open_pos = None  # dict with keys: entry_idx, entry_price, direction (+1 for yes/long, -1 for no/short), size_usd, entry_capital
    for i, row in candles.iterrows():
        ts = int(row["timestamp"])
        price = float(row["price"])
        equity_points.append({"timestamp": ts, "capital": capital, "candle_idx": int(row["candle_idx"])})
        # check open pos closure conditions
        if open_pos is not None:
            holds = i - open_pos["entry_i"]
            # close if hold exceeded
            should_close = False
            close_reason = None
            if holds >= params.hold_candles:
                should_close = True
                close_reason = "time"
            else:
                # check opposite signal at this candle
                if row["signal"] == "BUY_YES" and open_pos["direction"] == -1:
                    should_close = True
                    close_reason = "opposite_signal"
                if row["signal"] == "BUY_NO" and open_pos["direction"] == 1:
                    should_close = True
                    close_reason = "opposite_signal"
            if should_close:
                exit_price = price * (1 + params.slippage * (1 if open_pos["direction"] < 0 else -1))
                # direction: +1 for long YES, -1 for short NO
                pnl = open_pos["direction"] * (exit_price - open_pos["entry_price"]) * (open_pos["size_usd"] / open_pos["entry_price"])
                # subtract commission (both entry & exit)
                commission_cost = open_pos["size_usd"] * params.commission * 2
                pnl_after = pnl - commission_cost
                capital += pnl_after
                trades.append({
                    "entry_i": open_pos["entry_i"],
                    "exit_i": i,
                    "entry_ts": int(open_pos["entry_ts"]),
                    "exit_ts": int(ts),
                    "entry_price": open_pos["entry_price"],
                    "exit_price": exit_price,
                    "direction": "LONG" if open_pos["direction"] == 1 else "SHORT",
                    "size_usd": open_pos["size_usd"],
                    "pnl": pnl_after,
                    "gross_pnl": pnl,
                    "commission": commission_cost,
                    "close_reason": close_reason
                })
                open_pos = None

        # if no open position, consider opening based on this candle's signal
        if open_pos is None:
            sig = row["signal"]
            if sig in ("BUY_YES", "BUY_NO"):
                # open at next candle's price if exists; else open at current price
                if i + 1 < len(candles):
                    next_price = float(candles.iloc[i + 1]["price"])
                    entry_price = next_price * (1 + params.slippage * (1 if sig == "BUY_NO" else -1))
                    entry_i = i + 1
                    entry_ts = int(candles.iloc[i + 1]["timestamp"])
                else:
                    entry_price = price * (1 + params.slippage * (1 if sig == "BUY_NO" else -1))
                    entry_i = i
                    entry_ts = ts
                direction = 1 if sig == "BUY_YES" else -1
                size_usd = params.position_size
                # reserve size_usd from capital (we will assume unlimited margin if needed; simple implementation)
                open_pos = {
                    "entry_i": entry_i,
                    "entry_price": entry_price,
                    "entry_ts": entry_ts,
                    "direction": direction,
                    "size_usd": size_usd
                }

    # finalize equity curve points (append final)
    equity_df = pd.DataFrame(equity_points).drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    trades_df = pd.DataFrame(trades)
    # compute metrics
    total_return = (capital - params.initial_capital) / params.initial_capital if params.initial_capital else 0.0
    returns = equity_df["capital"].pct_change().fillna(0)
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * math.sqrt(252 * (24*3600) / (candles["timestamp"].diff().median() or 1))
    else:
        sharpe = float("nan")
    # max drawdown
    eq = equity_df["capital"].to_numpy()
    peak = -np.inf
    max_dd = 0.0
    peak = eq[0] if len(eq) > 0 else params.initial_capital
    running_max = np.maximum.accumulate(eq) if len(eq) > 0 else np.array([])
    drawdowns = (running_max - eq) / running_max if len(eq) > 0 else np.array([0.0])
    max_dd = float(np.nanmax(drawdowns)) if len(drawdowns) > 0 else 0.0

    metrics = {
        "initial_capital": params.initial_capital,
        "final_capital": capital,
        "total_return": total_return,
        "num_trades": len(trades_df),
        "sharpe": float(sharpe) if not math.isnan(sharpe) else None,
        "max_drawdown": max_dd
    }

    return trades_df, equity_df, metrics


# ---------------------
# CLI
# ---------------------
def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--prices", required=True, help="CSV produced by discover_and_fetch (timestamp,price)")
    p.add_argument("--headlines", required=True, help="JSON file with headlines (list of objects with timestamp & text)")
    p.add_argument("--yes_implies_good_economy", required=True, type=lambda x: x.lower() in ("true", "1", "t", "y"), help="True|False")
    p.add_argument("--entry_threshold", type=float, default=0.2)
    p.add_argument("--hold_candles", type=int, default=24)
    p.add_argument("--position_size", type=float, default=1000.0)
    p.add_argument("--initial_capital", type=float, default=10000.0)
    p.add_argument("--out_trades_csv", help="optional path to write trades CSV")
    p.add_argument("--hf_score", action="store_true", help="If present, attempt to score headlines with FinBERT (requires transformers)")
    args = p.parse_args()

    candles = load_candles_csv(args.prices)
    headlines = load_headlines_json(args.headlines)
    params = BacktestParams(entry_threshold=args.entry_threshold,
                            hold_candles=args.hold_candles,
                            initial_capital=args.initial_capital,
                            position_size=args.position_size,
                            hf_score=args.hf_score)
    if args.hf_score and HF_AVAILABLE:
        params.hf_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

    trades_df, equity_df, metrics = simulate(candles, headlines, args.yes_implies_good_economy, params)
    print("METRICS")
    print(json.dumps(metrics, indent=2))
    print("TRADES")
    print(trades_df.to_string(index=False))
    if args.out_trades_csv:
        trades_df.to_csv(args.out_trades_csv, index=False)
        print("Wrote trades to", args.out_trades_csv)


if __name__ == "__main__":
    cli()
