#!/usr/bin/env python3
"""
signal_generator.py

Generate discrete trade signals from per-candle sentiment CSVs.

Produces a trades CSV suitable for consumption by backtester.py.

Usage examples:
  # simple (auto-locate sentiment file for ticker)
  python signal_generator.py --ticker KXCPI --entry_threshold 0.2 --hold_candles 24 --position_size 1000 --out trades_out.csv

  # direct files
  python signal_generator.py --sentiment_csv data/sentiment/KXCPI_sentiment.csv \
      --prices_csv data/prices/prices_KXCPI-25NOV-T0.3_20251101_20251113.csv \
      --entry_threshold 0.2 --hold_candles 24 --position_size 1000 \
      --yes_implies_good_economy True --out trades_out.csv

Notes / assumptions:
 - Sentiment CSV must contain at least: ts, iso_utc, price, n_headlines, <rolling_sentiment_X>.
 - For binary markets price is expected in [0,1]. The script uses a heuristic fair-value = 0.5 + sentiment/2.
 - position_size is treated as contract units (not dollars). Adjust to your convention if needed.
 - The script emits only entry trades. Use the backtester (or another script) to close/exit using hold_candles.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import re

def find_sentiment_file_for_ticker(ticker: str, sentiment_dir: Path):
    # look for <TICKER>_sentiment.csv
    candidates = list(sentiment_dir.glob(f"{ticker}_sentiment.csv")) + list(sentiment_dir.glob(f"{ticker}*_sentiment.csv"))
    if not candidates:
        # fallback: any file that contains ticker
        candidates = [p for p in sentiment_dir.glob("*.csv") if ticker in p.name]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

def infer_price_file_for_ticker(ticker: str, prices_dir: Path):
    candidates = list(prices_dir.glob(f"prices_{ticker}_*.csv")) + list(prices_dir.glob(f"prices_{ticker}*.csv"))
    if not candidates:
        candidates = [p for p in prices_dir.glob("*.csv") if ticker in p.name]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

def sniff_rolling_col(df: pd.DataFrame):
    # pick the first column matching rolling_sentiment_\d+
    for c in df.columns:
        if re.match(r"rolling_sentiment_\d+", c):
            return c
    # fallback to 'sentiment' if present
    if "sentiment" in df.columns:
        return "sentiment"
    # fallback to any column named 'rolling_sentiment' or 'sentiment_score'
    for alt in ("rolling_sentiment", "sentiment_score"):
        if alt in df.columns:
            return alt
    return None

def generate_signals_from_df(df: pd.DataFrame,
                             rolling_col: str,
                             yes_implies_good_economy: bool,
                             entry_threshold: float,
                             hold_candles: int,
                             position_size: float,
                             min_headlines: int = 1):
    """
    df must have: ts, iso_utc, price, n_headlines, and rolling_col
    Returns list of trades (dict)
    """
    trades = []
    in_position_until = None  # timestamp when current position should be considered open until (ts)
    # iterate rows in chronological order
    for i, row in df.sort_values("ts").reset_index(drop=True).iterrows():
        ts = int(row["ts"])
        iso_utc = row.get("iso_utc", "")
        price = float(row["price"]) if not pd.isna(row["price"]) else None
        n_headlines = int(row.get("n_headlines", 0))
        sentiment = row.get(rolling_col, None)

        # skip if not enough evidence
        if sentiment is None or n_headlines < min_headlines:
            continue

        # simple cooldown: if already in a position that hasn't expired, skip opening another
        # in_position_until is ts where we do allow another trade (strict >)
        if in_position_until is not None and ts <= in_position_until:
            continue

        # enforce threshold
        if abs(sentiment) < entry_threshold:
            continue

        # require price present
        if price is None or np.isnan(price):
            continue

        # heuristic: implied fair price based on sentiment; for binary-type contracts typical price range 0..1
        fair = 0.5 + float(sentiment) / 2.0
        # clamp
        fair = max(0.0, min(1.0, fair))

        # decide side depending on yes_implies_good_economy
        # positive sentiment => "YES is good" if yes_implies_good_economy True
        buy_yes_if_positive = bool(yes_implies_good_economy)

        # Determine candidate side and check price gap
        side = None
        reason = None
        gap = 0.0
        if sentiment > 0:
            # candidate to buy the 'positive' side
            if buy_yes_if_positive:
                # BUY_YES when price is sufficiently below fair
                gap = fair - price
                if gap > 0 and (gap >= 0.05 or (gap >= 0.01 and abs(sentiment) >= entry_threshold*1.5)):
                    side = "BUY_YES"
                    reason = f"sentiment_pos={sentiment:.3f} fair={fair:.3f} price_gap={gap:.3f}"
            else:
                # positive sentiment implies NO (rare), but follow mapping
                gap = price - fair
                if gap > 0 and (gap >= 0.05 or (gap >= 0.01 and abs(sentiment) >= entry_threshold*1.5)):
                    side = "BUY_NO"
                    reason = f"sentiment_pos={sentiment:.3f} fair={fair:.3f} price_gap={gap:.3f}"
        elif sentiment < 0:
            # candidate to buy the 'negative' side
            if buy_yes_if_positive:
                # negative sentiment => BUY_NO when price sufficiently above fair
                gap = price - fair
                if gap > 0 and (gap >= 0.05 or (gap >= 0.01 and abs(sentiment) >= entry_threshold*1.5)):
                    side = "BUY_NO"
                    reason = f"sentiment_neg={sentiment:.3f} fair={fair:.3f} price_gap={gap:.3f}"
            else:
                # negative sentiment implies BUY_YES (if mapping inverted)
                gap = fair - price
                if gap > 0 and (gap >= 0.05 or (gap >= 0.01 and abs(sentiment) >= entry_threshold*1.5)):
                    side = "BUY_YES"
                    reason = f"sentiment_neg={sentiment:.3f} fair={fair:.3f} price_gap={gap:.3f}"

        # If condition met, emit trade
        if side is not None:
            # quantity: we treat position_size as contract units (leave unchanged)
            qty = float(position_size)
            trades.append({
                "entry_ts": ts,
                "iso_utc": iso_utc,
                "side": side,
                "price": float(price),
                "quantity": qty,
                "hold_candles": int(hold_candles),
                "reason": reason
            })
            # cool-down: mark in position until ts + (hold_candles * candle_interval) but we only have candle count
            # Using candle-based cooldown: compute index-based; easier: store until next ts + hold_candles (by index)
            in_position_until = ts + (hold_candles * 1)  # logical placeholder -- backtester uses hold_candles field
            # we intentionally set simple numeric to block overlapped entries on identical ts values
    return trades

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", help="Ticker symbol (used to auto-find sentiment/prices files)", default=None)
    p.add_argument("--sentiment_csv", help="Path to sentiment CSV (overrides --ticker)")
    p.add_argument("--prices_csv", help="Optional path to raw prices CSV (used to validate prices)", default=None)
    p.add_argument("--sentiment_col", help="Column name for rolling sentiment (auto-detected by default)", default=None)
    p.add_argument("--sentiment_dir", default="data/sentiment", help="Directory with per-ticker sentiment CSVs")
    p.add_argument("--prices_dir", default="data/prices", help="Directory with price CSVs to auto-detect if needed")
    p.add_argument("--yes_implies_good_economy", type=lambda s: s.lower() in ("1","true","yes"), default=True)
    p.add_argument("--entry_threshold", type=float, default=0.2)
    p.add_argument("--hold_candles", type=int, default=24)
    p.add_argument("--position_size", type=float, default=1000.0)
    p.add_argument("--min_headlines", type=int, default=1, help="minimum headlines in candle to consider signal")
    p.add_argument("--out", dest="out_trades_csv", default="data/trades/trades_out.csv")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    sentiment_path = None
    prices_path = None
    sdir = Path(args.sentiment_dir)
    pdir = Path(args.prices_dir)

    if args.sentiment_csv:
        sentiment_path = Path(args.sentiment_csv)
    elif args.ticker:
        sentiment_path = find_sentiment_file_for_ticker(args.ticker, sdir)
        if sentiment_path is None:
            print(f"ERROR: no sentiment CSV found for ticker {args.ticker} in {sdir}", file=sys.stderr)
            sys.exit(2)
    else:
        print("ERROR: provide --sentiment_csv or --ticker", file=sys.stderr)
        sys.exit(2)

    if args.prices_csv:
        prices_path = Path(args.prices_csv)
    elif args.ticker:
        prices_path = infer_price_file_for_ticker(args.ticker, pdir)
        # prices optional; we prefer sentiment CSV which usually contains price column
    # load sentiment
    df = pd.read_csv(sentiment_path)
    if "ts" not in df.columns:
        # try to find timestamp column
        if "timestamp" in df.columns:
            df = df.rename(columns={"timestamp":"ts"})
    if "ts" not in df.columns:
        print(f"ERROR: sentiment CSV {sentiment_path} missing 'ts' column", file=sys.stderr)
        sys.exit(2)
    # ensure price column exists
    if "price" not in df.columns:
        if prices_path and prices_path.exists():
            p_df = pd.read_csv(prices_path)
            # try to merge by ts
            if "timestamp" in p_df.columns and "price" in p_df.columns:
                p_df = p_df.rename(columns={"timestamp":"ts"})
            if "ts" in p_df.columns and "price" in p_df.columns:
                merged = pd.merge(df, p_df[["ts","price"]], on="ts", how="left")
                df = merged
            else:
                print(f"[WARN] Unable to find price column in provided price file {prices_path}", file=sys.stderr)
        else:
            print(f"[WARN] sentiment CSV {sentiment_path} lacks 'price' column and no prices file provided. We'll only emit signals when price present.", file=sys.stderr)

    # detect rolling column
    rolling_col = args.sentiment_col or sniff_rolling_col(df)
    if rolling_col is None:
        print(f"ERROR: couldn't detect rolling sentiment column in {sentiment_path}. Provide --sentiment_col", file=sys.stderr)
        sys.exit(2)

    # basic cleaning
    df = df.dropna(subset=["ts"]).copy()
    df["ts"] = df["ts"].astype(int)
    # ensure iso_utc present
    if "iso_utc" not in df.columns:
        # try create from ts
        try:
            df["iso_utc"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            df["iso_utc"] = ""

    out_dir = Path(args.out_trades_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out_trades_csv)
    if out_path.exists() and not args.force:
        print(f"ERROR: output file {out_path} exists. Use --force to overwrite.", file=sys.stderr)
        sys.exit(2)

    trades = generate_signals_from_df(df,
                                      rolling_col=rolling_col,
                                      yes_implies_good_economy=args.yes_implies_good_economy,
                                      entry_threshold=args.entry_threshold,
                                      hold_candles=args.hold_candles,
                                      position_size=args.position_size,
                                      min_headlines=args.min_headlines)

    if not trades:
        print("[INFO] No trade signals generated with current thresholds/parameters.")
    else:
        out_df = pd.DataFrame(trades)
        # order columns
        cols = ["entry_ts","iso_utc","side","price","quantity","hold_candles","reason"]
        out_df = out_df.loc[:, cols]
        out_df.to_csv(out_path, index=False)
        print(f"[OK] Wrote {len(out_df)} trades to {out_path}")

if __name__ == "__main__":
    main()
