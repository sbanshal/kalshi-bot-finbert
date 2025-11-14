#!/usr/bin/env python3
"""
sentiment_aggregator.py

Aggregate FinBERT-scored headlines onto candlesticks and produce per-candle sentiment CSVs.

Usage examples:
  # Aggregate every scored file found under data/headlines_scored/
  python sentiment_aggregator.py

  # Single-run for a specific scored file
  python sentiment_aggregator.py --scored data/headlines_scored/KXCPI_20251113_scored.json

Options:
  --headlines-dir    directory with scored files (default: data/headlines_scored)
  --prices-dir       directory with price CSVs (default: data/prices)
  --out-dir          directory for per-market sentiment CSVs (default: data/sentiment)
  --rolling-window   rolling window in candles for 'rolling_sentiment' (default: 3)
  --match_tolerance  seconds tolerance when mapping headlines to nearest candle (default: 3600)
  --force            overwrite existing output files
  --quiet            reduce logs
"""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import re

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--headlines-dir", default="data/headlines_scored")
    p.add_argument("--prices-dir", default="data/prices")
    p.add_argument("--out-dir", default="data/sentiment")
    p.add_argument("--scored", default=None, help="optional: single scored headlines file to process")
    p.add_argument("--rolling-window", type=int, default=3, help="rolling window size (candles)")
    p.add_argument("--match-tolerance", type=int, default=3600, help="seconds tolerance to map headline -> candle")
    p.add_argument("--force", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()

ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"

def iso_to_ts(s):
    try:
        return int(datetime.fromisoformat(s.replace("Z","+00:00")).timestamp())
    except Exception:
        try:
            # try parsing naive formats
            return int(pd.to_datetime(s, utc=True).timestamp())
        except Exception:
            return None

def infer_date_from_filename(p: Path):
    # look for YYYYMMDD or YYYY-MM-DD in filename
    m = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", p.name)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None

def load_scored_headlines(path: Path):
    j = json.loads(path.read_text(encoding="utf-8"))
    heads = j.get("headlines", [])
    rows = []
    for i, h in enumerate(heads):
        text = h.get("text") or h.get("title") or h.get("headline") or ""
        sentiment = h.get("sentiment")
        # find any timestamp-like fields
        ts = None
        for k in ("published_at", "published", "timestamp", "iso_utc", "date"):
            if k in h and h[k]:
                # field may be numeric or iso string
                v = h[k]
                if isinstance(v, (int, float)):
                    ts = int(v)
                elif isinstance(v, str):
                    ts = iso_to_ts(v)
                if ts:
                    break
        rows.append({"text": text, "sentiment": sentiment, "ts": ts, "raw": h})
    return rows

def find_price_file(prices_dir: Path, ticker: str):
    # search for files named prices_<ticker>_*.csv
    candidates = list(prices_dir.glob(f"prices_{ticker}_*.csv")) + list(prices_dir.glob(f"prices_{ticker}*.csv"))
    if not candidates:
        # try any file that contains ticker
        candidates = [f for f in prices_dir.glob("*.csv") if ticker in f.name]
    if not candidates:
        return None
    # prefer the latest by mtime
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

def read_candles(csv_path: Path):
    # expect columns: timestamp,price,iso_utc (timestamp in epoch seconds)
    df = pd.read_csv(csv_path)
    # normalize columns
    if "timestamp" not in df.columns and "ts" in df.columns:
        df = df.rename(columns={"ts": "timestamp"})
    if "price" not in df.columns:
        # try close/open etc
        for c in ("close","Close","price_usd","mid"):
            if c in df.columns:
                df = df.rename(columns={c:"price"}); break
    # parse iso_utc if available
    if "iso_utc" in df.columns:
        df["ts"] = pd.to_datetime(df["iso_utc"], utc=True).astype(int) // 10**9
    elif "timestamp" in df.columns:
        df["ts"] = df["timestamp"].astype(int)
        # create iso_utc
        df["iso_utc"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.strftime(ISO_FMT)
    else:
        raise RuntimeError(f"Price CSV {csv_path} missing timestamp/iso_utc")
    # ensure sorted ascending
    df = df.sort_values("ts").reset_index(drop=True)
    return df[["ts","iso_utc","price"]]

def map_headlines_to_candles(head_rows, candles_df, fallback_day_ts=None, tolerance=3600):
    # head_rows: list of dicts with 'ts' (epoch seconds or None) and 'sentiment'
    # candles_df: DataFrame with 'ts' sorted ascending
    if candles_df is None or candles_df.shape[0] == 0:
        # group all headlines into a single synthetic candle at fallback_day_ts
        grouped = {}
        # fallback_day_ts must be provided
        ts0 = fallback_day_ts or int(pd.Timestamp.now(tz="UTC").timestamp())
        agg = {"ts": ts0, "iso_utc": datetime.fromtimestamp(ts0, tz=timezone.utc).strftime(ISO_FMT), "price": np.nan, "n_headlines": 0, "sentiment": np.nan}
        sents = []
        for h in head_rows:
            if h.get("sentiment") is not None:
                sents.append(h["sentiment"])
        if sents:
            agg["sentiment"] = float(np.mean([s for s in sents]))
            agg["n_headlines"] = len(sents)
        return pd.DataFrame([agg])
    # create DataFrame of headlines with resolved ts (fall back to nan)
    heads = []
    for h in head_rows:
        heads.append({"ts": h.get("ts"), "sentiment": h.get("sentiment")})
    heads_df = pd.DataFrame(heads)
    # if no ts at all, set ts to fallback_day_ts
    if heads_df["ts"].isnull().all():
        # set to middle of candles range
        fallback = int(candles_df["ts"].iloc[len(candles_df)//2])
        heads_df["ts"] = fallback
    # merge_asof requires both sorted
    heads_df = heads_df.sort_values("ts").reset_index(drop=True)
    c = candles_df[["ts"]].copy()
    c = c.rename(columns={"ts":"candle_ts"})
    # use merge_asof to find nearest previous candle. We'll also check next candle distance.
    # First align to previous candle
    merged_prev = pd.merge_asof(heads_df.sort_values("ts"), candles_df.rename(columns={"ts":"candle_ts"}).sort_values("candle_ts"), left_on="ts", right_on="candle_ts", direction="backward")
    # Now align to next candle
    merged_next = pd.merge_asof(heads_df.sort_values("ts"), candles_df.rename(columns={"ts":"candle_ts"}).sort_values("candle_ts"), left_on="ts", right_on="candle_ts", direction="forward")
    # choose nearest of prev or next
    assigned = []
    for i, row in heads_df.sort_values("ts").reset_index(drop=True).iterrows():
        prev_row = merged_prev.iloc[i]
        next_row = merged_next.iloc[i]
        ts_head = row["ts"]
        cand_prev_ts = prev_row.get("candle_ts")
        cand_next_ts = next_row.get("candle_ts")
        best_cand_ts = None
        best_dist = None
        if not pd.isnull(cand_prev_ts):
            d = abs(int(ts_head) - int(cand_prev_ts))
            best_cand_ts = int(cand_prev_ts); best_dist = d
        if not pd.isnull(cand_next_ts):
            d2 = abs(int(cand_next_ts) - int(ts_head))
            if best_dist is None or d2 < best_dist:
                best_cand_ts = int(cand_next_ts); best_dist = d2
        # if distance exceeds tolerance, we may still assign (or drop). We'll keep but mark dist.
        assigned.append({"head_ts": int(ts_head), "candle_ts": best_cand_ts, "dist": best_dist, "sentiment": row.get("sentiment")})
    assigned_df = pd.DataFrame(assigned)
    # group by candle_ts
    grouped = assigned_df.groupby("candle_ts").agg(n_headlines=("sentiment","count"), sentiment=("sentiment", lambda arr: float(np.mean([x for x in arr if x is not None])) if len(arr)>0 else np.nan)).reset_index()
    # join with candles_df to include price/iso_utc
    result = pd.merge(candles_df.rename(columns={"ts":"candle_ts"}), grouped, on="candle_ts", how="left")
    result = result.rename(columns={"candle_ts":"ts", "n_headlines":"n_headlines", "sentiment":"sentiment"})
    # fill n_headlines with 0 where NaN
    result["n_headlines"] = result["n_headlines"].fillna(0).astype(int)
    return result[["ts","iso_utc","price","n_headlines","sentiment"]]

def process_one_file(scored_path: Path, prices_dir: Path, out_dir: Path, rolling_window: int, match_tolerance: int, force=False, quiet=False):
    # infer ticker from filename: <TICKER>_YYYYMMDD_scored.json or <TICKER>_YYYYMMDD.json
    name = scored_path.stem
    # remove trailing _scored if present
    if name.endswith("_scored"):
        name = name[:-7]
    # ticker is up to first underscore
    parts = name.split("_")
    if len(parts) >= 2 and re.match(r"^[A-Z0-9]+", parts[0]):
        ticker = parts[0]
    else:
        # fallback: use entire stem
        ticker = name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ticker}_sentiment.csv"
    if out_path.exists() and not force:
        if not quiet:
            print(f"[SKIP] {out_path} exists (use --force to overwrite)")
        return out_path

    heads = load_scored_headlines(scored_path)
    # convert none sentiments to NaN and drop
    heads = [h for h in heads if h.get("sentiment") is not None]
    if not heads:
        if not quiet:
            print(f"[WARN] no scored headlines found in {scored_path}")
        return None

    # try find price file
    price_file = find_price_file(Path(prices_dir), ticker)
    candles_df = None
    if price_file:
        try:
            candles_df = read_candles(price_file)
        except Exception as e:
            if not quiet:
                print(f"[WARN] failed to read price file {price_file}: {e}")
            candles_df = None

    # fallback_day_ts: try infer from filename date
    file_date = infer_date_from_filename(scored_path)
    fallback_ts = None
    if file_date:
        try:
            fallback_ts = int(pd.to_datetime(file_date, utc=True).timestamp())
        except Exception:
            fallback_ts = None

    mapped = map_headlines_to_candles(heads, candles_df, fallback_day_ts=fallback_ts, tolerance=match_tolerance)

    # compute rolling sentiment (on mapped rows, sorted by ts)
    df = mapped.sort_values("ts").reset_index(drop=True)
    if "sentiment" not in df.columns:
        df["sentiment"] = np.nan
    # rolling on sentiment; require min_periods=1 to produce values when available
    try:
        df[f"rolling_sentiment_{rolling_window}"] = df["sentiment"].rolling(window=rolling_window, min_periods=1).mean()
    except Exception:
        df[f"rolling_sentiment_{rolling_window}"] = df["sentiment"]

    # write out with columns: ts,iso_utc,price,n_headlines,sentiment,rolling_sentiment_W
    out_cols = ["ts","iso_utc","price","n_headlines","sentiment",f"rolling_sentiment_{rolling_window}"]
    out_df = df.loc[:, out_cols]
    out_df.to_csv(out_path, index=False)
    if not quiet:
        print(f"[OK] wrote {out_path} (rows={len(out_df)})")
    return out_path

def main():
    args = parse_args()
    hdr = Path(args.headlines_dir)
    prd = Path(args.prices_dir)
    outd = Path(args.out_dir)
    if args.scored:
        files = [Path(args.scored)]
    else:
        if not hdr.exists():
            print(f"[INFO] headlines dir {hdr} missing — nothing to do")
            return
        files = sorted(hdr.glob("*.json"))
        if not files:
            print(f"[INFO] no scored headline files in {hdr}")
            return
    for f in files:
        process_one_file(f, prd, outd, rolling_window=args.rolling_window, match_tolerance=args.match_tolerance, force=args.force, quiet=args.quiet)

if __name__ == "__main__":
    main()

