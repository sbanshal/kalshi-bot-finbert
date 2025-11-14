#!/usr/bin/env python3
"""
Enhanced sentiment_aggregator.py

Replaces/augments existing aggregator with robust price-file merge and ts inference:
- looks for price files in data/prices/ and data/
- attempts to parse timestamps in price CSVs (epoch sec, epoch ms, ISO)
- uses pd.merge_asof to match headlines to nearest price candle (on timestamp)
- if headline ts==0 / missing, infers ts from price file date-range midpoint,
  filename date, or file mtime (in that order)
- marks inferred timestamps with inferred_ts=True and inferred_from string
- outputs CSVs to data/sentiment/<TICKER>_sentiment.csv

CLI arguments:
  --headlines-file HEADLINES
  --prices-dir DATA_DIR (default: data/, also searches data/prices/)
  --out-dir OUT_DIR (default: data/sentiment/)
  --prefer-nearest (flag) if present, prefer nearest price candle; otherwise
                  use exact candle mapping when possible.
  --inferred-ts-marker COLUMN_NAME (default: inferred_ts)
  --dry-run (flag) to not write files, just report actions

Behavioral defaults:
  - If price file timestamp column detected, uses it for merging.
  - If price file has no timestamp column but filename contains dates,
    infers a midpoint timestamp (UTC midnight of midpoint day).
"""

from __future__ import annotations
import argparse
import logging
import os
import re
import glob
import json
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Tuple, Dict

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sentiment_aggregator")

# Common timestamp column candidates in price CSVs
TIMESTAMP_COL_CANDIDATES = ["ts", "timestamp", "time", "date", "datetime", "open_time"]


def find_price_files(ticker: str, search_dirs: List[str]) -> List[str]:
    """
    Find price files that appear to belong to `ticker` under provided directories.
    Exclude output folders like 'sentiment' and 'trades' to avoid reading generated files.
    """
    ticker_token = ticker.split('_')[0].split('-')[0]
    tokens_to_try = sorted(list({ticker, ticker_token}), key=lambda x: -len(x))
    matches = []
    exclude_dirs = {"sentiment", "trades", "__pycache__"}

    for d in search_dirs:
        if not d:
            continue
        for token in tokens_to_try:
            pattern = os.path.join(d, "**", f"*{re.escape(token)}*.csv")
            found = [p for p in glob.glob(pattern, recursive=True)
                     if not any(ex in p.split(os.sep) for ex in exclude_dirs)]
            if found:
                matches.extend(found)

    unique = sorted(set(matches), key=lambda x: os.path.getmtime(x), reverse=True)
    logger.debug("find_price_files(%s) -> tried tokens %s -> %d files", ticker, tokens_to_try, len(unique))
    return unique

def detect_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    for c in TIMESTAMP_COL_CANDIDATES:
        if c in df.columns:
            return c
        # case-insensitive search
    for col in df.columns:
        if col.lower() in TIMESTAMP_COL_CANDIDATES:
            return col
    return None


def parse_ts_series(s: pd.Series) -> Tuple[pd.Series, str]:
    """
    Try to parse a pandas Series into epoch seconds (int) and return it along with format used.
    Tries:
      - numeric: if large values -> epoch ms -> convert to seconds
      - ISO strings: pd.to_datetime(..., utc=True)
    Returns (series_seconds, format_hint)
    """
    if pd.api.types.is_numeric_dtype(s):
        # numeric -> could be seconds or milliseconds
        maxv = int(s.dropna().abs().max()) if not s.empty else 0
        if maxv > 3_000_000_000_000:  # ms with ms scale (e.g., 1690000000000)
            logger.debug("Parsing numeric timestamp as milliseconds")
            return ( (s.fillna(0).astype('float64') / 1000).astype('int64'), "epoch_ms" )
        elif maxv > 3_000_000_000:  # somewhere between sec and ms; be conservative
            # assume ms
            logger.debug("Parsing numeric timestamp guessed as milliseconds (mid-range)")
            return ( (s.fillna(0).astype('float64') / 1000).astype('int64'), "epoch_ms_guess" )
        else:
            logger.debug("Parsing numeric timestamp as seconds")
            return ( s.fillna(0).astype('int64'), "epoch_s" )
    else:
        # try parsing as datetime strings
        try:
            parsed = pd.to_datetime(s, utc=True, errors='coerce')
            # convert to epoch seconds
            epoch = (parsed.view('int64') // 10**9).astype('Int64')
            logger.debug("Parsed string timestamps via pd.to_datetime")
            return ( epoch.fillna(0).astype('Int64'), "iso" )
        except Exception as e:
            logger.warning("Failed to parse timestamp series: %s", e)
            return ( pd.Series([0]*len(s), dtype='Int64'), "failed" )


def filename_date_range_midpoint(path: str) -> Optional[int]:
    """
    Try to extract date(s) from filename like ..._YYYYMMDD_YYYYMMDD.csv or ..._YYYYMMDD.csv
    Return epoch seconds for midpoint (UTC midnight of midpoint day) or None.
    """
    fname = os.path.basename(path)
    # find all YYYYMMDD groups
    dates = re.findall(r"(20\d{2}[01]\d[0-3]\d)", fname)
    if not dates:
        return None
    try:
        dt_list = [datetime.strptime(d, "%Y%m%d").replace(tzinfo=timezone.utc) for d in dates]
        if len(dt_list) == 1:
            mid = dt_list[0].replace(hour=0, minute=0, second=0)
        else:
            a, b = dt_list[0], dt_list[-1]
            mid_dt = a + (b - a) / 2
            mid = mid_dt.replace(hour=0, minute=0, second=0)
        return int(mid.timestamp())
    except Exception:
        return None


def file_mtime_epoch(path: str) -> int:
    return int(os.path.getmtime(path))


def load_price_df(path: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Load a price CSV and return (df, price_ts_col_name).
    - Treat an all-zero ts column as missing and infer price_ts from filename or mtime.
    - Normalize 'price' -> price_close.
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        logger.error("Failed to read price CSV %s: %s", path, e)
        raise

    df = df.reset_index(drop=True)

    ts_col = detect_timestamp_column(df)
    price_ts = None

    if ts_col:
        # If numeric and effectively empty (sum==0), treat as missing
        if pd.api.types.is_numeric_dtype(df[ts_col]):
            if int(df[ts_col].fillna(0).abs().sum()) == 0:
                logger.info("Timestamp column %s in %s is all zeros/NaN -> treating as missing", ts_col, path)
                ts_col = None
            else:
                series, fmt = parse_ts_series(df[ts_col])
                df["price_ts"] = series.astype('Int64')
                price_ts = "price_ts"
        else:
            parsed = pd.to_datetime(df[ts_col], utc=True, errors='coerce')
            if parsed.notna().any():
                df["price_ts"] = (parsed.view('int64') // 10**9).astype('Int64')
                price_ts = "price_ts"
            else:
                ts_col = None

    # If still no price_ts, infer from filename midpoint or file mtime
    if "price_ts" not in df.columns:
        mid = filename_date_range_midpoint(path)
        if mid:
            df["price_ts"] = mid
            price_ts = "price_ts"
            logger.info("Inferred price_ts from filename midpoint for %s -> %d", path, mid)
        else:
            mtime = file_mtime_epoch(path)
            df["price_ts"] = mtime
            price_ts = "price_ts"
            logger.info("Using file mtime as price_ts for %s -> %d", path, mtime)

    # Normalize price -> price_close
    lower = {c.lower(): c for c in df.columns}
    if "price" in lower:
        df["price_close"] = df[lower["price"]]
    else:
        for alt in ("close", "last", "adj_close"):
            if alt in lower:
                df["price_close"] = df[lower[alt]]
                break
    if "price_close" not in df.columns:
        df["price_close"] = np.nan
        logger.debug("No price_close found for %s (left as NaN)", path)

    return df, price_ts

def infer_ts_for_headlines(headlines_df: pd.DataFrame, price_files: List[str]) -> pd.DataFrame:
    """
    For rows in headlines_df lacking ts or with ts==0, try to infer timestamps:
      - if any price file has date range in filename -> use midpoint epoch
      - else use file mtime
    Adds columns:
      - inferred_ts (bool)
      - inferred_from (str) one of "price_file_midpoint", "price_file_filename_date", "file_mtime", or ""
    """
    df = headlines_df.copy()
    if "ts" not in df.columns:
        df["ts"] = 0
    df["inferred_ts"] = False
    df["inferred_from"] = ""

    zero_mask = df["ts"].fillna(0).astype('int64') == 0
    if not zero_mask.any():
        return df

    # Try price files in order
    for pf in price_files:
        mid = filename_date_range_midpoint(pf)
        if mid:
            logger.info("Inferring timestamps from price filename midpoint %s -> %s", pf, mid)
            # set ts for all zero rows to midpoint (only if still zero)
            assign_mask = (df["ts"].fillna(0).astype('int64') == 0)
            df.loc[assign_mask, "ts"] = mid
            df.loc[assign_mask, "inferred_ts"] = True
            df.loc[assign_mask, "inferred_from"] = f"price_file_midpoint:{os.path.basename(pf)}"
            break
    # any still zero -> fallback to file mtime of first price file (if exists)
    remaining_zero = (df["ts"].fillna(0).astype('int64') == 0)
    if remaining_zero.any() and price_files:
        mtime = file_mtime_epoch(price_files[0])
        logger.info("Inferring remaining timestamps from price file mtime %s -> %s", price_files[0], mtime)
        df.loc[remaining_zero, "ts"] = mtime
        df.loc[remaining_zero, "inferred_ts"] = True
        df.loc[remaining_zero, "inferred_from"] = f"price_file_mtime:{os.path.basename(price_files[0])}"

    # final fallback: set to file mtime of headline file (if available in _meta)
    remaining_zero = (df["ts"].fillna(0).astype('int64') == 0)
    if remaining_zero.any():
        now_ts = int(datetime.now(timezone.utc).timestamp())
        logger.warning("Setting %d remaining headline rows ts to now (%s).", remaining_zero.sum(), now_ts)
        df.loc[remaining_zero, "ts"] = now_ts
        df.loc[remaining_zero, "inferred_ts"] = True
        df.loc[remaining_zero, "inferred_from"] = "now_fallback"
    return df


def merge_headlines_with_prices(headlines_df: pd.DataFrame,
                                price_df: pd.DataFrame,
                                price_ts_col: Optional[str],
                                prefer_nearest: bool = True) -> pd.DataFrame:
    """
    Merge headlines (with 'ts' epoch seconds) to price_df that contains 'price_ts' and price_close.
    Ensures no duplicate 'ts' columns and returns consistent column names.
    """
    h = headlines_df.copy()
    p = price_df.copy()

    if "ts" not in h.columns:
        raise ValueError("headlines_df must contain column 'ts'")

    # ensure canonical types
    h["ts"] = h["ts"].astype('int64')

    # ensure price_ts exists and is int
    if price_ts_col and price_ts_col in p.columns and price_ts_col != "price_ts":
        p = p.rename(columns={price_ts_col: "price_ts"})
    if "price_ts" not in p.columns:
        p["price_ts"] = np.arange(len(p)).astype('int64')
        logger.warning("price_df had no price_ts, created monotonic index.")

    p["price_ts"] = p["price_ts"].astype('int64')

    # Sort
    h_sorted = h.sort_values("ts").reset_index(drop=True)
    p_sorted = p.sort_values("price_ts").reset_index(drop=True)

    # Merge: use merge_asof backward and forward (if prefer_nearest) on distinct column names
    left = pd.merge_asof(h_sorted, p_sorted, left_on="ts", right_on="price_ts", direction="backward")
    if prefer_nearest:
        right = pd.merge_asof(h_sorted, p_sorted, left_on="ts", right_on="price_ts", direction="forward")
        chosen_rows = []
        for i in range(len(h_sorted)):
            lrow = left.iloc[i]
            rrow = right.iloc[i]
            h_ts = int(h_sorted.iloc[i]["ts"])
            l_pt = int(lrow.get("price_ts")) if not pd.isna(lrow.get("price_ts")) else np.nan
            r_pt = int(rrow.get("price_ts")) if not pd.isna(rrow.get("price_ts")) else np.nan
            ldiff = abs(h_ts - l_pt) if not np.isnan(l_pt) else np.inf
            rdiff = abs(r_pt - h_ts) if not np.isnan(r_pt) else np.inf
            chosen = lrow if ldiff <= rdiff else rrow
            chosen_rows.append(chosen)
        merged = pd.DataFrame(chosen_rows).reset_index(drop=True)
    else:
        merged = left

    # Remove duplicate 'ts' columns if any (keep headline 'ts')
    cols = []
    seen = set()
    for c in merged.columns:
        # if column name like 'ts_x' or 'ts_y' appear, map to 'ts' once
        base = c
        if c.endswith("_x") or c.endswith("_y"):
            base = c[:-2]
        if base == "ts":
            if "ts" not in seen:
                cols.append("ts")
                seen.add("ts")
            # skip additional ts aliases
            continue
        if c in seen:
            continue
        cols.append(c)
        seen.add(c)
    merged = merged.loc[:, [c for c in merged.columns if c in cols]]

    # Ensure canonical price columns exist
    for c in ["price_ts", "price_close", "price_open", "price_high", "price_low"]:
        if c not in merged.columns:
            merged[c] = np.nan

    # Final column ordering: keep original headline fields (except ts), then canonical ts and price cols
    headline_cols = [c for c in h.columns if c != "ts"]
    out_cols = headline_cols + ["ts", "inferred_ts", "inferred_from", "price_ts", "price_close", "price_open", "price_high", "price_low"]
    out_cols = [c for c in out_cols if c in merged.columns]
    merged = merged[out_cols]

    return merged


def save_sentiment_output(merged_df: pd.DataFrame, ticker: str, out_dir: str, dry_run: bool = False) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ticker}_sentiment.csv")
    if dry_run:
        logger.info("[dry-run] Would write %s rows -> %s", len(merged_df), out_path)
        return out_path
    merged_df.to_csv(out_path, index=False)
    logger.info("Wrote sentiment CSV %s (%d rows)", out_path, len(merged_df))
    return out_path


def infer_ticker_from_headlines_file(headlines_path: str) -> Optional[str]:
    """
    Infer a ticker/market id from headlines filename.

    Rules:
      - strip directory and extension
      - remove common prefix 'headlines_' if present
      - return the first token before '_' or '-' (so 'KXCPI_20251113.json' -> 'KXCPI')
      - fallback: whole basename without extension
    """
    fname = os.path.basename(headlines_path)
    base = os.path.splitext(fname)[0]
    # remove common prefix
    if base.startswith("headlines_"):
        base = base[len("headlines_"):]
    # token before first underscore or dash
    token = re.split(r"[_\-]", base)[0]
    token = token.strip()
    if token:
        return token
    return base


def run(headlines_file: str,
        prices_dirs: List[str] = None,
        out_dir: str = "data/sentiment",
        prefer_nearest: bool = True,
        inferred_ts_marker: str = "inferred_ts",
        dry_run: bool = False):
    if prices_dirs is None:
        prices_dirs = ["data/prices", "data"]

    logger.info("Loading headlines from %s", headlines_file)
    try:
        headlines = pd.read_json(headlines_file, lines=False)
    except ValueError:
        # try newline-delimited
        headlines = pd.read_json(headlines_file, lines=True)

    # normalize expected columns
    if "ts" not in headlines.columns:
        # if the headline JSON uses 'timestamp' or 'time' -> normalize
        for c in ["timestamp", "time", "date", "datetime"]:
            if c in headlines.columns:
                headlines["ts"] = headlines[c]
                break
    # convert ts to int if possible
    if "ts" in headlines.columns:
        try:
            headlines["ts"] = headlines["ts"].astype('Int64')
        except Exception:
            # attempt parse
            headlines["ts"] = pd.to_datetime(headlines["ts"], utc=True, errors='coerce').view('int64') // 10**9
            headlines["ts"] = headlines["ts"].fillna(0).astype('Int64')
    else:
        headlines["ts"] = 0

    # infer ticker
    ticker = infer_ticker_from_headlines_file(headlines_file) or "UNKNOWN"
    logger.info("Inferred ticker/market: %s", ticker)

    # find price files
    price_files = find_price_files(ticker, prices_dirs)
    if not price_files:
        # try fuzzy search single token
        logger.warning("No price files found for ticker %s under %s", ticker, prices_dirs)
    else:
        logger.info("Found %d price files; using top candidate: %s", len(price_files), price_files[0])

    # infer ts for any missing headline ts
    headlines = infer_ts_for_headlines(headlines, price_files)

    # attempt merge using the first matching price file that can be loaded and parsed
    merged = None
    used_price_file = None
    for pf in price_files:
        try:
            price_df, price_ts_col = load_price_df(pf)
        except Exception:
            logger.exception("Skipping unreadable price file %s", pf)
            continue
        try:
            merged = merge_headlines_with_prices(headlines, price_df, price_ts_col, prefer_nearest=prefer_nearest)
            used_price_file = pf
            logger.info("Merged headlines with price file %s", pf)
            break
        except Exception:
            logger.exception("Failed to merge with price file %s; trying next", pf)
            continue

    if merged is None:
        # No price file merge succeeded; produce a fallback sentiment file with inferred ts
        logger.warning("No successful price merge; producing fallback sentiment CSV with inferred ts only.")
        merged = headlines.copy()
        # add placeholder price columns so downstream consumers won't break
        merged["price_close"] = np.nan
        merged["price_open"] = np.nan
        merged["price_high"] = np.nan
        merged["price_low"] = np.nan

    # make sure inferred_ts marker exists
    if inferred_ts_marker != "inferred_ts" and "inferred_ts" in merged.columns:
        merged[inferred_ts_marker] = merged["inferred_ts"]
    elif inferred_ts_marker not in merged.columns:
        merged[inferred_ts_marker] = merged.get("inferred_ts", False)

    # Save
    saved_path = save_sentiment_output(merged, ticker, out_dir, dry_run=dry_run)
    logger.info("Done. Output: %s (used price file: %s)", saved_path, used_price_file)
    return saved_path


def parse_args():
    p = argparse.ArgumentParser(description="Sentiment aggregator with price-file merging + ts inference")
    p.add_argument("headlines_file", help="Path to headlines JSON (or CSV) to aggregate (e.g. data/headlines_<TICKER>_YYYYMMDD.json)")
    p.add_argument("--prices-dir", nargs="+", default=["data/prices", "data"], help="Directories to search for price CSVs")
    p.add_argument("--out-dir", default="data/sentiment", help="Output directory for aggregated sentiment CSVs")
    p.add_argument("--prefer-nearest", action="store_true", help="Prefer nearest price candle (forward/backward) when merging")
    p.add_argument("--inferred-ts-marker", default="inferred_ts", help="Column name to mark inferred timestamps")
    p.add_argument("--dry-run", action="store_true", help="Do not write output files; only simulate")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.headlines_file, prices_dirs=args.prices_dir, out_dir=args.out_dir,
        prefer_nearest=args.prefer_nearest, inferred_ts_marker=args.inferred_ts_marker, dry_run=args.dry_run)


# ----------------------
# Minimal unit-test skeleton (expand for your CI)
# ----------------------
def _test_filename_date_range_midpoint():
    # single date
    p = "prices_KXCPI-25NOV-T0.3_20251101.csv"
    mid = filename_date_range_midpoint(p)
    assert mid is not None and isinstance(mid, int)
    # range
    p2 = "prices_KXCPI-25NOV-T0.3_20251101_20251113.csv"
    mid2 = filename_date_range_midpoint(p2)
    assert mid2 is not None and isinstance(mid2, int)
    assert mid2 >= mid

if __name__ == "__main__" and False:
    # quick manual test
    _test_filename_date_range_midpoint()
