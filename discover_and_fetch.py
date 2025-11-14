#!/usr/bin/env python3
"""
discover_and_fetch.py

- Discover series by tag:   --tag "Inflation" --list_series
- List markets for series:  --series_ticker KXCPI --list_markets
- Fetch candlesticks:       --series_ticker KXCPI --market_ticker KXCPI-25NOV-T0.3 --fetch_candles --start 2025-01-01 --end 2025-02-01

Writes a CSV with columns: timestamp, price, iso_utc (and optionally raw_json if requested).

Env var: KALSHI_API_KEY (optional). You can also pass --api_key on CLI.
"""

import os
import sys
import time
import json
import argparse
import logging
from typing import Optional, Dict, List, Any
import requests
from datetime import datetime, timezone, timedelta

BASE = "https://api.elections.kalshi.com/trade-api/v2"
LOG = logging.getLogger("discover_and_fetch")


def setup_logging(level=logging.INFO):
    h = logging.StreamHandler(sys.stdout)
    fmt = "[%(levelname)s] %(message)s"
    h.setFormatter(logging.Formatter(fmt))
    LOG.addHandler(h)
    LOG.setLevel(level)


def parse_date_or_ts(v: str) -> int:
    """
    Accept either an integer unix timestamp (seconds) or a YYYY-MM-DD date.
    Returns unix seconds (int), UTC midnight for YYYY-MM-DD.
    """
    if v is None:
        raise ValueError("date string is required")
    v = str(v)
    if v.isdigit():
        return int(v)
    # try several common formats
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            dt = datetime.strptime(v, fmt)
            # assume naive times are UTC
            dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except Exception:
            continue
    # try ISO parse fallback
    try:
        dt = datetime.fromisoformat(v)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return int(dt.timestamp())
    except Exception:
        raise ValueError(f"Could not parse date/timestamp: {v}")


def get_headers(api_key: Optional[str]) -> Dict[str, str]:
    key = api_key or os.environ.get("KALSHI_API_KEY")
    headers = {"Accept": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def request_get(path: str, params: Dict = None, api_key: Optional[str] = None, timeout: int = 30) -> Any:
    url = f"{BASE}{path}"
    headers = get_headers(api_key)
    LOG.debug("GET %s %s", url, params or "")
    resp = requests.get(url, params=params or {}, headers=headers, timeout=timeout)
    try:
        resp.raise_for_status()
    except Exception as e:
        LOG.error("Request failed: %s - %s", resp.status_code, resp.text)
        raise
    return resp.json()


def discover_series_by_tag(tag: str, api_key: Optional[str] = None) -> List[Dict]:
    """
    Attempt to find series matching a tag. Endpoint assumed: /series?tag={tag}
    Returns list of series dicts (raw).
    """
    LOG.info("Discovering series for tag: %s", tag)
    path = "/series"
    params = {"tag": tag, "limit": 200}
    data = request_get(path, params=params, api_key=api_key)
    # attempt to extract list in common shapes
    for k in ("series", "data", "results"):
        if isinstance(data, dict) and k in data:
            return data[k]
    if isinstance(data, list):
        return data
    # fallback: return whole payload as single item
    return [data]


def list_markets_for_series(series_ticker: str, api_key: Optional[str] = None) -> List[Dict]:
    """
    List markets for a series. Endpoint assumed: /series/{series_ticker}/markets
    """
    LOG.info("Listing markets for series: %s", series_ticker)
    path = f"/series/{series_ticker}/markets"
    params = {"limit": 500}
    data = request_get(path, params=params, api_key=api_key)
    for k in ("markets", "data", "results"):
        if isinstance(data, dict) and k in data:
            return data[k]
    if isinstance(data, list):
        return data
    return [data]


def extract_price_from_candle(c: Dict) -> Optional[float]:
    """
    Choose a sensible price field from a candlestick dict.
    Prefer 'close' then 'price' then mid point of (bid,ask) or (open+close)/2.
    """
    for key in ("close", "price", "last", "mid"):
        if key in c and c[key] is not None:
            try:
                return float(c[key])
            except Exception:
                pass
    # try open and close
    if "open" in c and "close" in c:
        try:
            return (float(c["open"]) + float(c["close"])) / 2.0
        except Exception:
            pass
    # if nothing found, return None
    return None


def fetch_candles(series_ticker: str,
                  market_ticker: str,
                  start_ts: int,
                  end_ts: int,
                  period_interval: int = 3600,
                  api_key: Optional[str] = None,
                  out_csv: Optional[str] = None,
                  include_raw: bool = False,
                  page_window_seconds: Optional[int] = 7 * 24 * 3600) -> List[Dict]:
    """
    Fetch candlesticks using GET /series/{series_ticker}/markets/{market_ticker}/candlesticks

    For large ranges, this function pages by window (default 7 days) to avoid server-side limits.
    Returns normalized list of {"timestamp": int, "price": float, "raw": {...}} sorted by timestamp ascending.
    If out_csv provided, writes CSV file.
    """
    LOG.info("Fetching candles for %s / %s from %s to %s (period_interval=%s)",
             series_ticker, market_ticker, start_ts, end_ts, period_interval)

    all_candles: List[Dict] = []
    current_start = int(start_ts)
    window = int(page_window_seconds) if page_window_seconds else (end_ts - start_ts)
    headers = get_headers(api_key)

    while current_start < end_ts:
        current_end = min(current_start + window, end_ts)
        path = f"/series/{series_ticker}/markets/{market_ticker}/candlesticks"
        params = {
            "start_ts": int(current_start),
            "end_ts": int(current_end),
            "period_interval": int(period_interval),
        }
        LOG.debug("Requesting window %s -> %s", current_start, current_end)
        url = f"{BASE}{path}"
        resp = requests.get(url, params=params, headers=headers, timeout=60)
        try:
            resp.raise_for_status()
        except Exception:
            LOG.error("Failed to fetch candlesticks: %s %s", resp.status_code, resp.text)
            raise
        payload = resp.json()
        # find actual list in typical locations
        candidates = None
        for k in ("candlesticks", "candles", "data", "results", "items"):
            if isinstance(payload, dict) and k in payload:
                candidates = payload[k]
                break
        if candidates is None and isinstance(payload, list):
            candidates = payload
        if candidates is None:
            LOG.warning("Unexpected candlesticks payload shape, storing raw payload chunk")
            candidates = [payload]

        for c in candidates:
            ts = int(c.get("timestamp") or c.get("ts") or c.get("time") or c.get("t") or 0)
            price = extract_price_from_candle(c)
            all_candles.append({"timestamp": ts, "price": price, "raw": c})

        # advance window
        current_start = current_end + 1
        # polite sleep to avoid rate limits
        time.sleep(0.05)

    # dedupe & sort by timestamp
    unique = {}
    for c in all_candles:
        unique[c["timestamp"]] = c  # last write wins
    normalized = [unique[k] for k in sorted(unique.keys())]

    # optionally write CSV
    if out_csv:
        import csv
        LOG.info("Writing CSV to %s", out_csv)
        with open(out_csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            hdr = ["timestamp", "price", "iso_utc"]
            if include_raw:
                hdr.append("raw_json")
            writer.writerow(hdr)
            for r in normalized:
                iso = datetime.fromtimestamp(r["timestamp"], tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                row = [r["timestamp"], r["price"], iso]
                if include_raw:
                    row.append(json.dumps(r["raw"], ensure_ascii=False))
                writer.writerow(row)

    return normalized


def cli():
    p = argparse.ArgumentParser(description="Discover Kalshi series/markets and fetch candlesticks.")
    p.add_argument("--tag", help="Tag to search series by (e.g. Inflation)")
    p.add_argument("--list_series", action="store_true", help="List series for provided --tag")
    p.add_argument("--series_ticker", help="Series ticker (e.g. KXCPI)")
    p.add_argument("--list_markets", action="store_true", help="List markets for provided --series_ticker")
    p.add_argument("--market_ticker", help="Market ticker (e.g. KXCPI-25NOV-T0.3)")
    p.add_argument("--fetch_candles", action="store_true", help="Fetch candlesticks for series+market")
    p.add_argument("--start", help="Start date or unix ts. Formats: YYYY-MM-DD or unix seconds", default=None)
    p.add_argument("--end", help="End date or unix ts. Formats: YYYY-MM-DD or unix seconds", default=None)
    p.add_argument("--period_interval", type=int, default=3600, help="candlestick period interval in seconds (default 3600)")
    p.add_argument("--api_key", help="API key (fallback to KALSHI_API_KEY env var if omitted)")
    p.add_argument("--out_csv", help="Path to write CSV (default constructed from market & dates)", default=None)
    p.add_argument("--include_raw", action="store_true", help="Include raw JSON in CSV")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    if args.list_series:
        if not args.tag:
            LOG.error("--list_series requires --tag")
            sys.exit(2)
        series = discover_series_by_tag(args.tag, api_key=args.api_key)
        print(json.dumps(series, indent=2, ensure_ascii=False))
        return

    if args.list_markets:
        if not args.series_ticker:
            LOG.error("--list_markets requires --series_ticker")
            sys.exit(2)
        markets = list_markets_for_series(args.series_ticker, api_key=args.api_key)
        print(json.dumps(markets, indent=2, ensure_ascii=False))
        return

    if args.fetch_candles:
        if not (args.series_ticker and args.market_ticker):
            LOG.error("--fetch_candles requires --series_ticker and --market_ticker")
            sys.exit(2)
        if not (args.start and args.end):
            LOG.error("--fetch_candles requires --start and --end")
            sys.exit(2)
        start_ts = parse_date_or_ts(args.start)
        end_ts = parse_date_or_ts(args.end)
        if start_ts >= end_ts:
            LOG.error("start must be < end")
            sys.exit(2)
        # default out_csv naming
        if args.out_csv:
            out_csv = args.out_csv
        else:
            s_iso = datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime("%Y%m%d")
            e_iso = datetime.fromtimestamp(end_ts, tz=timezone.utc).strftime("%Y%m%d")
            out_csv = f"prices_{args.market_ticker}_{s_iso}_{e_iso}.csv"
        candles = fetch_candles(args.series_ticker,
                                args.market_ticker,
                                start_ts,
                                end_ts,
                                period_interval=args.period_interval,
                                api_key=args.api_key,
                                out_csv=out_csv,
                                include_raw=args.include_raw)
        LOG.info("Fetched %d candle rows", len(candles))
        return

    p.print_help()


if __name__ == "__main__":
    cli()
