# fetch_prices.py
# Usage examples:
#   python fetch_prices.py --market_id SP500_above_5000 --start 2025-11-01T00:00:00Z --end 2025-11-13T23:59:00Z --out prices.csv
#   python fetch_prices.py --list_markets   # lists markets (first page)
#
# Notes:
# - Uses Kalshi public endpoints. No auth required for market discovery in many cases.
# - API shape can differ; this script is robust to a couple of common JSON shapes and falls
#   back to plausible keys. If an endpoint requires auth for your account, add API key logic.
# - Granularity: try "1m", "1h" etc. depending on the market's supported series.
# - Output: timestamp in ISO format and price (YES contract price in decimal 0..1).

import argparse, requests, csv, sys, time
from datetime import datetime, timezone
import pandas as pd
import os

BASE = "https://api.elections.kalshi.com/trade-api/v2"

def list_markets(limit=50):
    url = f"{BASE}/markets"
    params = {"limit": limit}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    j = r.json()
    return j.get("markets", []), j.get("cursor")

def fetch_market_series(market_id, start=None, end=None, granularity="1m", limit=10000):
    """
    Returns a list of dicts with at least keys ('timestamp', 'price') if possible.
    The Kalshi API may return different keys; this function tries several common shapes:
      - {"series": [{"timestamp": "...", "yes_price": 0.52}, ...]}
      - {"candles": [{"time":"...", "open_yes":..., "close_yes":...}, ...]}
      - {"data": [{"t":"...", "p":...}, ...]}
    """
    url = f"{BASE}/markets/{market_id}/series"
    params = {"granularity": granularity, "limit": limit}
    if start: params["start"] = start
    if end: params["end"] = end
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()

    # try common keys
    candidates = []
    if isinstance(j, dict):
        # keys with lists
        for key in ("series", "candles", "data", "points", "prices"):
            if key in j and isinstance(j[key], list):
                candidates = j[key]
                break
        # some endpoints return markets as single object with 'market_series'
        if not candidates:
            for k,v in j.items():
                if isinstance(v, list) and v and any(isinstance(x, dict) for x in v):
                    candidates = v
                    break
    elif isinstance(j, list):
        candidates = j

    if not candidates:
        # fallback - try to parse nested structures
        raise ValueError("No series-like array found in response. Response keys: " + ", ".join(list(j.keys())))

    out = []
    for rec in candidates:
        # many possible schemas — try to extract a timestamp and YES price
        ts = None
        price = None
        # timestamp possible keys
        for k in ("timestamp","time","t","date","datetime","ts"):
            if k in rec:
                ts = rec[k]; break
        # yes-price possible keys
        for k in ("yes_price","yes","close_yes","close","p","price","value"):
            if k in rec:
                price = rec[k]; break
        # some candles have open/close - prefer 'close_yes' then 'close'
        if price is None:
            if "open_yes" in rec and "close_yes" in rec:
                price = rec["close_yes"]
            elif "open" in rec and "close" in rec:
                price = rec["close"]
        # if ts is numeric epoch (seconds)
        if ts is not None and isinstance(ts, (int,float)):
            # assume epoch seconds
            try:
                ts = datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
            except Exception:
                ts = str(ts)
        # normalise if ts contains trailing microseconds etc.
        if ts is None:
            continue
        out.append({"timestamp": str(ts), "price": float(price) if price is not None else None, "raw": rec})
    # filter out None prices
    out = [x for x in out if x["price"] is not None]
    # sort by timestamp
    try:
        out = sorted(out, key=lambda r: r["timestamp"])
    except Exception:
        pass
    return out

def save_csv(rows, out_fn):
    # normalize timestamps to ISO (attempt)
    with open(out_fn, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp","price"])
        for r in rows:
            w.writerow([r["timestamp"], r["price"]])
    print("Wrote", out_fn)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list_markets", action="store_true", help="List first page of markets")
    p.add_argument("--market_id", help="Market id (e.g., Inflation_above_3 or market uuid)")
    p.add_argument("--start", help="ISO start timestamp e.g. 2025-11-01T00:00:00Z")
    p.add_argument("--end", help="ISO end timestamp e.g. 2025-11-13T23:59:00Z")
    p.add_argument("--granularity", default="1m", help="granularity (1m, 1h, 1d)")
    p.add_argument("--out", default=None, help="Output CSV filename")
    args = p.parse_args()

    if args.list_markets:
        ms, cur = list_markets(limit=100)
        print(f"Found {len(ms)} markets (first page):")
        for m in ms:
            print(m.get("id") or m.get("slug") or m.get("ticker_name") or m.get("title"))
        return

    if not args.market_id:
        print("Please provide --market_id or use --list_markets")
        sys.exit(1)

    print("Fetching series for", args.market_id)
    # fetch
    rows = fetch_market_series(args.market_id, start=args.start, end=args.end, granularity=args.granularity)
    if not rows:
        print("No rows returned for market", args.market_id)
        sys.exit(1)

    out_fn = args.out or f"prices_{args.market_id}_{args.start or 'start'}_{args.end or 'end'}.csv"
    save_csv(rows, out_fn)

if __name__ == "__main__":
    main()
