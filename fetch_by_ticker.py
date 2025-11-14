# fetch_by_ticker.py
# Usage:
#  python fetch_by_ticker.py --series_ticker KXCPI --match "KXCPI-25NOV-T0.3" --start 2025-10-30T00:00:00Z --end 2025-11-13T23:59:00Z --out out.csv
#
# This script:
#  - lists markets for the series_ticker (via /markets?series_ticker=...)
#  - finds the market whose ticker/slug/title contains the match string
#  - prints the whole market dict (so you can inspect id/ticker/slug fields)
#  - uses the market's 'id' field (or other suitable field) to call /markets/{id}/series and save CSV

import argparse, requests, csv, os, sys
BASE = "https://api.elections.kalshi.com/trade-api/v2"
HEADERS = {}
if os.environ.get("KALSHI_API_KEY"):
    HEADERS["Authorization"] = "Bearer " + os.environ.get("KALSHI_API_KEY")

def list_markets_for_series(ticker, limit=200):
    params = {"series_ticker": ticker, "limit": limit}
    r = requests.get(f"{BASE}/markets", params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json().get("markets", [])

def fetch_series_by_market_id(market_id, start=None, end=None, granularity="1h", limit=10000):
    params = {"granularity": granularity, "limit": limit}
    if start: params["start"] = start
    if end: params["end"] = end
    url = f"{BASE}/markets/{market_id}/series"
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    # return response object for inspection on error
    return r

def save_csv_rows(rows, out_fn):
    with open(out_fn, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp","price"])
        for r in rows:
            w.writerow([r["timestamp"], r["price"]])
    print("Wrote", out_fn)

def normalize_rows_from_response(j):
    candidates = None
    if isinstance(j, dict):
        for key in ("series","candles","data","points","prices"):
            if key in j and isinstance(j[key], list):
                candidates = j[key]; break
        if candidates is None:
            for k,v in j.items():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    candidates = v; break
    elif isinstance(j, list):
        candidates = j
    if not candidates:
        return []
    rows = []
    from datetime import datetime, timezone
    for rec in candidates:
        ts = None; price = None
        for k in ("timestamp","time","t","date","datetime","ts"):
            if k in rec:
                ts = rec[k]; break
        for k in ("yes_price","yes","close_yes","close","p","price","value"):
            if k in rec:
                price = rec[k]; break
        if price is None:
            if "close_yes" in rec: price = rec["close_yes"]
            elif "close" in rec: price = rec["close"]
        if ts is None:
            continue
        # normalize numeric timestamps
        try:
            if isinstance(ts, (int,float)):
                ts = datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
            else:
                ts = str(ts)
        except Exception:
            ts = str(ts)
        if price is None:
            continue
        rows.append({"timestamp": ts, "price": float(price)})
    rows = sorted(rows, key=lambda r: r["timestamp"])
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--series_ticker", required=True)
    p.add_argument("--match", required=True, help="substring to match against market ticker/slug/title")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--granularity", default="1h")
    p.add_argument("--out", default="prices_out.csv")
    args = p.parse_args()

    print("Listing markets for series:", args.series_ticker)
    markets = list_markets_for_series(args.series_ticker, limit=500)
    if not markets:
        print("No markets returned for series:", args.series_ticker)
        sys.exit(1)

    # try to find a match
    found = None
    for m in markets:
        # check id, ticker, slug, title lowercased
        for key in ("id","ticker","slug","title"):
            val = m.get(key) or ""
            if args.match.lower() in str(val).lower():
                found = m
                break
        if found:
            break

    if not found:
        print("No market matched that substring. Available markets (first 20):")
        for m in markets[:20]:
            print("  - id:", m.get("id"), "ticker:", m.get("ticker"), "slug:", m.get("slug"), "title:", m.get("title"))
        sys.exit(1)

    print("Matched market (full object):")
    import json
    print(json.dumps(found, indent=2))
    market_id = found.get("id") or found.get("ticker") or found.get("slug")
    print("Using market_id:", market_id)

    # fetch series using that market id
    print("Fetching series for market id:", market_id)
    r = fetch_series_by_market_id(market_id, start=args.start, end=args.end, granularity=args.granularity)
    if r.status_code != 200:
        print("Series fetch failed with HTTP", r.status_code)
        print("Response:", r.text[:1000])
        sys.exit(1)
    j = r.json()
    rows = normalize_rows_from_response(j)
    if not rows:
        print("No rows parsed from response; dumping JSON keys:", list(j.keys()))
        print("Sample response (truncated):", str(j)[:1000])
        sys.exit(1)
    save_csv_rows(rows, args.out)

if __name__ == "__main__":
    main()
