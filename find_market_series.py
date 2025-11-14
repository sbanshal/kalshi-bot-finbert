#!/usr/bin/env python3
"""
find_market_series.py

Usage:
  python find_market_series.py KXPROLLS

It will try common tags and list_series -> list_markets to find which series contains the target market ticker.
"""
import sys
import subprocess
import json

TAGS = ["Economics","Jobs","Employment","Crypto","Financials","Companies","Politics","Inflation","PCE","Fed","Markets","Entertainment"]

def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
        return out
    except subprocess.CalledProcessError:
        return ""

def list_series_for_tag(tag):
    cmd = ["python", "discover_and_fetch.py", "--tag", tag, "--list_series"]
    out = run_cmd(cmd)
    try:
        return json.loads(out)
    except Exception:
        # sometimes the CLI prints multiple things; try to brute force a JSON substring
        try:
            return json.loads(out[out.find("["):out.rfind("]")+1])
        except Exception:
            return []

def list_markets_for_series(series_ticker):
    cmd = ["python", "discover_and_fetch.py", "--series_ticker", series_ticker, "--list_markets"]
    out = run_cmd(cmd)
    try:
        return json.loads(out)
    except Exception:
        try:
            return json.loads(out[out.find("["):out.rfind("]")+1])
        except Exception:
            return []

def find_series_for_market(market_ticker):
    found = []
    for tag in TAGS:
        series = list_series_for_tag(tag)
        for s in series:
            st = s.get("ticker")
            if not st:
                continue
            markets = list_markets_for_series(st)
            for m in markets:
                if isinstance(m, dict) and m.get("ticker") == market_ticker:
                    found.append({"series": st, "series_title": s.get("title"), "market": m.get("ticker"), "market_title": m.get("title")})
    return found

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python find_market_series.py <MARKET_TICKER>")
        sys.exit(1)
    target = sys.argv[1].strip()
    print("Searching for", target, "across common tags...")
    results = find_series_for_market(target)
    if not results:
        print("No series found for", target, "— try running discover_and_fetch.py --list_series --tag <Tag> manually and inspect output.")
    else:
        print("Found the following candidate series -> market mappings:")
        print(json.dumps(results, indent=2))
