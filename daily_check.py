# daily_check.py
# pip install requests pandas spacy sklearn transformers

import requests, json, time
import pandas as pd
from datetime import datetime, timedelta
from utils import extract_keywords_and_polarity, compute_confidence, backfill_market

KALSHI_MARKETS_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"
SNAPSHOT_CSV = "data/markets_snapshot.csv"
MAPPING_JSON = "market_mapping.json"
AUTO_ONBOARD_THRESHOLD = 0.85

def fetch_all_markets(limit=200):
    markets = []
    cursor = None
    params = {"limit": limit}
    while True:
        if cursor:
            params['cursor'] = cursor
        r = requests.get(KALSHI_MARKETS_URL, params=params); r.raise_for_status()
        j = r.json()
        markets.extend(j.get("markets", []))
        cursor = j.get("cursor")
        if not cursor:
            break
    return pd.json_normalize(markets)

def load_snapshot():
    try:
        return pd.read_csv(SNAPSHOT_CSV)
    except FileNotFoundError:
        return pd.DataFrame()

def save_snapshot(df):
    df.to_csv(SNAPSHOT_CSV, index=False)

def load_mappings():
    try:
        return json.load(open(MAPPING_JSON))
    except FileNotFoundError:
        return {}

def save_mappings(m):
    json.dump(m, open(MAPPING_JSON, "w"), indent=2)

def run():
    print("Fetching markets...")
    df = fetch_all_markets()
    df_econ = df[df['category'].str.contains("econom", case=False, na=False)].copy()
    prev = load_snapshot()
    prev_ids = set(prev['id'].astype(str)) if not prev.empty else set()
    new_markets = []
    for _, row in df_econ.iterrows():
        mid = str(row['id'])
        if mid not in prev_ids:
            new_markets.append(row)
    if not new_markets:
        print("No new economics markets.")
        save_snapshot(df_econ)
        return

    mappings = load_mappings()
    for row in new_markets:
        mid = str(row['id'])
        title = row.get('title','')
        desc = row.get('description', '')
        profile = {'id': mid, 'title': title, 'description': desc, 'created_at': datetime.utcnow().isoformat()}
        # auto-extract
        kw, polarity = extract_keywords_and_polarity(title, desc)
        confidence = compute_confidence(title, desc, kw, polarity, row)
        entry = {
            "market_id": mid,
            "title": title,
            "keywords": kw,
            "yes_implies_good_economy": polarity,
            "confidence": confidence,
            "status": "auto_onboarded_paper" if confidence >= AUTO_ONBOARD_THRESHOLD else "needs_review",
            "created_at": profile['created_at']
        }
        mappings[mid] = entry
        print(f"New market {mid} -> conf {confidence:.2f} status {entry['status']}")
        # auto backfill if auto_onboarded_paper
        if entry['status'] == 'auto_onboarded_paper':
            backfill_market(entry['market_id'], kw)
            entry['backfill_done_at'] = datetime.utcnow().isoformat()
    save_mappings(mappings)
    save_snapshot(df_econ)
    # optionally notify via webhook / slack
    print("Done.")

if __name__ == "__main__":
    run()
