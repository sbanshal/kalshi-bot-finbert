# news_ingestor.py
"""
Usage:
  from news_ingestor import get_headlines_for_markets, fetch_headlines_newsapi
  mapping = json.load(open("market_mapping.json"))
  headlines_by_market = get_headlines_for_markets(mapping, hours_back=24)
"""

import os
import time
import json
import requests
import feedparser
from datetime import datetime, timedelta
from dateutil import parser as dateparser

NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")

NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"
# small blacklist to remove junk
STOP_SOURCES = set(["Twitter", "X", "Unknown"])

def fetch_headlines_newsapi(q, from_dt, to_dt, page=1, page_size=100):
    """
    Query NewsAPI with q (string). Returns list of articles (title, publishedAt, url, source).
    """
    key = NEWSAPI_KEY
    if not key:
        raise RuntimeError("NEWSAPI_KEY not set in env. Set export NEWSAPI_KEY=...")
    results = []
    params = {
        "q": q,
        "from": from_dt.isoformat(),
        "to": to_dt.isoformat(),
        "pageSize": page_size,
        "page": page,
        "sortBy": "relevancy",
        "language": "en",
        "apiKey": key,
    }
    r = requests.get(NEWSAPI_ENDPOINT, params=params, timeout=15)
    r.raise_for_status()
    j = r.json()
    for a in j.get("articles", []):
        src = a.get("source", {}).get("name", "")
        if src in STOP_SOURCES:
            continue
        results.append({
            "title": a.get("title") or "",
            "description": a.get("description") or "",
            "publishedAt": a.get("publishedAt"),
            "url": a.get("url"),
            "source": src
        })
    return results

def fetch_headlines_rss(feed_url):
    """
    Fetch headlines from an RSS feed (feedparser). Returns list of items.
    """
    out = []
    try:
        d = feedparser.parse(feed_url)
        for e in d.entries:
            pub = None
            try:
                pub = dateparser.parse(e.get("published") or e.get("updated") or "")
            except Exception:
                pub = datetime.utcnow()
            out.append({
                "title": e.get("title",""),
                "description": e.get("summary",""),
                "publishedAt": pub.isoformat() if hasattr(pub, "isoformat") else datetime.utcnow().isoformat(),
                "url": e.get("link",""),
                "source": d.feed.get("title","rss")
            })
    except Exception:
        pass
    return out

def dedupe_articles(articles):
    seen = set()
    out = []
    for a in articles:
        key = (a.get("title","").strip().lower(), a.get("publishedAt","")[:19])
        if key in seen: 
            continue
        seen.add(key)
        out.append(a)
    return out

def get_headlines_for_markets(mapping, hours_back=24, sources_rss=None):
    """
    mapping: dict loaded from market_mapping.json
    hours_back: look back window
    sources_rss: list of rss feed urls to fall back to (e.g., ['https://www.reuters.com/rssFeed/worldNews'])
    Returns: dict {market_id: [article_dicts ...]}
    """
    end = datetime.utcnow()
    start = end - timedelta(hours=hours_back)
    result = {}
    # For each market, create a keyword query string for NewsAPI (joined by OR)
    for mid, info in mapping.items():
        kws = [k for k in info.get("keywords", []) if k and len(k) > 1]
        if not kws:
            continue
        # newsapi query: exact phrases quoted
        q = " OR ".join([f'"{k}"' for k in kws[:5]])
        articles = []
        # try newsapi (fast, relevant)
        try:
            articles = fetch_headlines_newsapi(q, start, end)
        except Exception:
            # fallback to RSS scraping of provided feeds or some defaults
            articles = []
            rss_feeds = sources_rss or [
                "https://www.reuters.com/finance/economy/rss",
                "https://www.reuters.com/markets/us/rss",
                "https://www.bloomberg.com/feed/podcast/news",
            ]
            for feed in rss_feeds:
                feed_items = fetch_headlines_rss(feed)
                # keep items that match any keyword
                for it in feed_items:
                    t = (it.get("title","") + " " + it.get("description","")).lower()
                    if any(k.lower() in t for k in kws):
                        articles.append(it)
        # normalize timestamps and basic fields
        normalized = []
        for a in articles:
            t = a.get("publishedAt") or a.get("published_at") or a.get("published")
            try:
                dt = dateparser.parse(t)
            except Exception:
                dt = datetime.utcnow()
            normalized.append({
                "title": a.get("title",""),
                "description": a.get("description",""),
                "publishedAt": dt.isoformat(),
                "url": a.get("url",""),
                "source": a.get("source","")
            })
        normalized = dedupe_articles(normalized)
        result[mid] = normalized
        # small sleep to respect rate limits
        time.sleep(0.2)
    return result

def save_headlines(result, outdir="headlines"):
    os.makedirs(outdir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fn = os.path.join(outdir, f"headlines_{ts}.json")
    with open(fn, "w") as f:
        json.dump(result, f, indent=2)
    return fn

if __name__ == "__main__":
    import sys
    mapping = json.load(open("market_mapping.json"))
    print("Fetching headlines (24h)...")
    res = get_headlines_for_markets(mapping, hours_back=24)
    p = save_headlines(res)
    print("Saved to", p)
    # quick summary
    for mid, arr in res.items():
        print(mid, len(arr))
