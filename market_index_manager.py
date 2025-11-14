#!/usr/bin/env python3
"""
market_index_manager.py

Scans Kalshi via your existing discover_and_fetch.py CLI to build/update a
market index of relevant markets.

Usage:
  # One-shot update
  python market_index_manager.py --out data/market_index.json

  # Dry-run (prints counts, does not write)
  python market_index_manager.py --out data/market_index.json --dry-run

Notes:
- This wrapper expects discover_and_fetch.py to exist in the same repo and
  that it supports:
    --tag <Tag> --list_series
    --series_ticker <SERIES> --list_markets
  and prints valid JSON for those calls.
- The script is defensive about CLI output and will attempt to extract JSON
  blocks if discover_and_fetch prints other log lines.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Any

# --- CONFIGURE ---
# Tags to scan for candidate series (expand as needed)
SCAN_TAGS = [
    "Economics", "Inflation", "PCE", "Fed", "Jobs", "Employment", "Crypto",
    "Financials", "Policy", "Companies", "Markets"
]

# Category whitelist (keeps markets where category matches)
CATEGORY_WHITELIST = {
    "Economics", "Crypto", "Financials", "Politics", "World"
}

# Text blacklist (if title/tags contain any of these, skip market)
TEXT_BLACKLIST = {
    "Sport", "Football", "Baseball", "Soccer", "Hockey", "NFL", "NBA",
    "Oscars", "Grammy", "Movies", "Television", "Awards", "Rotten",
    "Streaming", "Concert"
}

# Where to read/write
DATA_DIR = Path("data")
OUT_INDEX = DATA_DIR / "market_index.json"
REMOVED_ARCHIVE = DATA_DIR / "market_index_removed.json"
BACKUP = DATA_DIR / "market_index.backup.json"

# CLI to call (relative path)
DISCOVER_CLI = "python discover_and_fetch.py"

# Maximum time to wait for subprocess (seconds)
SUBPROCESS_TIMEOUT = 60

# --------------------

def run_cli(cmd: List[str], timeout: int = SUBPROCESS_TIMEOUT) -> str:
    """Run a CLI command (list_series / list_markets) and return stdout as string.
       Returns empty string on failure."""
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=timeout)
        return out
    except subprocess.CalledProcessError as e:
        print(f"[WARN] command failed: {' '.join(cmd)} -> {e}", file=sys.stderr)
        return ""
    except Exception as e:
        print(f"[WARN] command exception: {' '.join(cmd)} -> {e}", file=sys.stderr)
        return ""


def extract_json_candidate(s: str) -> Optional[Any]:
    """Attempt to extract a JSON value from CLI output. Handles either a top-level
       list or object printed among logs by picking the first [ ... ] or { ... } block."""
    s = s.strip()
    if not s:
        return None
    # quick path: if s seems valid JSON
    try:
        return json.loads(s)
    except Exception:
        pass
    # find first bracketed JSON block
    start = s.find("[")
    end = s.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start:end+1])
        except Exception:
            pass
    # try curly object
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start:end+1])
        except Exception:
            pass
    return None


def list_series_for_tag(tag: str) -> List[Dict]:
    cmd = (DISCOVER_CLI + f" --tag \"{tag}\" --list_series").split()
    out = run_cli(cmd)
    j = extract_json_candidate(out)
    if isinstance(j, list):
        return j
    return []


def list_markets_for_series(series_ticker: str) -> List[Dict]:
    cmd = (DISCOVER_CLI + f" --series_ticker {series_ticker} --list_markets").split()
    out = run_cli(cmd)
    j = extract_json_candidate(out)
    if isinstance(j, list):
        return j
    return []


def market_is_relevant(m: Dict) -> bool:
    # check category whitelist first (if present)
    cat = (m.get("category") or "").strip()
    if cat and cat not in CATEGORY_WHITELIST:
        # category is present but not in whitelist -> skip
        return False
    # check tags/title for blacklist terms
    title = (m.get("title") or "").lower()
    tags = [t.lower() for t in (m.get("tags") or []) if isinstance(t, str)]
    for bad in TEXT_BLACKLIST:
        if bad.lower() in title:
            return False
        for t in tags:
            if bad.lower() in t:
                return False
    # otherwise keep (we're permissive if category absent)
    return True


def normalize_market_obj(series_ticker: str, market_obj: Dict) -> Dict:
    """Return a small normalized dict for the index."""
    return {
        "ticker": market_obj.get("ticker"),
        "title": market_obj.get("title"),
        "series": series_ticker,
        "category": market_obj.get("category"),
        "tags": market_obj.get("tags"),
        "contract_url": market_obj.get("contract_url"),
        "contract_terms_url": market_obj.get("contract_terms_url"),
        "fee_type": market_obj.get("fee_type"),
        "frequency": market_obj.get("frequency"),
        "last_seen": int(time.time())
    }


def load_existing_index(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    try:
        arr = json.loads(path.read_text())
        if isinstance(arr, list):
            return {m["ticker"]: m for m in arr if isinstance(m, dict) and m.get("ticker")}
    except Exception as e:
        print(f"[WARN] failed to read existing index: {e}", file=sys.stderr)
    return {}


def write_index(path: Path, markets_map: Dict[str, Dict]):
    arr = list(markets_map.values())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(arr, indent=2))


def main(args):
    print("[INFO] Starting market index manager")
    # load existing
    existing = load_existing_index(args.out)
    print(f"[INFO] existing markets in index: {len(existing)}")

    discovered_map: Dict[str, Dict] = {}

    # iterate tags -> series -> markets
    for tag in SCAN_TAGS:
        print(f"[INFO] scanning tag: {tag}")
        series_list = list_series_for_tag(tag)
        print(f"  found {len(series_list)} series for tag '{tag}'")
        for s in series_list:
            st = s.get("ticker")
            if not st:
                continue
            markets = list_markets_for_series(st)
            # some series_list entries may already be markets; handle that case too
            if not isinstance(markets, list):
                markets = []
            for m in markets:
                ticker = m.get("ticker")
                if not ticker:
                    continue
                # filter relevance
                if not market_is_relevant(m):
                    # skip but optionally note
                    continue
                normalized = normalize_market_obj(st, m)
                discovered_map[ticker] = normalized

    print(f"[INFO] discovered relevant markets: {len(discovered_map)}")

    # Merge with existing index:
    merged = dict(existing)  # copy
    # Add/update discovered markets
    for t, obj in discovered_map.items():
        if t in merged:
            # update last_seen but keep other fields (so we don't clobber manual edits)
            merged[t].update(obj)
            merged[t]["last_seen"] = int(time.time())
        else:
            merged[t] = obj

    # Detect removed markets: those in existing but not discovered on this run
    removed = {}
    for t, obj in list(merged.items()):
        if t not in discovered_map:
            # treat as removed/closed
            removed[t] = merged.pop(t)

    # Backup old index
    if args.out.exists():
        BACKUP.write_text(args.out.read_text())

    # Write outputs (unless dry-run)
    if args.dry_run:
        print("[DRY RUN] would write index with", len(merged), "markets; removed:", len(removed))
    else:
        print(f"[INFO] writing {len(merged)} markets to {args.out}")
        write_index(args.out, merged)
        if removed:
            print(f"[INFO] archiving {len(removed)} removed markets to {REMOVED_ARCHIVE}")
            # append to removed archive (keep history)
            prev = []
            if REMOVED_ARCHIVE.exists():
                try:
                    prev = json.loads(REMOVED_ARCHIVE.read_text())
                except Exception:
                    prev = []
            # attach timestamp
            ts = int(time.time())
            for k, v in removed.items():
                v["_removed_at"] = ts
                prev.append(v)
            REMOVED_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
            REMOVED_ARCHIVE.write_text(json.dumps(prev, indent=2))

    print("[INFO] done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(OUT_INDEX), help="path to output market index JSON (array)")
    p.add_argument("--dry-run", action="store_true", help="don't write files; just report")
    args = p.parse_args()
    main(args)
