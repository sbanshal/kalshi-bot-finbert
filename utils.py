# utils.py
# Helpers for daily_check.py and signal_tester.py
# pip install spacy scikit-learn nltk
# (transformers used only in signal_tester if available)

import re
import json
from collections import Counter
from datetime import datetime
import os

# --- simple keyword extraction from title/description ---
def extract_keywords_and_polarity(title: str, description: str):
    """
    Returns (keywords:list[str], yes_implies_good_economy:bool)
    Heuristic:
      - extract capitalized tokens, nouns-like tokens (simple regex),
      - look for comparator words "above, below, increase, decrease, exceed"
      - decide polarity: if YES equals 'higher' (e.g., "Inflation > X") then yes_implies_good_economy=False
        else default True (YES implies good economy).
    """
    text = (title or "") + " " + (description or "")
    # token extraction (quick heuristic)
    tokens = re.findall(r"[A-Za-z0-9%./\-]+", text)
    tokens = [t for t in tokens if len(t) >= 2]
    # pick most common tokens (exclude stop-ish words)
    stop = set(["the","in","on","of","and","to","by","for","vs","vs.","vs"])
    tokens_clean = [t for t in tokens if t.lower() not in stop]
    candidate_counts = Counter(tokens_clean)
    # top tokens
    top = [t for t, _ in candidate_counts.most_common(12)]
    # heuristics for polarity
    text_l = text.lower()
    # if title suggests "Inflation >" or contains "above" "exceed" with an inflation-like token
    negative_indicators = ["inflation", "cpi", "consumer price", "deficit", "default", "recession", "higher", "increase"]
    comparator_patterns = ["above", "exceed", "greater than", ">", "rise", "higher than"]
    # If market YES means worse macro (inflation > X), then yes_implies_good_economy = False
    yes_good = True
    # naive rule: if text mentions "inflation" or "cpi" and contains comparator or ">" assume YES = more inflation => bad
    if any(tok in text_l for tok in ["inflation","cpi"]) and any(p in text_l for p in comparator_patterns):
        yes_good = False
    # if title mentions "unemployment" and comparator "<" (unemployment < X), YES is good
    if "unemployment" in text_l and ("<" in text or "less than" in text_l or "below" in text_l):
        yes_good = True
    # if title mentions "rate hike" or "raise rates", YES likely = hawkish -> event = higher rates (not good for some)
    # leave as default unless clear
    return top, yes_good

# --- confidence scoring ---
def compute_confidence(title, description, keywords, polarity, market_row):
    """
    Return confidence in [0,1] that the algorithm mapped keywords/polarity correctly.
    We use lightweight heuristics:
      - title clarity (presence of comparator symbols or words) -> up to 0.2
      - presence of economic tokens in keywords -> up to 0.4
      - metadata liquidity boost if provided in market_row -> up to 0.2
      - short-title boost -> up to 0.2
    """
    score = 0.0
    t = (title or "").lower()
    desc = (description or "").lower()
    # title clarity
    if any(x in t for x in [">", "<", "above", "below", "exceed", "less than", "greater than", "rise", "fall"]):
        score += 0.2
    # economic keywords
    econ_tokens = ["inflation","cpi","unemployment","gdp","fed","fed funds","fed funds rate","pce","retail sales","payroll"]
    keyword_hits = sum(1 for k in keywords if any(e in k.lower() for e in econ_tokens))
    score += min(0.4, 0.08 * keyword_hits)
    # liquidity / volume indicator from market_row if present
    try:
        vol = float(market_row.get("volume", 0) or 0)
    except:
        vol = 0
    if vol > 0:
        # small boost
        score += 0.1
        if vol > 1000:
            score += 0.05
    # short-title boost (clear concise titles)
    if len(t.split()) <= 8:
        score += 0.15
    # clamp
    return min(1.0, score)

# --- backfill placeholder (lightweight) ---
def backfill_market(market_id, keywords, days=180, out_dir="backfill"):
    """
    Light-weight backfill: create a directory and write a placeholder file.
    In full implementation this would:
      - fetch historical Kalshi price series for market_id
      - fetch historical headlines matching keywords
      - run a quick sentiment check
    Here we create a small manifest file so daily_check.py can record backfill completion.
    """
    os.makedirs(out_dir, exist_ok=True)
    fn = os.path.join(out_dir, f"{market_id}_manifest.json")
    manifest = {
        "market_id": market_id,
        "keywords": keywords,
        "backfill_days": days,
        "backfill_started_at": datetime.utcnow().isoformat(),
        "status": "placeholder_done"
    }
    with open(fn, "w") as f:
        json.dump(manifest, f, indent=2)
    return fn
