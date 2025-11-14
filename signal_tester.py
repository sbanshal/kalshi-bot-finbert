# signal_tester.py
# Usage:
#   python signal_tester.py
#
# It reads market_mapping.json (in current folder), then prompts for test headlines or uses built-in examples.

import json
import os
import numpy as np
from collections import defaultdict
from datetime import datetime
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# Load mapping (your mapping file)
MAPPING_JSON = "market_mapping.json"
with open(MAPPING_JSON, "r") as f:
    mapping = json.load(f)

# simple sentiment fallback lexicon (if FinBERT not present)
POS_LEX = ["beat","beats","surge","rise","drop below expectations","below expectations","better than","decrease","decline"]
NEG_LEX = ["miss","missed","beat expectations?","worse than","higher than","increase","spike","soars","jumps","raise","hike","higher"]

def lex_score(text):
    t = text.lower()
    score = 0.0
    for p in POS_LEX:
        if p in t: score += 0.6
    for n in NEG_LEX:
        if n in t: score -= 0.6
    # normalize
    score = max(-1.0, min(1.0, score))
    return score

# If HF available, build a finbert pipeline (note: model download ~ few hundred MB)
def build_hf_pipeline():
    if not HF_AVAILABLE:
        return None
    try:
        # Use a finance-tuned sentiment model if available (may need internet)
        pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert")
        return pipe
    except Exception as e:
        print("HF pipeline build failed:", e)
        return None

HF_PIPE = build_hf_pipeline()

def score_headline(text):
    """
    Return score in [-1,1] where positive means 'good for economy' by default.
    If HF pipeline exists, use it; else use lex_score.
    """
    if HF_PIPE is not None:
        try:
            out = HF_PIPE(text[:512])
            # output example: [{'label':'positive','score':0.99}]
            lab = out[0]['label'].lower()
            sc = float(out[0]['score'])
            if 'positive' in lab:
                return min(1.0, sc)
            elif 'negative' in lab:
                return -min(1.0, sc)
            else:
                # neutral -> 0
                return 0.0
        except Exception:
            return lex_score(text)
    else:
        return lex_score(text)

# Map headlines to markets by keyword matching
def headlines_to_market_scores(headlines, mapping):
    """
    headlines: list[str] of (timestamped) headlines; here we ignore timestamps for simplicity
    mapping: dict from market_id -> {keywords, yes_implies_good_economy, ...}
    returns: dict market_id -> aggregated sentiment (mean of matched headline scores)
    """
    market_scores = {}
    market_counts = {}
    for mid, info in mapping.items():
        kws = [k.lower() for k in info.get("keywords", [])]
        matched_scores = []
        for h in headlines:
            text = h.strip()
            t_low = text.lower()
            # match if any keyword is in headline
            if any(k in t_low for k in kws):
                s = score_headline(text)
                matched_scores.append(s)
        if matched_scores:
            avg = float(np.mean(matched_scores))
            # apply polarity inversion if YES implies bad economy (mapping stores yes_implies_good_economy)
            yes_good = bool(info.get("yes_implies_good_economy", True))
            adjusted = avg if yes_good else -avg
            market_scores[mid] = {"raw": avg, "adjusted": adjusted, "count": len(matched_scores), "title": info.get("title","")}
    return market_scores

def signal_from_adjusted(adj, open_thresh=0.25, close_thresh=0.15):
    """
    Convert adjusted sentiment to trade action:
      adj > open_thresh -> BUY YES (long YES)
      adj < -open_thresh -> BUY NO (long NO)
      between -> HOLD
    """
    if adj > open_thresh:
        return "BUY_YES"
    if adj < -open_thresh:
        return "BUY_NO"
    return "HOLD"

if __name__ == "__main__":
    # Example test headlines. Replace or load from file.
    test_headlines = [
        "CPI falls 0.3% month-on-month, weakest reading since 2021",
        "Unemployment rate unexpectedly drops to 3.5% as payrolls rise",
        "Fed officials signal possible rate hike amid stronger labor market",
        "Retail sales jump 1.2%, beating estimates",
        "Inflation higher than expected; CPI YoY rises to 3.5%"
    ]

    print("HF available:", HF_PIPE is not None)
    print(f"Loaded {len(mapping)} mapped markets from {MAPPING_JSON}")
    scores = headlines_to_market_scores(test_headlines, mapping)

    if not scores:
        print("No test headlines matched mapping keywords. Consider expanding keywords for markets.")
    else:
        print("\n=== Market signals ===")
        for mid, d in scores.items():
            action = signal_from_adjusted(d["adjusted"])
            print(f"\nMarket: {mid}")
            print("Title:", d.get("title",""))
            print("Matched headlines:", d["count"])
            print(f"Raw sentiment (mean) = {d['raw']:.3f}")
            print(f"Adjusted sentiment = {d['adjusted']:.3f}")
            print("Action:", action)
