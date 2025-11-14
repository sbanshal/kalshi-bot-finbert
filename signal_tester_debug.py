# signal_tester_debug.py
import json, os
import numpy as np
from transformers import pipeline

print("HF pipeline building (FinBERT)...")
pipe = pipeline("sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert")
print("Pipeline ready.\n")

MAPPING_JSON = "market_mapping.json"
with open(MAPPING_JSON) as f:
    mapping = json.load(f)

def score_headline(text):
    out = pipe(text[:512])[0]
    lab, sc = out["label"].lower(), float(out["score"])
    if "positive" in lab:
        return sc
    elif "negative" in lab:
        return -sc
    return 0.0

test_headlines = [
    "CPI falls 0.3% month-on-month, weakest reading since 2021",
    "Unemployment rate unexpectedly drops to 3.5% as payrolls rise",
    "Fed officials signal possible rate hike amid stronger labor market",
    "Retail sales jump 1.2%, beating estimates",
    "Inflation higher than expected; CPI YoY rises to 3.5%"
]

for mid, info in mapping.items():
    kws = [k.lower() for k in info.get("keywords", [])]
    matches, scores = [], []
    for h in test_headlines:
        if any(k in h.lower() for k in kws):
            s = score_headline(h)
            matches.append((h, s))
            scores.append(s)
    if not matches:
        continue
    avg = float(np.mean(scores))
    yes_good = bool(info.get("yes_implies_good_economy", True))
    adj = avg if yes_good else -avg
    action = "BUY_YES" if adj > 0.25 else "BUY_NO" if adj < -0.25 else "HOLD"

    print(f"\nMarket: {mid}")
    print("Title:", info.get("title", ""))
    for txt, s in matches:
        print(f"  {s:+.3f}  {txt}")
    print(f"Raw mean = {avg:+.3f}, Adjusted = {adj:+.3f}, Action = {action}")













