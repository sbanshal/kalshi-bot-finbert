#!/usr/bin/env python3
"""
finbert_scoring.py

Batch-score headlines using a FinBERT model from Hugging Face.

Usage:
  python finbert_scoring.py --in headlines_20251113.json --out headlines_scored_20251113.json

Outputs: same JSON object/array structure as input with added fields per headline:
  - sentiment: float (model_pos_prob - model_neg_prob) approx in [-1,1]
  - sentiment_probs: {"pos": float, "neg": float, "neutral": float} when available

Options:
  --model           HF model id (default: ProsusAI/finbert)
  --batch_size      inference batch size (default: 16)
  --device          'cpu' or 'cuda' or integer CUDA index (default: 'cpu')
  --max_length      max token length for tokenizer (default: 256)
  --num_workers     number of worker processes for tokenization (not used for torch inference) (default: 1)
  --force           overwrite output if exists
  --quiet           minimal output
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# defensive imports (transformers + torch)
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging
    from torch.nn.functional import softmax
except Exception as e:
    print("ERROR: This script requires 'transformers' and 'torch'. Install with:")
    print("  pip install transformers torch")
    raise

hf_logging.set_verbosity_error()

DEFAULT_MODEL = "ProsusAI/finbert"

def load_input(path: Path) -> Dict[str, Any]:
    txt = path.read_text(encoding="utf-8")
    try:
        data = json.loads(txt)
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON from {path}: {e}")
    # normalize shapes: accept list of headlines or {"headlines": [...]} or {"items": [...]}
    if isinstance(data, list):
        return {"headlines": data}
    if isinstance(data, dict):
        # try common keys
        if "headlines" in data and isinstance(data["headlines"], list):
            return data
        # if top-level has 'items' or 'results' fallback
        for k in ("items", "results", "articles"):
            if k in data and isinstance(data[k], list):
                return {"headlines": data[k]}
    # last resort: if dict has string keys that are objects with 'text' assume those are headlines map
    # but we will error out if shape unknown
    raise RuntimeError("Input JSON must be an array or contain a top-level 'headlines' list (or 'items'/'results').")

def extract_text_from_headline(item: Dict[str, Any]) -> str:
    # common keys: 'title', 'headline', 'text'
    for key in ("text", "headline", "title"):
        if key in item and isinstance(item[key], str) and item[key].strip():
            return item[key].strip()
    # fallback: join all string values
    txts = [v.strip() for v in item.values() if isinstance(v, str) and v.strip()]
    return " ".join(txts)[:1000] if txts else ""

def make_batches(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def compute_sentiment_logits(logits_tensor, label_map=None):
    # logits_tensor: torch.Tensor shape (batch, num_labels)
    probs = softmax(logits_tensor, dim=-1).cpu().tolist()
    # label_map: list of labels in model order (if available)
    return probs

def infer_batch_texts(texts: List[str], tokenizer, model, device, max_length: int):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    for k in list(enc.keys()):
        enc[k] = enc[k].to(device)
    with torch.no_grad():
        out = model(**enc)
        logits = out.logits
        probs = softmax(logits, dim=-1).cpu().tolist()  # list of list
    return probs

def map_probs_to_sentiment(probs: List[float], label_order: List[str]) -> Dict[str, Any]:
    """
    Map probability vector to a numeric sentiment:
      sentiment = prob_pos - prob_neg (range approx -1..1)
    label_order: list like ['negative','neutral','positive'] or other
    """
    mapping = {"pos": 0.0, "neg": 0.0, "neutral": 0.0}
    # Attempt to find the indices for pos/neg/neutral
    label_lower = [l.lower() for l in label_order]
    for i, l in enumerate(label_lower):
        if "pos" in l or "positive" in l:
            mapping["pos"] = probs[i]
        elif "neg" in l or "negative" in l:
            mapping["neg"] = probs[i]
        elif "neu" in l or "neutral" in l:
            mapping["neutral"] = probs[i]
        else:
            # unknown label, try to heuristically map by extremes later
            pass
    # fallback: if no mapping found, attempt heuristic: assume 3-label model: [neg, neutral, pos]
    if sum(mapping.values()) == 0.0 and len(probs) == 3:
        mapping["neg"], mapping["neutral"], mapping["pos"] = probs[0], probs[1], probs[2]
    # final fallback: if single-dim or binary model
    if sum(mapping.values()) == 0.0:
        # if binary model (2 items) assume [neg,pos]
        if len(probs) == 2:
            mapping["neg"], mapping["pos"] = probs[0], probs[1]
        else:
            # assign pos = max prob, neg = min prob, neutral = 1 - (pos+neg) clipped
            mx = max(probs); mn = min(probs)
            mapping["pos"] = mx
            mapping["neg"] = mn
            mapping["neutral"] = max(0.0, 1.0 - (mx + mn))
    sentiment = mapping["pos"] - mapping["neg"]
    # clamp for safety
    sentiment = max(-1.0, min(1.0, sentiment))
    return {"sentiment": float(sentiment), "sentiment_probs": mapping}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inpath", required=True, help="input headlines JSON")
    p.add_argument("--out", dest="outpath", required=True, help="output scored JSON")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"HuggingFace model id (default: {DEFAULT_MODEL})")
    p.add_argument("--batch_size", type=int, default=16, help="inference batch size")
    p.add_argument("--device", default="cpu", help="'cpu' or 'cuda' or integer index")
    p.add_argument("--max_length", type=int, default=256, help="max tokens per headline")
    p.add_argument("--num_workers", type=int, default=1, help="tokenization workers (not used for torch inference)")
    p.add_argument("--force", action="store_true", help="overwrite existing output file")
    p.add_argument("--quiet", action="store_true", help="minimal logging")
    args = p.parse_args()

    inpath = Path(args.inpath)
    outpath = Path(args.outpath)
    if outpath.exists() and not args.force:
        print(f"ERROR: {outpath} already exists. Use --force to overwrite.", file=sys.stderr)
        sys.exit(1)
    if not inpath.exists():
        print(f"ERROR: input file {inpath} not found", file=sys.stderr)
        sys.exit(1)

    # device resolution
    if args.device.isdigit():
        device = f"cuda:{args.device}"
    else:
        device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; falling back to CPU")
        device = "cpu"

    device_torch = torch.device(device)

    if not args.quiet:
        print(f"[INFO] Loading model {args.model} on device {device_torch} ...")

    # load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(device_torch)
    model.eval()

    # obtain label order if available (some models expose config.id2label)
    label_order = []
    try:
        cfg = model.config
        if hasattr(cfg, "id2label") and cfg.id2label:
            # id2label may be a dict of ints to label strings
            id2label = cfg.id2label
            label_order = [id2label[str(i)] if str(i) in id2label else id2label[i] for i in sorted(id2label.keys(), key=lambda x: int(x))]
    except Exception:
        label_order = []

    if not label_order:
        # fallback: some configs expose labels differently
        try:
            if hasattr(model.config, "label2id") and model.config.label2id:
                # invert label2id into order by id
                l2i = model.config.label2id
                label_order = [None] * len(l2i)
                for lab, idx in l2i.items():
                    label_order[idx] = lab
        except Exception:
            label_order = []

    # fallback default label order common for FinBERT: ['negative','neutral','positive']
    if not label_order:
        label_order = ["negative", "neutral", "positive"]

    if not args.quiet:
        print(f"[INFO] Label order assumed: {label_order}")

    # load input
    data = load_input(inpath)
    headlines = data["headlines"]
    if not isinstance(headlines, list):
        print("ERROR: 'headlines' must be a list", file=sys.stderr)
        sys.exit(1)
    total = len(headlines)
    if not args.quiet:
        print(f"[INFO] Found {total} headlines to score")

    # prepare results container
    out_items = []
    # preserve original top-level structure
    top_level = {k:v for k,v in data.items() if k != "headlines"}

    # prepare text list and mapping indices
    texts = []
    idx_to_item = []
    for i, item in enumerate(headlines):
        text = extract_text_from_headline(item)
        if not text:
            # skip empty text but keep placeholder
            texts.append("")
            idx_to_item.append(i)
        else:
            texts.append(text)
            idx_to_item.append(i)

    # batch inference
    bs = max(1, args.batch_size)
    scored_count = 0
    # We'll produce a new array of headline objects with sentiment fields merged into original item
    out_headlines = list(headlines)  # copy original

    for batch_texts in make_batches(list(enumerate(texts)), bs):
        batch_indices = [idx for idx, _ in batch_texts]
        batch_text_values = [txt for _, txt in batch_texts]
        # handle entirely-empty batch
        if all((not t) for t in batch_text_values):
            # set neutral 0 sentiment for these
            for bi in batch_indices:
                original_idx = bi
                item = dict(out_headlines[original_idx]) if isinstance(out_headlines[original_idx], dict) else {"text": out_headlines[original_idx]}
                item.setdefault("sentiment", 0.0)
                item.setdefault("sentiment_probs", {"pos": 0.0, "neg": 0.0, "neutral": 1.0})
                out_headlines[original_idx] = item
                scored_count += 1
            continue

        try:
            probs_batch = infer_batch_texts(batch_text_values, tokenizer, model, device_torch, max_length=args.max_length)
        except Exception as e:
            print(f"[ERROR] inference failed on batch: {e}", file=sys.stderr)
            # fallback: mark as neutral
            for bi in batch_indices:
                original_idx = bi
                item = dict(out_headlines[original_idx]) if isinstance(out_headlines[original_idx], dict) else {"text": out_headlines[original_idx]}
                item.setdefault("sentiment", 0.0)
                item.setdefault("sentiment_probs", {"pos": 0.0, "neg": 0.0, "neutral": 1.0})
                out_headlines[original_idx] = item
                scored_count += 1
            continue

        for local_i, probs in enumerate(probs_batch):
            original_idx = batch_indices[local_i]
            # map probs -> sentiment using label_order
            mapped = map_probs_to_sentiment(probs, label_order)
            item = dict(out_headlines[original_idx]) if isinstance(out_headlines[original_idx], dict) else {"text": out_headlines[original_idx]}
            item["sentiment"] = mapped["sentiment"]
            item["sentiment_probs"] = mapped["sentiment_probs"]
            out_headlines[original_idx] = item
            scored_count += 1

        if not args.quiet:
            print(f"[INFO] Scored {scored_count}/{total}", end="\r")

    if not args.quiet:
        print()

    # attach results to top-level and write
    out_obj = dict(top_level)
    out_obj["headlines"] = out_headlines

    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")

    # summary
    sentiments = [h.get("sentiment") for h in out_headlines if isinstance(h, dict) and "sentiment" in h]
    n_scored = len(sentiments)
    mean_sent = float(sum(sentiments)/n_scored) if n_scored else 0.0
    if not args.quiet:
        print(f"[INFO] Wrote {outpath} — scored: {n_scored}/{total}, mean sentiment: {mean_sent:.4f}")

if __name__ == "__main__":
    main()

