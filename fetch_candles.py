def fetch_candles(series_ticker: str,
                  market_ticker: str,
                  start_ts: int,
                  end_ts: int,
                  period_interval: int = 3600,
                  api_key: Optional[str] = None,
                  out_csv: Optional[str] = None,
                  include_raw: bool = False,
                  page_window_seconds: Optional[int] = 7 * 24 * 3600) -> List[Dict]:
    """
    Fetch candlesticks using GET /series/{series_ticker}/markets/{market_ticker}/candlesticks

    This enhanced version saves raw payloads to a debug JSON file and logs the first payload.
    """
    LOG.info("Fetching candles for %s / %s from %s to %s (period_interval=%s)",
             series_ticker, market_ticker, start_ts, end_ts, period_interval)

    all_candles: List[Dict] = []
    current_start = int(start_ts)
    window = int(page_window_seconds) if page_window_seconds else (end_ts - start_ts)
    headers = get_headers(api_key)
    raw_chunks = []
    first_chunk_logged = False

    while current_start < end_ts:
        current_end = min(current_start + window, end_ts)
        path = f"/series/{series_ticker}/markets/{market_ticker}/candlesticks"
        params = {
            "start_ts": int(current_start),
            "end_ts": int(current_end),
            "period_interval": int(period_interval),
        }
        LOG.debug("Requesting window %s -> %s", current_start, current_end)
        url = f"{BASE}{path}"
        resp = requests.get(url, params=params, headers=headers, timeout=60)
        try:
            resp.raise_for_status()
        except Exception:
            LOG.error("Failed to fetch candlesticks: %s %s", resp.status_code, resp.text)
            raise
        payload = resp.json()
        raw_chunks.append({"start": current_start, "end": current_end, "payload": payload})

        # Log the first payload for quick debugging
        if not first_chunk_logged:
            LOG.info("DEBUG: first raw payload snippet (partial): %s", json.dumps(payload if isinstance(payload, dict) else {"payload_type": type(payload).__name__}, ensure_ascii=False)[:1200])
            first_chunk_logged = True

        # find actual list in typical locations
        candidates = None
        for k in ("candlesticks", "candles", "data", "results", "items"):
            if isinstance(payload, dict) and k in payload:
                candidates = payload[k]
                break
        if candidates is None and isinstance(payload, list):
            candidates = payload
        if candidates is None:
            # No recognized list — store raw payload as-is to raw_chunks and continue
            LOG.warning("Unexpected candlesticks payload shape for window %s-%s; saving raw payload for inspection.", current_start, current_end)
            candidates = []

        for c in candidates:
            # Robust timestamp extraction: try several keys and ensure non-zero valid ts
            ts_candidates = [c.get(k) for k in ("timestamp", "ts", "time", "t", "start_ts", "unix_time") if isinstance(c, dict)]
            ts = None
            for t in ts_candidates:
                try:
                    if t is None:
                        continue
                    t_int = int(t)
                    if t_int > 1000000000:  # reasonable Unix seconds lower bound (~2001-09-09)
                        ts = t_int
                        break
                except Exception:
                    continue
            if ts is None:
                # fallback: attempt to parse iso strings in known fields
                for kf in ("time_iso", "iso", "time"):
                    v = c.get(kf) if isinstance(c, dict) else None
                    if isinstance(v, str):
                        try:
                            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
                            ts = int(dt.timestamp())
                            break
                        except Exception:
                            continue

            # extract price robustly
            price = extract_price_from_candle(c)

            # only append if timestamp & price are plausible
            if ts is None:
                LOG.debug("Skipping candle with missing timestamp: raw=%s", str(c)[:400])
                continue
            all_candles.append({"timestamp": ts, "price": price, "raw": c})

        # advance window
        current_start = current_end + 1
        time.sleep(0.05)

    # save raw_chunks to file for inspection if include_raw True
    if include_raw:
        safe_name = f"candles_raw_{market_ticker}_{start_ts}_{end_ts}.json".replace("/", "_")
        try:
            with open(safe_name, "w", encoding="utf-8") as fh:
                json.dump(raw_chunks, fh, ensure_ascii=False, indent=2)
            LOG.info("Saved raw API chunks to %s", safe_name)
        except Exception as e:
            LOG.warning("Failed to write raw chunks file: %s", e)

    # dedupe & sort by timestamp
    unique = {}
    for c in all_candles:
        # if price is None, keep but mark price as NaN (to preserve timeline); filter later when writing CSV
        unique[c["timestamp"]] = c
    normalized = [unique[k] for k in sorted(unique.keys())]

    # optionally write CSV
    if out_csv:
        import csv
        LOG.info("Writing CSV to %s", out_csv)
        with open(out_csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            hdr = ["timestamp", "price", "iso_utc"]
            if include_raw:
                hdr.append("raw_json")
            writer.writerow(hdr)
            for r in normalized:
                iso = datetime.fromtimestamp(r["timestamp"], tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                # skip rows with implausible timestamps
                if r["timestamp"] <= 1000000000:
                    LOG.debug("Skipping writing implausible timestamp row: %s", r)
                    continue
                row = [r["timestamp"], r["price"], iso]
                if include_raw:
                    row.append(json.dumps(r["raw"], ensure_ascii=False))
                writer.writerow(row)

    return normalized
