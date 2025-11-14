#!/usr/bin/env python3
"""
batch_score_headlines.py

Batch-run finbert_scoring.py over all JSON files in data/headlines/
and write outputs to data/headlines_scored/.

Usage:
  python batch_score_headlines.py           # default: 4 workers, skip existing
  python batch_score_headlines.py --workers 1 --force

Notes:
- This script shells out to `python finbert_scoring.py --in <in> --out <out> ...`
  so it will behave exactly like the scoring CLI you already ran.
- Adjust `FINBERT_SCRIPT` if you placed finbert_scoring.py somewhere else.
"""
import argparse
import subprocess
import sys
from pathlib import Path
from multiprocessing import Pool
import shlex

ROOT = Path(".").resolve()
HEADLINES_DIR = ROOT / "data" / "headlines"
OUT_DIR = ROOT / "data" / "headlines_scored"
FINBERT_SCRIPT = ROOT / "finbert_scoring.py"

def find_headline_files():
    if not HEADLINES_DIR.exists():
        return []
    return sorted(HEADLINES_DIR.glob("*.json"))

def make_outpath(inpath: Path) -> Path:
    return OUT_DIR / (inpath.stem + "_scored.json")

def score_one(args):
    inpath, outpath, batch_size, device, force = args
    # skip existing unless force
    if outpath.exists() and not force:
        return (str(inpath), "skipped", 0)
    cmd = [
        sys.executable,
        str(FINBERT_SCRIPT),
        "--in", str(inpath),
        "--out", str(outpath),
        "--batch_size", str(batch_size),
        "--device", device,
        "--force"
    ]
    # Hide model loading logs? we keep them for debugging
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        # optionally capture summary from stdout
        return (str(inpath), "done", proc.returncode, proc.stdout + proc.stderr)
    except subprocess.CalledProcessError as e:
        return (str(inpath), "error", e.returncode, e.stdout + e.stderr)
    except Exception as e:
        return (str(inpath), "error", -1, str(e))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4, help="number of parallel workers")
    p.add_argument("--batch-size", type=int, default=16, help="batch size for finbert_scoring")
    p.add_argument("--device", default="cpu", help="device to pass to finbert_scoring (cpu or cuda)")
    p.add_argument("--force", action="store_true", help="overwrite existing scored outputs")
    p.add_argument("--dry-run", action="store_true", help="print what would be done and exit")
    args = p.parse_args()

    if not FINBERT_SCRIPT.exists():
        print(f"ERROR: finbert script not found at {FINBERT_SCRIPT}", file=sys.stderr)
        sys.exit(2)

    files = find_headline_files()
    if not files:
        print(f"[INFO] No headline files found in {HEADLINES_DIR}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    for f in files:
        outp = make_outpath(f)
        tasks.append((f, outp, args.batch_size, args.device, args.force))

    if args.dry_run:
        for (f, outp, *_ ) in tasks:
            if outp.exists() and not args.force:
                print("SKIP (exists):", f, "->", outp)
            else:
                print("WILL SCORE: ", f, "->", outp)
        return

    # run sequentially if workers == 1 to preserve readable logs
    if args.workers <= 1:
        for t in tasks:
            inname, status, code, *rest = score_one(t)
            if status == "done":
                print(f"[OK]  {inname}")
            elif status == "skipped":
                print(f"[SKIP] {inname} (scored already)")
            else:
                print(f"[ERR]  {inname} (code={code})")
                print(rest[0] if rest else "")
    else:
        with Pool(processes=args.workers) as pool:
            results = pool.map(score_one, tasks)
        # report
        for res in results:
            inname = res[0]
            status = res[1]
            if status == "done":
                print(f"[OK]  {inname}")
            elif status == "skipped":
                print(f"[SKIP] {inname} (scored already)")
            else:
                code = res[2]
                out = res[3] if len(res) > 3 else ""
                print(f"[ERR]  {inname} (code={code})")
                if out:
                    print(out)

if __name__ == "__main__":
    main()
