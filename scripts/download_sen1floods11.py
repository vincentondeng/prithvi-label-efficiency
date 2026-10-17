"""Download the Sen1Floods11 hand-labeled Sentinel-2 chips and labels
listed in the split CSVs. Standard library only; safe to re-run."""
import argparse
import csv
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled"
SPLITS = ["train", "valid", "test", "bolivia"]


def chip_files(split_csv):
    """Yield (s2_name, label_name) for each row of a split CSV.
    The CSVs list Sentinel-1 files; the S2 file has the same name with S2Hand."""
    with open(split_csv) as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                yield row[0].strip().replace("S1Hand", "S2Hand"), row[1].strip()


def download(url, dest, retries=3):
    if dest.exists() and dest.stat().st_size > 0:
        return "skipped"
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as out:
                expected = int(r.headers.get("Content-Length", -1))
                got = 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    got += len(chunk)
            if expected >= 0 and got != expected:
                raise IOError(f"size mismatch: expected {expected}, got {got}")
            tmp.rename(dest)
            return "downloaded"
        except (urllib.error.URLError, IOError) as e:
            if attempt == retries:
                raise
            time.sleep(2 * attempt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits-dir", default="splits")
    p.add_argument("--root", default="data/sen1floods11")
    p.add_argument("--limit", type=int, default=None, help="max chips per split (for testing)")
    args = p.parse_args()

    root = Path(args.root)
    (root / "S2Hand").mkdir(parents=True, exist_ok=True)
    (root / "LabelHand").mkdir(parents=True, exist_ok=True)

    failed = []
    for split in SPLITS:
        pairs = list(chip_files(Path(args.splits_dir) / f"flood_{split}_data.csv"))
        if args.limit:
            pairs = pairs[: args.limit]
        for i, (s2, label) in enumerate(pairs, 1):
            for folder, name in (("S2Hand", s2), ("LabelHand", label)):
                try:
                    status = download(f"{BASE}/{folder}/{name}", root / folder / name)
                except Exception as e:
                    failed.append((folder, name, str(e)))
                    status = "FAILED"
                if status != "skipped":
                    print(f"[{split} {i}/{len(pairs)}] {folder}/{name}: {status}")
    print(f"\nDone. Failed: {len(failed)}")
    for item in failed:
        print("  ", item)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
