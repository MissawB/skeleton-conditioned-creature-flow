"""Builds the Articulation-XL creature cache, one shard per row group of the Parquet mirror (resumable)."""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.axl.build import build_split

def parse_range(text: str):
    start, end = text.split(":")
    return range(int(start), int(end))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--out", default="data/cache/axl")
    parser.add_argument("--points", type=int, default=8192)
    parser.add_argument("--row-groups", type=parse_range, default=None, help="start:end, e.g. 0:10")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    start = time.time()
    last = [start]

    def log(message):
        now = time.time()
        print(f"{message} ({now - last[0]:.0f} s, total {now - start:.0f} s)", flush=True)
        last[0] = now

    build_split(args.split, args.out, num_points=args.points, row_groups=args.row_groups, workers=args.workers, log=log)

if __name__ == "__main__":
    main()
