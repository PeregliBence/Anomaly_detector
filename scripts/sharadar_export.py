"""Pull every Sharadar table to local parquet. Run while the subscription is active.

Usage:
    python scripts/sharadar_export.py --list    # what the key is entitled to
    python scripts/sharadar_export.py           # export anything missing
    python scripts/sharadar_export.py --verify  # coverage report of what is on disk
    python scripts/sharadar_export.py --force   # re-download everything

Needs SHARADAR_API_KEY in .env, copied from the sharadar.com account — not the
Nasdaq Data Link key, which only reaches the free sample.

Only cancel the subscription once --verify reports "ok" on every row.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import setup_logging
from src.sources import sharadar

logger = setup_logging("INFO")


def _print_verify() -> None:
    report = sharadar.verify()
    print("\n===== Local Sharadar snapshot =====")
    print(report.to_string(index=False))
    bad = report[report["status"] != "ok"]["table"].tolist()
    if bad:
        print(f"\nNot complete: {bad}. Do not cancel the subscription yet.")
    else:
        print("\nEvery table passes its coverage check. Safe to cancel the subscription.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sharadar bulk export")
    parser.add_argument("--list", action="store_true", help="show the entitlement manifest")
    parser.add_argument("--verify", action="store_true", help="coverage report only")
    parser.add_argument("--force", action="store_true", help="re-download existing tables")
    parser.add_argument("--history", default="full", choices=["full", "10y", "5y"])
    args = parser.parse_args()

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    if args.list:
        manifest = sharadar.list_files(args.history)
        cols = ["table", "name", "sizeLabel", "canDownload", "modified"]
        print(manifest[[c for c in cols if c in manifest]].to_string(index=False))
        return

    if args.verify:
        _print_verify()
        return

    sharadar.export_all(history=args.history, force=args.force)
    _print_verify()


if __name__ == "__main__":
    main()
