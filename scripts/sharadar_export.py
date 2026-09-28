"""Pull every Sharadar table to local parquet. Run while the subscription is active.

Usage:
    python scripts/sharadar_export.py           # export anything missing
    python scripts/sharadar_export.py --verify  # report what is on disk
    python scripts/sharadar_export.py --force   # re-download everything

The subscription is one month; the retention term Sharadar granted in writing
is two years. Run this, check the verification table, and only cancel once
every row reads "ok".
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Sharadar bulk export")
    parser.add_argument("--verify", action="store_true", help="report local snapshot only")
    parser.add_argument("--force", action="store_true", help="re-download existing tables")
    parser.add_argument("--table", default="", help="export a single table")
    args = parser.parse_args()

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    if args.verify:
        report = sharadar.verify()
        print("\n===== Local Sharadar snapshot =====")
        print(report.to_string(index=False))
        missing = report[report["status"] == "MISSING"]["table"].tolist()
        print(f"\n{len(missing)} missing: {missing}" if missing
              else "\nAll tables present. Safe to cancel the subscription.")
        return

    if args.table:
        sharadar.export_table(args.table.upper(), force=args.force)
    else:
        sharadar.export_all(force=args.force)

    report = sharadar.verify()
    print("\n===== Export complete =====")
    print(report.to_string(index=False))

    missing = report[report["status"] == "MISSING"]["table"].tolist()
    if missing:
        print(f"\nStill missing: {missing}. Re-run before cancelling.")
    else:
        print("\nAll tables present. Safe to cancel the subscription.")


if __name__ == "__main__":
    main()
