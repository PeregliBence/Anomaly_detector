"""Sharadar bulk export via Sharadar's own API.

Sharadar sells directly and serves paid data from api.sharadar.com with a key
issued on the sharadar.com account. A Nasdaq Data Link key only reaches the
free sample (the Dow 30 over a few months of 2018), which is easy to mistake
for a real export — hence the coverage checks in `verify`.

The subscription is deliberately short: one month, during which every table
is pulled in full and kept locally. Sharadar granted a two-year retention term
for this research in writing, so the historical study runs off the local
snapshot rather than a live subscription. Everything is exported, including
tables with no immediate use: a table not pulled during that month cannot be
fetched later without re-subscribing, and disk is cheap.

The original zip archives are kept alongside the parquet conversions. They are
the licensed snapshot itself; the parquet files are derived and rebuildable.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src.config import DATA_DIR, SHARADAR_KEY, require

logger = logging.getLogger(__name__)

API = "https://api.sharadar.com/v1.0"
SNAPSHOT_DIR = DATA_DIR / "sharadar"
RAW_DIR = SNAPSHOT_DIR / "raw"

PURPOSE = {
    "stocks": "Equity prices, daily, including delisted securities (SEP)",
    "funds": "Fund prices, daily (SFP)",
    "fundamentals": "Core fundamentals, all dimensions incl. as-reported (SF1)",
    "insiders": "Insider transactions, Form 4 (SF2)",
    "holdings": "Institutional holdings detail, 13F (SF3)",
    "holdings_ticker": "Institutional holdings aggregated by ticker (SF3A)",
    "holdings_investor": "Institutional holdings aggregated by investor (SF3B)",
    "tickers": "Security metadata incl. delisting status",
    "actions": "Corporate actions",
    "events": "Material corporate events incl. earnings dates",
    "daily": "Daily valuation metrics: market cap, EV, P/E",
    "metrics": "Price-based metrics",
    "sp500": "S&P 500 constituent changes",
    "descriptions": "Column definitions for every table",
}

# Minimum coverage that separates a real export from the free sample. The
# sample holds ~30 tickers over a few months, so any of these thresholds
# catches it.
COVERAGE = {
    "stocks": {"min_tickers": 5000, "min_years": 15},
    "daily": {"min_tickers": 5000, "min_years": 15},
    "fundamentals": {"min_tickers": 5000, "min_years": 15,
                     "dimensions": {"ARQ", "ARY", "ART", "MRQ", "MRY", "MRT"}},
    "insiders": {"min_tickers": 3000, "min_years": 10},
    "holdings": {"min_tickers": 3000, "min_years": 10},
    "tickers": {"min_rows": 20000},
    "events": {"min_tickers": 3000, "min_years": 15},
    "actions": {"min_tickers": 3000, "min_years": 15},
}

_URL_FIELDS = ("url", "downloadUrl", "download_url", "presignedUrl", "signedUrl", "href", "link")


# ---------------------------------------------------------------------------
# Listing and download
# ---------------------------------------------------------------------------

def list_files(history: str = "full") -> pd.DataFrame:
    """The bulk manifest for one history tier, with download entitlement."""
    key = require(SHARADAR_KEY, "SHARADAR_API_KEY")
    resp = requests.get(f"{API}/bulk", params={"api_key": key}, timeout=120)
    resp.raise_for_status()
    payload = resp.json()

    if not payload.get("signedIn"):
        logger.warning("Sharadar reports signedIn=false: the key is not a "
                       "subscriber key. Copy it from your sharadar.com account.")

    items = pd.DataFrame(payload.get("items", []))
    if items.empty:
        return items
    return items[items["history"] == history].reset_index(drop=True)


def _download_url(item: dict) -> str | None:
    for field in _URL_FIELDS:
        if item.get(field):
            return item[field]
    return None


def _stream_to_disk(url: str, path: Path, expected_size: int | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")

    with requests.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        written = 0
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                fh.write(chunk)
                written += len(chunk)

    if expected_size and written != expected_size:
        tmp.unlink(missing_ok=True)
        raise IOError(f"{path.name}: got {written} bytes, manifest says {expected_size}")
    tmp.replace(path)


def _zip_to_parquet(zip_path: Path, out_path: Path) -> int:
    """Convert the single CSV inside a bulk archive to parquet.

    pyarrow handles multi-gigabyte CSVs with far less memory than pandas. A
    large block size lets type inference see enough rows to avoid guessing a
    column's type from a sparse prefix; pandas is the fallback for the rare
    table whose types still conflict.
    """
    import pyarrow.csv as pacsv
    import pyarrow.parquet as pq

    extract_dir = RAW_DIR / "_extract"
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        csv_path = Path(zf.extract(member, extract_dir))

    try:
        table = pacsv.read_csv(
            csv_path, read_options=pacsv.ReadOptions(block_size=256 * 1024 * 1024)
        )
        pq.write_table(table, out_path, compression="zstd")
        rows = table.num_rows
    except Exception as e:
        logger.warning("  pyarrow could not parse %s (%s); falling back to pandas",
                       csv_path.name, e)
        df = pd.read_csv(csv_path, low_memory=False)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.to_parquet(out_path, index=False, compression="zstd")
        rows = len(df)
    finally:
        csv_path.unlink(missing_ok=True)

    return rows


def export_file(item: dict, force: bool = False) -> int | None:
    """Download one archive and convert it. Returns the row count."""
    table = item["table"]
    out_path = SNAPSHOT_DIR / f"{table}.parquet"
    zip_path = RAW_DIR / item["name"]

    if out_path.exists() and not force:
        logger.info("%s: already exported, skipping", table)
        return None

    if not item.get("canDownload"):
        raise PermissionError(f"{table}: not entitled to download ({item.get('historyLabel')})")

    url = _download_url(item)
    if url is None:
        raise KeyError(f"{table}: manifest item has no download link; fields: {sorted(item)}")

    logger.info("%s: downloading %s (%s)", table, item["name"], item.get("sizeLabel", "?"))
    _stream_to_disk(url, zip_path, item.get("size"))

    logger.info("  %s: converting to parquet", table)
    rows = _zip_to_parquet(zip_path, out_path)
    logger.info("  %s: %d rows -> %s", table, rows, out_path.name)
    return rows


def export_all(history: str = "full", force: bool = False) -> None:
    """Export every table in the chosen history tier."""
    manifest = list_files(history)
    if manifest.empty:
        raise RuntimeError("Bulk manifest is empty")

    blocked = manifest[~manifest["canDownload"].astype(bool)]
    if not blocked.empty:
        logger.error("Not entitled to %d of %d files at '%s' history: %s",
                     len(blocked), len(manifest), history, blocked["table"].tolist())

    # Small tables first, so a failure on a multi-gigabyte file does not
    # leave the cheap-but-essential ones (tickers, events) undownloaded.
    for item in manifest.sort_values("size").to_dict("records"):
        try:
            export_file(item, force=force)
        except Exception as e:
            logger.error("%s: export failed: %s", item["table"], e)


# ---------------------------------------------------------------------------
# Reading and verification
# ---------------------------------------------------------------------------

def load(table: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read an exported table from the local snapshot."""
    path = SNAPSHOT_DIR / f"{table}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"'{table}' has not been exported. Run scripts/sharadar_export.py "
            f"while the subscription is active."
        )
    return pd.read_parquet(path, columns=columns)


def _coverage(table: str, path: Path) -> dict:
    import pyarrow.parquet as pq

    schema_cols = set(pq.read_schema(path).names)
    wanted = [c for c in ("ticker", "date", "dimension", "isdelisted") if c in schema_cols]
    df = pd.read_parquet(path, columns=wanted) if wanted else pd.DataFrame()
    rows = pq.ParquetFile(path).metadata.num_rows

    info = {"rows": rows}
    if "ticker" in df:
        info["tickers"] = int(df["ticker"].nunique())
    if "date" in df:
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        if not dates.empty:
            info["date_min"] = str(dates.min().date())
            info["date_max"] = str(dates.max().date())
            info["years"] = round((dates.max() - dates.min()).days / 365.25, 1)
    if "dimension" in df:
        info["dimensions"] = ",".join(sorted(df["dimension"].dropna().unique()))
    if "isdelisted" in df:
        info["delisted"] = int((df["isdelisted"] == "Y").sum())
    return info


def _judge(table: str, info: dict) -> str:
    rule = COVERAGE.get(table)
    if not rule:
        return "ok" if info.get("rows", 0) > 0 else "EMPTY"

    problems = []
    if info.get("tickers", 0) < rule.get("min_tickers", 0):
        problems.append(f"tickers<{rule['min_tickers']}")
    if info.get("years", 0) < rule.get("min_years", 0):
        problems.append(f"years<{rule['min_years']}")
    if info.get("rows", 0) < rule.get("min_rows", 0):
        problems.append(f"rows<{rule['min_rows']}")
    if "dimensions" in rule:
        missing = rule["dimensions"] - set(str(info.get("dimensions", "")).split(","))
        if missing:
            problems.append(f"missing dims {sorted(missing)}")

    return "ok" if not problems else "INCOMPLETE: " + "; ".join(problems)


def verify() -> pd.DataFrame:
    """Coverage report for the local snapshot.

    Presence is not enough: the free sample is a valid file with the right
    columns. Each table is checked against the coverage a full-history export
    must reach, so a sample or truncated download cannot pass as complete.
    """
    rows = []
    for table, purpose in PURPOSE.items():
        path = SNAPSHOT_DIR / f"{table}.parquet"
        if not path.exists():
            rows.append({"table": table, "status": "MISSING", "purpose": purpose})
            continue
        info = _coverage(table, path)
        rows.append({
            "table": table, "status": _judge(table, info),
            "size_mb": round(path.stat().st_size / 1e6, 1), **info,
        })
    return pd.DataFrame(rows)
