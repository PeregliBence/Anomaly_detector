"""Sharadar bulk export via Nasdaq Data Link.

The subscription is deliberately short: one month, during which every table is
pulled in full and kept locally. Sharadar granted a two-year retention term
for this research in writing, so the historical study runs off the local
snapshot rather than a live subscription.

Everything is therefore exported, including tables with no immediate use — a
table not pulled during that month cannot be fetched later without
re-subscribing, and disk is free.
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src.config import DATA_DIR, SHARADAR_KEY, require

logger = logging.getLogger(__name__)

API = "https://data.nasdaq.com/api/v3/datatables/SHARADAR"
SNAPSHOT_DIR = DATA_DIR / "sharadar"

# Every table in the bundle, with what it is for.
TABLES = {
    "SEP": "Equity prices, daily, including delisted securities",
    "SFP": "Fund prices, daily",
    "SF1": "Core fundamentals, all dimensions including as-reported",
    "SF2": "Insider transactions (Form 4)",
    "SF3": "Institutional holdings (13F)",
    "TICKERS": "Security metadata including delisting dates",
    "ACTIONS": "Corporate actions",
    "EVENTS": "Event dates including earnings announcements",
    "DAILY": "Daily valuation metrics: market cap, EV, P/E",
    "SP500": "S&P 500 constituent changes",
}

# Date column per table, used to verify coverage after download.
DATE_COLUMN = {
    "SEP": "date", "SFP": "date", "SF1": "datekey", "SF2": "filingdate",
    "SF3": "calendardate", "ACTIONS": "date", "EVENTS": "date",
    "DAILY": "date", "SP500": "date",
}


def _request_export(table: str, key: str) -> dict:
    resp = requests.get(
        f"{API}/{table}.json",
        params={"qopts.export": "true", "api_key": key},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("datatable_bulk_download", {}).get("file", {})


def _wait_for_file(table: str, key: str, timeout_s: int = 1800) -> str:
    """Poll until Sharadar has regenerated the snapshot, then return its link."""
    deadline = time.time() + timeout_s
    delay = 15

    while time.time() < deadline:
        info = _request_export(table, key)
        status, link = info.get("status"), info.get("link")

        if status == "fresh" and link:
            return link
        logger.info("  %s: snapshot %s, waiting %ds", table, status, delay)
        time.sleep(delay)
        delay = min(delay * 1.5, 120)

    raise TimeoutError(f"{table}: snapshot not ready within {timeout_s}s")


def _download_to_parquet(table: str, link: str, out_path: Path) -> pd.DataFrame:
    logger.info("  %s: downloading", table)
    resp = requests.get(link, timeout=1800)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            df = pd.read_csv(fh, low_memory=False)

    date_col = DATE_COLUMN.get(table)
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


def export_table(table: str, force: bool = False) -> pd.DataFrame | None:
    """Export one table to the local snapshot directory."""
    key = require(SHARADAR_KEY, "NASDAQ_DATA_LINK_API_KEY")
    out_path = SNAPSHOT_DIR / f"{table}.parquet"

    if out_path.exists() and not force:
        logger.info("%s: already exported, skipping", table)
        return pd.read_parquet(out_path)

    logger.info("%s: requesting export (%s)", table, TABLES.get(table, ""))
    link = _wait_for_file(table, key)
    df = _download_to_parquet(table, link, out_path)
    logger.info("  %s: %d rows, %d columns -> %s", table, len(df), df.shape[1], out_path)
    return df


def export_all(force: bool = False) -> dict[str, dict]:
    """Export every table in the bundle. Returns a per-table summary."""
    summary = {}

    for table in TABLES:
        try:
            df = export_table(table, force=force)
            summary[table] = _describe(table, df)
        except Exception as e:
            logger.error("%s: export failed: %s", table, e)
            summary[table] = {"error": str(e)}

    return summary


def _describe(table: str, df: pd.DataFrame | None) -> dict:
    if df is None or df.empty:
        return {"rows": 0}

    info = {"rows": len(df), "columns": df.shape[1]}

    date_col = DATE_COLUMN.get(table)
    if date_col and date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if not dates.empty:
            info["date_min"] = str(dates.min().date())
            info["date_max"] = str(dates.max().date())

    if "ticker" in df.columns:
        info["tickers"] = int(df["ticker"].nunique())
    if "dimension" in df.columns:
        info["dimensions"] = sorted(df["dimension"].dropna().unique().tolist())

    return info


def load(table: str) -> pd.DataFrame:
    """Read a previously exported table from the local snapshot."""
    path = SNAPSHOT_DIR / f"{table}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{table} has not been exported. Run scripts/sharadar_export.py "
            f"while the subscription is active."
        )
    return pd.read_parquet(path)


def verify() -> pd.DataFrame:
    """Summarise what is on disk, so gaps surface before the subscription ends."""
    rows = []
    for table, purpose in TABLES.items():
        path = SNAPSHOT_DIR / f"{table}.parquet"
        if not path.exists():
            rows.append({"table": table, "status": "MISSING", "purpose": purpose})
            continue
        df = pd.read_parquet(path)
        rows.append({
            "table": table, "status": "ok", "purpose": purpose,
            "size_mb": round(path.stat().st_size / 1e6, 1),
            **_describe(table, df),
        })
    return pd.DataFrame(rows)
