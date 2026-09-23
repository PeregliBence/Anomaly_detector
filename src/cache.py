"""Parquet cache for downloaded data.

A full backfill walks the same history hundreds of times. Without a cache each
pass would re-hit the vendor, which is slow, rate-limited and — for the paid
sources — metered.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import CACHE_DIR

logger = logging.getLogger(__name__)


def _path(name: str) -> Path:
    safe = name.replace("/", "_").replace("^", "IDX_").replace("=", "_").replace(":", "_")
    return CACHE_DIR / f"{safe}.parquet"


def read(name: str) -> pd.DataFrame | None:
    path = _path(name)
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        logger.warning("Cache read failed for %s: %s", name, e)
        return None


def write(name: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(_path(name))
    except Exception as e:
        logger.warning("Cache write failed for %s: %s", name, e)


def clear(name: str) -> None:
    path = _path(name)
    if path.exists():
        path.unlink()
