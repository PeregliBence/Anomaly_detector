"""Paths, credentials and the constants the pipeline is built around."""

from __future__ import annotations

import logging
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
STORE_PATH = DATA_DIR / "research.db"
RESULTS_DIR = ROOT / "results"

SHARADAR_KEY = os.environ.get("NASDAQ_DATA_LINK_API_KEY", "")
POLYGON_KEY = os.environ.get("POLYGON_API_KEY", "")
FINNHUB_KEY = os.environ.get("FINNHUB_API_KEY", "")

# Evaluation protocol. These are the numbers the thesis has to defend, so they
# live in one place rather than scattered through call sites.
HORIZONS = (5, 20, 60, 120)
DEFAULT_HORIZON = 20

# Walk-forward shape: warm up on this many trading days before the first
# prediction, then refit every REFIT_EVERY days.
MIN_TRAIN_DAYS = 756
REFIT_EVERY = 21

# A cross-sectional rank means nothing across a handful of names.
MIN_NAMES_PER_DATE = 20

# Universe: rank by dollar volume and keep the most liquid names. Microcaps
# carry unreliable prices and could not be traded on the signal anyway.
UNIVERSE_SIZE = 1000
UNIVERSE_REBALANCE_DAYS = 63

START_DATE = "2005-01-01"


def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
        force=True,
    )
    return logging.getLogger("alpha")


def require(key: str, name: str) -> str:
    if not key:
        raise RuntimeError(
            f"{name} is not set. Add it to .env at the project root."
        )
    return key
