"""Panel assembly: many instruments, one learning problem.

Training one model per instrument wastes most of the available data — a few
thousand rows each, spent learning that instrument's noise. Stacking every
instrument into one long-format panel turns the same history into millions of
rows of a single problem, which is how the empirical asset pricing literature
uses it.

The panel also unlocks the cross-sectional target. Instead of asking whether
an instrument rises, ask whether it beats the rest of the field over the same
window. That removes the market factor, which dominates single-asset returns,
and fixes the base rate at 50% by construction, so any edge above it is skill
rather than a restatement of the market's drift.

Forward returns for several horizons are built in one pass: the features do
not depend on the horizon, and recomputing them per horizon is the expensive
part.
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.config import MIN_NAMES_PER_DATE, UNIVERSE_REBALANCE_DAYS, UNIVERSE_SIZE
from src.features import build_features, build_market_features, forward_return

logger = logging.getLogger(__name__)

# Raw dollar volume spans orders of magnitude across names and eras; it drives
# universe selection but is never a model input. Forward returns are labels.
NON_FEATURE_COLS = ["dollar_volume"]
FWD_PREFIX = "fwd_return_"


def fwd_col(horizon: int) -> str:
    return f"{FWD_PREFIX}{horizon}"


def _member_dates(keep: pd.MultiIndex) -> dict[str, pd.DatetimeIndex]:
    frame = keep.to_frame(index=False)
    return {t: pd.DatetimeIndex(g["date"]) for t, g in frame.groupby("ticker")}


def sequence_columns(df: pd.DataFrame, feats: pd.DataFrame, seq_len: int) -> pd.DataFrame:
    """The last `seq_len` days of three daily channels, as lag columns.

    Named seq_<channel>_<lag>, lag 0 being the current day. The LSTM reshapes
    them into a sequence; every other model receives the same columns flat,
    so all architectures see identical information.
    """
    channels = {
        "ret": feats["log_return"],
        "vol": np.log(feats["volume_spike"].clip(lower=1e-6)),
        "rng": np.log(df["High"] / df["Low"]).replace([np.inf, -np.inf], np.nan),
    }
    cols = {
        f"seq_{name}_{lag:02d}": series.shift(lag)
        for name, series in channels.items()
        for lag in range(seq_len)
    }
    return pd.DataFrame(cols, index=feats.index)


def build_panel(
    frames: dict[str, pd.DataFrame] | Iterable[tuple[str, pd.DataFrame]],
    horizons: Sequence[int],
    market: pd.DataFrame | None = None,
    min_rows: int = 300,
    delisted: set[str] | None = None,
    keep: pd.MultiIndex | None = None,
    seq_len: int = 0,
) -> pd.DataFrame:
    """Stack per-instrument frames into one panel indexed by (date, ticker).

    `keep` restricts the rows to universe membership after each instrument's
    features are computed, so warm-up windows still see the full history.
    `seq_len` > 0 adds daily lag columns for sequence models.
    Features are stored as float32 to keep a multi-million-row panel in RAM.
    """
    delisted = delisted or set()
    items = frames.items() if isinstance(frames, dict) else frames
    member_dates = _member_dates(keep) if keep is not None else None

    blocks = []
    for ticker, df in items:
        if len(df) < min_rows:
            continue
        if member_dates is not None and ticker not in member_dates:
            continue

        feats = build_features(df)
        if seq_len:
            feats = feats.join(sequence_columns(df, feats, seq_len))
        for h in horizons:
            feats[fwd_col(h)] = forward_return(df, h, delisted=ticker in delisted)

        if member_dates is not None:
            feats = feats[feats.index.isin(member_dates[ticker])]
        if feats.empty:
            continue

        feats["ticker"] = ticker
        feats = feats.set_index("ticker", append=True)
        feats.index.names = ["date", "ticker"]
        blocks.append(feats.astype("float32"))

    if not blocks:
        return pd.DataFrame()

    panel = pd.concat(blocks).sort_index()
    del blocks

    if market is not None and not market.empty:
        market_feats = build_market_features(market).astype("float32")
        panel = panel.join(market_feats, on="date")

    logger.info("Panel: %d rows, %d instruments, %d columns",
                len(panel), panel.index.get_level_values("ticker").nunique(), panel.shape[1])
    return panel


def apply_universe_filter(
    panel: pd.DataFrame,
    size: int = UNIVERSE_SIZE,
    rebalance_days: int = UNIVERSE_REBALANCE_DAYS,
) -> pd.DataFrame:
    """Keep the most liquid instruments, reselected periodically.

    Used for the yfinance path; the Sharadar path selects membership up front
    with universe.liquid_membership, before features are built.
    """
    if "dollar_volume" not in panel.columns:
        return panel

    dates = panel.index.get_level_values("date").unique().sort_values()
    keep = pd.Series(False, index=panel.index)

    for i in range(0, len(dates), rebalance_days):
        window = dates[i:i + rebalance_days]
        ranked = panel.xs(dates[i], level="date")["dollar_volume"].dropna().nlargest(size)
        if ranked.empty:
            continue
        in_window = panel.index.get_level_values("date").isin(window)
        in_universe = panel.index.get_level_values("ticker").isin(set(ranked.index))
        keep |= in_window & in_universe

    return panel[keep]


def add_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Features that only exist relative to the rest of the field.

    Computed within a date, so nothing leaks: on any given day the whole
    cross-section is already observable.
    """
    out = panel
    by_date = out.groupby(level="date")

    rank_cols = [
        "mom_12_1", "mom_6_1", "mom_20", "mom_60",
        "reversal_5", "reversal_21",
        "vol_20", "vol_252", "vol_ratio",
        "dist_sma_50", "dist_52w_high", "drawdown",
        "rsi_14", "log_dollar_volume", "illiquidity",
    ]
    for col in rank_cols:
        if col in out.columns:
            out[f"cs_{col}"] = by_date[col].rank(pct=True).astype("float32")

    out["cs_dispersion"] = by_date["log_return"].transform("std").astype("float32")
    out["cs_breadth"] = (out["log_return"] > 0).groupby(level="date").transform("mean").astype("float32")
    market_return = by_date["log_return"].transform("mean")
    out["cs_excess_return"] = (out["log_return"] - market_return).astype("float32")
    out["cs_n_names"] = by_date["log_return"].transform("count").astype("float32")

    return out


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def cross_sectional_target(fwd: pd.Series) -> pd.Series:
    """1 where the instrument beat the cross-sectional median, else 0.

    Dates with too few names are left unlabelled rather than ranked against a
    field too small to mean anything.
    """
    by_date = fwd.groupby(level="date")
    counts = by_date.transform("count")
    median = by_date.transform("median")
    return (fwd > median).astype("float32").where(fwd.notna() & (counts >= MIN_NAMES_PER_DATE))


def absolute_target(fwd: pd.Series) -> pd.Series:
    """1 where the instrument rose over the horizon, else 0."""
    return (fwd > 0).astype("float32").where(fwd.notna())


TARGETS = {
    "cross_sectional": cross_sectional_target,
    "absolute": absolute_target,
}


def split_xy(
    panel: pd.DataFrame,
    horizon: int,
    target_mode: str = "cross_sectional",
    max_nan_frac: float = 0.3,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separate the panel into (features, target, forward return) for one horizon.

    Columns missing on more than `max_nan_frac` of rows are dropped rather than
    forcing every row through the longest warm-up any single feature needs.
    """
    if target_mode not in TARGETS:
        raise ValueError(f"Unknown target mode '{target_mode}'. Use one of {list(TARGETS)}")
    col = fwd_col(horizon)
    if col not in panel.columns:
        raise KeyError(f"Panel has no forward return for horizon {horizon}")

    fwd = panel[col]
    y = TARGETS[target_mode](fwd)

    drop = [c for c in panel.columns if c.startswith(FWD_PREFIX) or c in NON_FEATURE_COLS]
    X = panel.drop(columns=drop)
    X = X.replace([np.inf, -np.inf], np.nan)

    sparse = X.columns[X.isna().mean() > max_nan_frac]
    if len(sparse):
        logger.info("Dropping %d sparse columns: %s", len(sparse), list(sparse))
        X = X.drop(columns=sparse)

    complete = X.notna().all(axis=1)
    logger.info("Usable rows for h=%d: %d of %d", horizon, complete.sum(), len(X))
    return X[complete], y[complete], fwd[complete]
