"""Panel dataset assembly across many instruments.

Training one model per ticker wastes most of the available data: a single
instrument contributes a few thousand rows, and the model spends its capacity
on that instrument's noise. Stacking every instrument into one long-format
panel turns the same history into hundreds of thousands of rows of a single
learning problem, which is what the empirical asset pricing literature does.

The panel also unlocks the cross-sectional target: instead of asking whether
an instrument rises, ask whether it beats the other instruments over the same
window. That removes the market factor, which dominates single-asset returns,
and pins the base rate at 50% by construction.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.features import build_features, forward_return

logger = logging.getLogger(__name__)

# A cross-sectional rank is meaningless across a handful of names.
MIN_NAMES_PER_DATE = 10


def build_panel(
    frames: dict[str, pd.DataFrame],
    horizon: int = 20,
    min_rows: int = 300,
) -> pd.DataFrame:
    """Stack per-ticker OHLCV frames into one long-format feature panel.

    Returns a frame indexed by (date, ticker) carrying the causal features plus
    `fwd_return`, the realised forward return the targets are derived from.
    """
    blocks = []

    for ticker, df in frames.items():
        if len(df) < min_rows:
            logger.debug("%s: only %d rows, skipping", ticker, len(df))
            continue

        feats = build_features(df)
        feats["fwd_return"] = forward_return(df, horizon)
        feats = feats.loc[build_features(df).notna().all(axis=1)]

        if feats.empty:
            continue

        feats["ticker"] = ticker
        blocks.append(feats.set_index("ticker", append=True))

    if not blocks:
        return pd.DataFrame()

    panel = pd.concat(blocks).sort_index()
    panel.index.names = ["date", "ticker"]
    logger.info("Panel built: %d rows, %d instruments", len(panel), len(blocks))
    return panel


def add_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add features that only exist relative to the rest of the field.

    Each is computed within a date, so no future information enters: on any
    given day the whole cross-section is already observable.
    """
    out = panel.copy()
    by_date = out.groupby(level="date")

    # Where an instrument sits in the field on the features that carry the
    # documented cross-sectional effects.
    for col in ("mom_20", "mom_60", "vol_20", "dist_sma_50", "rsi_14"):
        if col in out.columns:
            out[f"cs_rank_{col}"] = by_date[col].rank(pct=True)

    # State of the field itself: how wide the spread of returns is, and how
    # much of the market is advancing.
    out["cs_dispersion"] = by_date["log_return"].transform("std")
    out["cs_breadth"] = by_date["log_return"].transform(lambda s: (s > 0).mean())
    out["cs_market_return"] = by_date["log_return"].transform("mean")

    # The instrument's move net of the field's move on the same day.
    out["excess_return"] = out["log_return"] - out["cs_market_return"]

    return out


def make_cross_sectional_target(panel: pd.DataFrame) -> pd.Series:
    """1 where the instrument beat the cross-sectional median, else 0.

    Dates holding too few instruments are left unlabelled rather than ranked
    against a field too small to mean anything.
    """
    fwd = panel["fwd_return"]
    counts = fwd.groupby(level="date").transform("count")
    median = fwd.groupby(level="date").transform("median")

    target = (fwd > median).astype(float)
    return target.where(fwd.notna() & (counts >= MIN_NAMES_PER_DATE))


def make_absolute_target(panel: pd.DataFrame) -> pd.Series:
    """1 where the instrument rose over the horizon, else 0."""
    fwd = panel["fwd_return"]
    return (fwd > 0).astype(float).where(fwd.notna())


TARGET_BUILDERS = {
    "cross_sectional": make_cross_sectional_target,
    "absolute": make_absolute_target,
}


def split_xy(
    panel: pd.DataFrame, target_mode: str = "cross_sectional"
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separate the panel into (features, target, forward return)."""
    if target_mode not in TARGET_BUILDERS:
        raise ValueError(f"Unknown target mode '{target_mode}'")

    y = TARGET_BUILDERS[target_mode](panel)
    fwd = panel["fwd_return"]
    X = panel.drop(columns=["fwd_return"]).replace([np.inf, -np.inf], np.nan)

    complete = X.notna().all(axis=1)
    return X.loc[complete], y.loc[complete], fwd.loc[complete]
