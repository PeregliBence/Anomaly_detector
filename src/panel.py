"""Panel assembly: many instruments, one learning problem.

Training one model per instrument wastes most of the available data — a few
thousand rows each, spent learning that instrument's noise. Stacking every
instrument into one long-format panel turns the same history into hundreds of
thousands of rows of a single problem, which is how the empirical asset
pricing literature uses it.

The panel also unlocks the cross-sectional target. Instead of asking whether
an instrument rises, ask whether it beats the rest of the field over the same
window. That removes the market factor, which dominates single-asset returns,
and fixes the base rate at 50% by construction, so any edge above it is skill
rather than a restatement of the market's drift.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import MIN_NAMES_PER_DATE, UNIVERSE_REBALANCE_DAYS, UNIVERSE_SIZE
from src.features import build_features, build_market_features, forward_return

logger = logging.getLogger(__name__)

# Kept for universe selection, never fed to a model: raw dollar volume spans
# several orders of magnitude across instruments and eras.
NON_FEATURE_COLS = ["dollar_volume", "fwd_return"]


def build_panel(
    frames: dict[str, pd.DataFrame],
    horizon: int,
    market: pd.DataFrame | None = None,
    min_rows: int = 300,
) -> pd.DataFrame:
    """Stack per-instrument frames into one panel indexed by (date, ticker)."""
    blocks = []

    for ticker, df in frames.items():
        if len(df) < min_rows:
            continue

        feats = build_features(df)
        feats["fwd_return"] = forward_return(df, horizon)
        feats["ticker"] = ticker
        blocks.append(feats.set_index("ticker", append=True))

    if not blocks:
        return pd.DataFrame()

    panel = pd.concat(blocks)
    panel.index.names = ["date", "ticker"]
    panel = panel.sort_index()

    if market is not None and not market.empty:
        market_feats = build_market_features(market)
        panel = panel.join(market_feats, on="date")

    logger.info("Panel: %d rows, %d instruments, %d columns",
                len(panel), len(blocks), panel.shape[1])
    return panel


def apply_universe_filter(
    panel: pd.DataFrame,
    size: int = UNIVERSE_SIZE,
    rebalance_days: int = UNIVERSE_REBALANCE_DAYS,
) -> pd.DataFrame:
    """Keep the most liquid instruments, reselected periodically.

    Selection uses trailing dollar volume only, so membership on any date is
    decided from information available then. Rebalancing on a slow schedule
    rather than daily avoids churning the universe on noise.
    """
    if "dollar_volume" not in panel.columns:
        return panel

    dates = panel.index.get_level_values("date").unique().sort_values()
    if len(dates) == 0:
        return panel

    keep = pd.Series(False, index=panel.index)

    for i in range(0, len(dates), rebalance_days):
        window = dates[i:i + rebalance_days]
        as_of = dates[i]

        ranked = (
            panel.xs(as_of, level="date")["dollar_volume"]
            .dropna()
            .nlargest(size)
        )
        if ranked.empty:
            continue

        selected = set(ranked.index)
        in_window = panel.index.get_level_values("date").isin(window)
        in_universe = panel.index.get_level_values("ticker").isin(selected)
        keep |= in_window & in_universe

    filtered = panel[keep]
    logger.info("Universe filter: %d rows retained of %d", len(filtered), len(panel))
    return filtered


def add_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Features that only exist relative to the rest of the field.

    Computed within a date, so nothing leaks: on any given day the whole
    cross-section is already observable.
    """
    out = panel.copy()
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
            out[f"cs_{col}"] = by_date[col].rank(pct=True)

    # State of the field itself.
    out["cs_dispersion"] = by_date["log_return"].transform("std")
    out["cs_breadth"] = by_date["log_return"].transform(lambda s: (s > 0).mean())
    market_return = by_date["log_return"].transform("mean")
    out["cs_excess_return"] = out["log_return"] - market_return
    out["cs_n_names"] = by_date["log_return"].transform("count")

    return out


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def cross_sectional_target(panel: pd.DataFrame) -> pd.Series:
    """1 where the instrument beat the cross-sectional median, else 0.

    Dates with too few names are left unlabelled rather than ranked against a
    field too small to mean anything.
    """
    fwd = panel["fwd_return"]
    by_date = fwd.groupby(level="date")
    counts = by_date.transform("count")
    median = by_date.transform("median")
    return (fwd > median).astype(float).where(fwd.notna() & (counts >= MIN_NAMES_PER_DATE))


def absolute_target(panel: pd.DataFrame) -> pd.Series:
    """1 where the instrument rose over the horizon, else 0."""
    fwd = panel["fwd_return"]
    return (fwd > 0).astype(float).where(fwd.notna())


TARGETS = {
    "cross_sectional": cross_sectional_target,
    "absolute": absolute_target,
}


def split_xy(
    panel: pd.DataFrame, target_mode: str = "cross_sectional", max_nan_frac: float = 0.3
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separate the panel into (features, target, forward return).

    Columns missing on more than `max_nan_frac` of rows are dropped rather than
    forcing every row through the longest warm-up any single feature needs.
    """
    if target_mode not in TARGETS:
        raise ValueError(f"Unknown target mode '{target_mode}'. Use one of {list(TARGETS)}")

    y = TARGETS[target_mode](panel)
    fwd = panel["fwd_return"]

    X = panel.drop(columns=[c for c in NON_FEATURE_COLS if c in panel.columns])
    X = X.replace([np.inf, -np.inf], np.nan)

    sparse = X.columns[X.isna().mean() > max_nan_frac]
    if len(sparse):
        logger.info("Dropping %d sparse columns: %s", len(sparse), list(sparse))
        X = X.drop(columns=sparse)

    complete = X.notna().all(axis=1)
    logger.info("Usable rows: %d of %d", complete.sum(), len(X))
    return X[complete], y[complete], fwd[complete]
