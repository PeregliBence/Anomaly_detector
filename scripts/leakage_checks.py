"""Leakage checks for the horizon sweep.

1. Univariate IC per feature. A feature that has seen the future shows up as
   implausibly predictive on its own; honest features on this problem rarely
   exceed |IC| = 0.05.
2. Placebo. LightGBM is trained on labels shuffled within each date and
   scored against the real returns. Shuffling destroys any real relationship,
   so the IC must collapse to zero. If it does not, the harness itself —
   alignment, purging, scoring — is manufacturing signal.
3. Feature importance. What the model actually leans on, for a sanity read.
4. Coverage start. Which feature delays the first usable date, since that
   decides whether 2008 lands in the test period.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import experiment, models
from src.config import setup_logging
from src.metrics import newey_west_tstat, rank_ic
from src.panel import split_xy
from src.validation import walk_forward

warnings.filterwarnings("ignore")
setup_logging("WARNING")
OUT_DIR = Path("results/horizon_sweep")


def univariate_ic(X: pd.DataFrame, fwd: pd.Series, stride: int = 5) -> pd.DataFrame:
    dates = X.index.get_level_values("date").unique().sort_values()[::stride]
    mask = X.index.get_level_values("date").isin(dates)
    Xs, fs = X[mask], fwd[mask]
    rows = []
    for col in Xs.columns:
        ic = rank_ic(pd.DataFrame({"score": Xs[col], "fwd_return": fs}), min_names=20)
        rows.append({"feature": col, "ic_mean": ic.mean()})
    out = pd.DataFrame(rows)
    out["abs_ic"] = out["ic_mean"].abs()
    return out.sort_values("abs_ic", ascending=False)


def shuffled_within_date(y: pd.Series, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    values = y.values.copy()
    for idx in y.groupby(level="date").indices.values():
        values[idx] = rng.permutation(values[idx])
    return pd.Series(values, index=y.index)


def main() -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 100)

    panel = experiment.load_sharadar_panel(horizons=[20, 120], start="2005-01-01", size=1000)

    print("\n===== First date each feature is available (latest 8) =====")
    first = panel.drop(columns=[c for c in panel if c.startswith("fwd_return_")]).apply(
        lambda s: s.first_valid_index()[0] if s.first_valid_index() is not None else pd.NaT
    )
    print(first.sort_values(ascending=False).head(8).to_string())

    for h in (20, 120):
        X, y, fwd = split_xy(panel, horizon=h)
        uni = univariate_ic(X, fwd)
        print(f"\n===== Univariate IC, h={h}: strongest 12 features =====")
        print(uni.head(12).round(4).to_string(index=False))
        flagged = uni[uni["abs_ic"] > 0.1]
        print(f"Features with |IC| > 0.1: {flagged['feature'].tolist() or 'none'}")

    X, y, fwd = split_xy(panel, horizon=20)
    placebo = walk_forward(
        X, shuffled_within_date(y), fwd, lambda: models.build("lightgbm"),
        horizon=20, refit_every=63, train_stride=5,
    )
    ic = rank_ic(placebo)
    t, _ = newey_west_tstat(ic, lags=19)
    print(f"\n===== Placebo (shuffled labels), h=20 =====\nIC {ic.mean():.4f}, t {t:.2f}  "
          f"(real run: IC 0.0313, t 3.43)")

    X, y, fwd = split_xy(panel, horizon=120)
    cutoff = pd.Timestamp("2018-01-01")
    train = (X.index.get_level_values("date") <= cutoff) & y.notna()
    stride = X.index.get_level_values("date").isin(
        X.index.get_level_values("date").unique()[::5])
    model = models.build("lightgbm").fit(X[train & stride], y[train & stride])
    importance = pd.Series(model.clf_.booster_.feature_importance("gain"), index=X.columns)
    print("\n===== LightGBM gain importance, h=120, trained to 2018 (top 15) =====")
    print((importance / importance.sum()).sort_values(ascending=False).head(15).round(3).to_string())


if __name__ == "__main__":
    main()
