"""Robustness checks on the horizon sweep's headline result.

1. Permutation test. A single shuffled-label run is one draw, not a test:
   with an expanding training window every fold learns from nearly the same
   shuffled labels, so the placebo model is effectively one random function
   held fixed across the whole test period. Several shuffles give the null
   distribution of IC; the real model is compared against it under an
   identical protocol.

2. Macro ablation. Market-state features are constant across names within a
   date, so they cannot rank on their own — they only modulate the
   cross-sectional features by regime. Dropping them asks whether that regime
   timing is doing the work. It also moves the first test date back to 2008,
   since the credit ETF series behind two macro features only begins in 2007.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import experiment, models
from src.config import setup_logging
from src.features import build_market_features
from src.metrics import newey_west_tstat, rank_ic
from src.panel import split_xy
from src.sources import yahoo
from src.validation import walk_forward

warnings.filterwarnings("ignore")
setup_logging("WARNING")

OUT_DIR = Path("results/robustness")
N_SHUFFLES = 10
# Coarser than the sweep to make ten placebo runs affordable; the real run
# used for comparison is repeated under exactly the same settings.
PERM_PROTOCOL = dict(refit_every=126, train_stride=10)
SWEEP_PROTOCOL = dict(refit_every=63, train_stride=5)


def shuffled_within_date(y: pd.Series, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    values = y.values.copy()
    for idx in y.groupby(level="date").indices.values():
        values[idx] = rng.permutation(values[idx])
    return pd.Series(values, index=y.index)


def lightgbm_ic(X, y, fwd, horizon, protocol) -> float:
    preds = walk_forward(X, y, fwd, lambda: models.build("lightgbm"),
                         horizon=horizon, **protocol)
    return float(rank_ic(preds).mean())


def permutation_test(panel: pd.DataFrame, horizon: int) -> dict:
    X, y, fwd = split_xy(panel, horizon=horizon)
    real = lightgbm_ic(X, y, fwd, horizon, PERM_PROTOCOL)
    null = []
    for seed in range(N_SHUFFLES):
        t = time.time()
        null.append(lightgbm_ic(X, shuffled_within_date(y, seed), fwd, horizon, PERM_PROTOCOL))
        print(f"  shuffle {seed}: IC {null[-1]:+.4f} ({time.time() - t:.0f}s)", flush=True)
    null = np.array(null)
    return {
        "horizon": horizon,
        "real_ic": real,
        "null_mean": null.mean(),
        "null_std": null.std(ddof=1),
        "null_max": null.max(),
        "z_vs_null": (real - null.mean()) / null.std(ddof=1),
        "empirical_p": (np.sum(null >= real) + 1) / (len(null) + 1),
    }


def macro_ablation(panel: pd.DataFrame, macro_cols: list[str], horizon: int) -> pd.DataFrame:
    stripped = panel.drop(columns=[c for c in macro_cols if c in panel.columns])
    _, scorecard = experiment.run(
        stripped, horizon=horizon,
        model_names=["random", "cs_momentum", "cs_low_vol", "logistic", "lightgbm"],
        **SWEEP_PROTOCOL,
    )
    first = split_xy(stripped, horizon=horizon)[0].index.get_level_values("date").min()
    scorecard["first_feature_date"] = first.date()
    return scorecard


def main() -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = experiment.load_sharadar_panel(horizons=[20, 60], start="2005-01-01", size=1000)
    macro_cols = list(build_market_features(yahoo.load_market_state("2003-01-01")).columns)

    print("\n===== Permutation test, LightGBM, h=20 =====", flush=True)
    perm = permutation_test(panel, horizon=20)
    pd.DataFrame([perm]).to_csv(OUT_DIR / "permutation_h20.csv", index=False)
    print(pd.DataFrame([perm]).round(4).to_string(index=False), flush=True)

    frames = []
    for h in (20, 60):
        print(f"\n===== Macro ablation, h={h} =====", flush=True)
        sc = macro_ablation(panel, macro_cols, horizon=h)
        frames.append(sc)
        cols = ["horizon", "model", "ic_mean", "ic_tstat", "ic_tstat_folds",
                "ic_positive_rate", "spread_mean", "n_dates", "first_feature_date"]
        print(sc[[c for c in cols if c in sc]].round(4).to_string(index=False), flush=True)
    pd.concat(frames).to_csv(OUT_DIR / "macro_ablation.csv", index=False)


if __name__ == "__main__":
    main()
