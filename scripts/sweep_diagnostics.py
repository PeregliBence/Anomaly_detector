"""Stress-test the horizon sweep before believing it.

A strong result from a backtest is a reason for suspicion, not celebration.
This checks the saved predictions two ways:

1. Stability over time: is the IC spread across years, or does one crisis
   carry it?
2. Where it comes from: how much of the IC sits in names that later delisted,
   whose final returns resolve at the last traded price.

The placebo and feature-importance checks need a refit and live in
scripts/leakage_checks.py.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import newey_west_tstat, rank_ic
from src.sources import sharadar

warnings.filterwarnings("ignore")
OUT_DIR = Path("results/horizon_sweep")


def yearly_ic(preds: pd.DataFrame) -> pd.Series:
    ic = rank_ic(preds)
    return ic.groupby(ic.index.year).mean()


def main() -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    meta = sharadar.equity_meta()
    delisted = set(meta.index[meta["isdelisted"] == "Y"])

    yearly = {}
    split_rows = []

    for h in (5, 20, 60, 120):
        preds = pd.read_parquet(OUT_DIR / f"predictions_h{h}.parquet")
        lgb = preds[preds["model"] == "lightgbm"]
        yearly[h] = yearly_ic(lgb)

        tickers = lgb.index.get_level_values("ticker")
        is_delisted = tickers.isin(delisted)
        for label, mask in (("all", np.ones(len(lgb), bool)),
                            ("survivors only", ~is_delisted),
                            ("delisted only", is_delisted)):
            sub = lgb[mask]
            ic = rank_ic(sub, min_names=20)
            t, _ = newey_west_tstat(ic, lags=h - 1)
            split_rows.append({"horizon": h, "subset": label, "share_rows": mask.mean(),
                               "ic_mean": ic.mean(), "ic_tstat": t, "n_dates": len(ic)})

    print("\n===== LightGBM IC by year =====")
    print(pd.DataFrame(yearly).round(3).to_string())

    print("\n===== LightGBM IC: survivors vs names that later delisted =====")
    print(pd.DataFrame(split_rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
