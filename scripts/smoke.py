"""End-to-end smoke test of the panel pipeline on free data.

Proves the chain runs — download, panel assembly, cross-sectional features,
date-purged walk-forward, scoring — before the paid sources are wired in.
Numbers here are not the thesis result: the universe is survivor-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import experiment, universe
from src.config import setup_logging

logger = setup_logging("INFO")

HORIZON = 20
MODELS = ["random", "constant", "cs_momentum", "cs_reversal", "logistic", "lightgbm"]


def main() -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    tickers = universe.small(n=50)
    panel = experiment.load_panel(tickers, horizon=HORIZON, start="2010-01-01", end="2026-09-01")

    if panel.empty:
        print("Panel is empty — nothing to run.")
        return

    dates = panel.index.get_level_values("date")
    print(f"\nPanel: {len(panel):,} rows | {panel.index.get_level_values('ticker').nunique()} "
          f"instruments | {dates.min().date()} -> {dates.max().date()} | {panel.shape[1]} columns")

    preds, scorecard = experiment.run(panel, horizon=HORIZON, model_names=MODELS)

    if scorecard.empty:
        print("No model produced predictions.")
        return

    out_dir = Path("results/smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    scorecard.to_csv(out_dir / "scorecard.csv", index=False)
    preds.to_parquet(out_dir / "predictions.parquet")

    cols = ["model", "n", "accuracy", "auc", "ic_mean", "ic_tstat",
            "ic_pvalue", "ic_positive_rate", "spread_mean", "n_dates"]
    present = [c for c in cols if c in scorecard.columns]
    print("\n===== Scorecard =====")
    print(scorecard[present].round(4).to_string(index=False))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
