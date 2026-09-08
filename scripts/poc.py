"""Proof of concept: direct directional classification vs the old approach.

Runs the full model set on a few tickers and prints the scoring table, so the
new pipeline's numbers can be compared against the legacy runner's output in
results/simple_smoke_long (AAPL / XGBoost / 35.3% directional accuracy).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data
from src.evaluate import run_experiment
from src.metrics import score_all

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")

TICKERS = ["AAPL", "MSFT", "^GSPC", "BTC-USD"]
MODELS = ["always_up", "base_rate", "random", "persistence", "logistic", "xgboost"]
TARGET_MODES = ["absolute", "excess"]
HORIZON = 5


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    all_preds = []
    for ticker in TICKERS:
        df = data.load(ticker, start="2010-01-01", end="2026-09-01")
        if df.empty:
            continue
        print(f"\n{ticker}: {len(df)} rows, {df.index.min().date()} -> {df.index.max().date()}")
        for mode in TARGET_MODES:
            preds = run_experiment(df, ticker, MODELS, horizon=HORIZON, target_mode=mode)
            if not preds.empty:
                all_preds.append(preds)

    if not all_preds:
        print("No predictions produced.")
        return

    preds = pd.concat(all_preds, ignore_index=True)
    out_dir = Path("results/poc")
    out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out_dir / "predictions.parquet")

    summary = score_all(preds)
    cols = [
        "ticker", "model", "target_mode", "n", "accuracy",
        "accuracy_ci_low", "accuracy_ci_high",
        "majority_baseline", "edge_vs_majority", "auc", "brier",
        "p_value_vs_majority",
    ]
    summary = summary[cols].round(4)
    summary.to_csv(out_dir / "summary.csv", index=False)

    for mode in TARGET_MODES:
        print(f"\n########## target = {mode} ##########")
        for ticker in summary["ticker"].unique():
            block = summary[(summary["ticker"] == ticker) & (summary["target_mode"] == mode)]
            if block.empty:
                continue
            print(f"\n===== {ticker} =====")
            print(block.drop(columns=["ticker", "target_mode"]).to_string(index=False))

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
