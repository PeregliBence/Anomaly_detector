"""Does cross-sectional signal grow with the width of the field?

The 50-name run found no signal. Two explanations fit: the cross-section is
too narrow for a per-date rank correlation to resolve anything, or large-cap
survivors are simply the wrong place to look. Both predict that widening the
field raises the momentum baseline's information coefficient.

This runs the same pipeline at several universe sizes and horizons. A rising
IC supports the breadth explanation and predicts that Sharadar's thousands of
names — including small caps and delisted securities — will help. A flat IC
means the problem lies elsewhere, which is worth knowing before paying.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import experiment, universe
from src.config import setup_logging

logger = setup_logging("WARNING")

SIZES = [30, 60, 110]
HORIZONS = [20, 60]
MODELS = ["random", "cs_momentum", "cs_reversal", "lightgbm"]


def main() -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    full = universe.default()
    rows = []

    for horizon in HORIZONS:
        for size in SIZES:
            tickers = full[:size]
            print(f"\n### {len(tickers)} instruments, horizon {horizon} ###", flush=True)

            panel = experiment.load_panel(
                tickers, horizon=horizon, start="2010-01-01", end="2026-09-01"
            )
            if panel.empty:
                continue

            _, scorecard = experiment.run(panel, horizon=horizon, model_names=MODELS)
            if scorecard.empty:
                continue

            scorecard["universe_size"] = len(tickers)
            rows.append(scorecard)

            cols = ["model", "ic_mean", "ic_tstat", "ic_positive_rate", "auc", "spread_mean"]
            print(scorecard[[c for c in cols if c in scorecard]].round(4).to_string(index=False),
                  flush=True)

    if not rows:
        print("No results.")
        return

    combined = pd.concat(rows, ignore_index=True)
    out_dir = Path("results/breadth")
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dir / "scorecard.csv", index=False)

    print("\n===== Momentum baseline IC by universe size =====")
    pivot = combined[combined["model"] == "cs_momentum"].pivot(
        index="universe_size", columns="horizon", values="ic_mean"
    )
    print(pivot.round(4).to_string())

    print("\n===== Momentum baseline t-stat by universe size =====")
    pivot_t = combined[combined["model"] == "cs_momentum"].pivot(
        index="universe_size", columns="horizon", values="ic_tstat"
    )
    print(pivot_t.round(3).to_string())
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
