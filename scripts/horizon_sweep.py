"""The horizon sweep on the survivorship-free universe.

This is the measurement the system's configuration is chosen from: which
horizon, if any, carries cross-sectional signal, and whether learned models
add anything over the published strategies used as baselines.

The panel is built once with every horizon's forward return; the horizons are
then run in order of importance so the most informative results land first.
Results are written as each horizon finishes, so an interrupted run still
leaves usable output.
"""

from __future__ import annotations

import gc
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import experiment
from src.config import setup_logging

warnings.filterwarnings("ignore", message="X does not have valid feature names")
logger = setup_logging("INFO")

HORIZONS = [20, 60, 5, 120]
MODELS = ["random", "cs_momentum", "cs_reversal", "cs_low_vol", "logistic", "lightgbm"]
START = "2005-01-01"
SIZE = 1000

WALK_FORWARD = dict(refit_every=63, train_stride=5)

OUT_DIR = Path("results/horizon_sweep")
SUMMARY_COLS = ["horizon", "model", "ic_mean", "ic_tstat", "ic_pvalue",
                "ic_positive_rate", "spread_mean", "auc", "n_dates"]


def main() -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    panel = experiment.load_sharadar_panel(horizons=HORIZONS, start=START, size=SIZE)
    print(f"\nPanel ready in {time.time() - t0:.0f}s: {len(panel):,} rows, "
          f"{panel.index.get_level_values('ticker').nunique()} names", flush=True)

    scorecards = []
    for horizon in HORIZONS:
        t = time.time()
        preds, scorecard = experiment.run(
            panel, horizon=horizon, model_names=MODELS, **WALK_FORWARD
        )
        if scorecard.empty:
            continue

        preds.to_parquet(OUT_DIR / f"predictions_h{horizon}.parquet")
        scorecards.append(scorecard)
        pd.concat(scorecards).to_csv(OUT_DIR / "scorecard.csv", index=False)

        print(f"\n===== horizon {horizon} ({(time.time() - t) / 60:.0f} min) =====", flush=True)
        print(scorecard[[c for c in SUMMARY_COLS if c in scorecard]].round(4).to_string(index=False),
              flush=True)

        del preds
        gc.collect()

    if not scorecards:
        print("No results.")
        return

    combined = pd.concat(scorecards, ignore_index=True)
    print("\n===== IC by model and horizon =====")
    print(combined.pivot(index="model", columns="horizon", values="ic_mean").round(4).to_string())
    print("\n===== IC t-stat (Newey-West) by model and horizon =====")
    print(combined.pivot(index="model", columns="horizon", values="ic_tstat").round(2).to_string())
    print(f"\nTotal {(time.time() - t0) / 60:.0f} min. Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
