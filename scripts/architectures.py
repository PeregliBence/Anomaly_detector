"""Model architecture comparison on the research universe.

Linear, two boosted-tree implementations, a feed-forward network and an LSTM,
against the published-strategy baselines, at the two horizons where the
horizon sweep found signal. Every learned model receives the same columns,
including 20 days of daily sequences, so differences come from the
architecture rather than from the information each model was given.

Models are refit annually, as Gu, Kelly and Xiu (2020) do, which keeps the
results comparable with theirs and the neural training time manageable.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import experiment
from src.config import setup_logging

warnings.filterwarnings("ignore")
setup_logging("WARNING")

HORIZONS = [5, 20]
MODELS = ["random", "cs_reversal", "cs_low_vol", "cs_momentum",
          "logistic", "lightgbm", "xgboost", "mlp", "lstm"]
PROTOCOL = dict(refit_every=252, train_stride=5)
SEQ_LEN = 20
OUT_DIR = Path("results/architectures")
COLS = ["horizon", "model", "ic_mean", "ic_tstat", "ic_tstat_folds", "ic_positive_rate",
        "spread_mean", "auc", "brier", "seconds"]


def main() -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    panel = experiment.load_sharadar_panel(horizons=HORIZONS, seq_len=SEQ_LEN)
    print(f"Panel: {len(panel):,} rows, {panel.shape[1]} columns ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for h in HORIZONS:
        for model in MODELS:
            t = time.time()
            preds, sc = experiment.run(panel, horizon=h, model_names=[model], **PROTOCOL)
            if sc.empty:
                continue
            sc["seconds"] = round(time.time() - t)
            rows.append(sc)
            preds.to_parquet(OUT_DIR / f"predictions_h{h}_{model}.parquet")
            pd.concat(rows).to_csv(OUT_DIR / "scorecard.csv", index=False)
            print(sc[[c for c in COLS if c in sc]].round(4).to_string(index=False, header=len(rows) == 1),
                  flush=True)

    combined = pd.concat(rows, ignore_index=True)
    print("\n===== IC by architecture =====")
    print(combined.pivot(index="model", columns="horizon", values="ic_mean").round(4).to_string())
    print("\n===== Fold-level t-stat =====")
    print(combined.pivot(index="model", columns="horizon", values="ic_tstat_folds").round(2).to_string())
    print(f"\nTotal {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
