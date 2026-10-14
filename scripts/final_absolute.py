"""Final evaluation of absolute direction: will the price be higher after h days?

The same panel, features and protocol as the cross-sectional final run, with
the label replaced by the sign of the stock's own return. The bar to clear is
"always up": equities drift upward, so predicting a rise every time is right
more often than not. The ensemble averages its members' probabilities rather
than their ranks, since here the probability is the forecast.
"""

from __future__ import annotations

import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import experiment, store
from src.config import setup_logging

warnings.filterwarnings("ignore")
setup_logging("WARNING")

PROTOCOL = dict(refit_every=252, train_stride=5)
MODELS = ["random", "constant", "logistic", "lightgbm", "xgboost", "mlp", "lstm"]
ENSEMBLE = ["lightgbm", "xgboost", "mlp", "lstm"]


def direction_scores(preds: pd.DataFrame) -> dict:
    d = preds.dropna(subset=["y_true", "score"])
    y, p = d["y_true"].astype(int).values, np.clip(d["score"].values, 0, 1)
    up_share = y.mean()
    return {"accuracy": ((p > 0.5) == y).mean(), "always_up_accuracy": up_share,
            "edge_vs_always_up": ((p > 0.5) == y).mean() - up_share,
            "predicted_up_share": (p > 0.5).mean(),
            "auc": roc_auc_score(y, p), "brier": brier_score_loss(y, p),
            "brier_base_rate": brier_score_loss(y, np.full(len(y), up_share))}


def main() -> None:
    pd.set_option("display.width", 200)
    t0 = time.time()
    panel = experiment.load_sharadar_panel(horizons=[5, 20], seq_len=20)
    panel = panel.drop(columns=[c for c in panel.columns if c.startswith("alt_")])

    rows = []
    for h in (5, 20):
        kept = {}
        for model in MODELS:
            preds, sc = experiment.run(panel, horizon=h, model_names=[model],
                                       target_mode="absolute", **PROTOCOL)
            if sc.empty:
                continue
            metrics = {**sc.iloc[0].drop(["model", "horizon", "target"]).to_dict(),
                       **direction_scores(preds)}
            store.save_experiment("final_abs", model, h, "absolute", "point_in_time",
                                  {**PROTOCOL, "seq_len": 20}, preds, metrics)
            rows.append({"horizon": h, "model": model, **direction_scores(preds)})
            print(rows[-1], flush=True)
            if model in ENSEMBLE:
                kept[model] = preds[["score", "y_true"]]

        if set(ENSEMBLE) <= set(kept):
            first = next(iter(kept.values()))
            ens = first[["y_true"]].assign(score=pd.concat([kept[m]["score"] for m in ENSEMBLE],
                                                           axis=1).mean(axis=1))
            rows.append({"horizon": h, "model": "ensemble", **direction_scores(ens)})
            print(rows[-1], flush=True)
        del kept
        gc.collect()

    out = pd.DataFrame(rows)
    Path("results/final").mkdir(parents=True, exist_ok=True)
    out.to_csv("results/final/absolute_summary.csv", index=False)
    print("\n" + out.round(4).to_string(index=False))
    print(f"\nTotal {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
