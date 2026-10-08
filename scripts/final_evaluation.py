"""Final evaluation with the final feature set, recorded in the research store.

Two variants share one panel and one protocol: price and market features
only, and the same plus the alternative-data features. The gap between them
is what fundamentals, earnings, corporate events and insider trades add.

At 5 and 20 days every architecture runs; at 60 and 120 days the baselines
and LightGBM, for the horizon comparison. Annual refits, as in Gu, Kelly and
Xiu (2020). Run from a clean working tree so every record carries an exact
commit.
"""

from __future__ import annotations

import gc
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import experiment, store
from src.config import setup_logging
from src.metrics import rank_ic

warnings.filterwarnings("ignore")
setup_logging("WARNING")

PROTOCOL = dict(refit_every=252, train_stride=5)
SEQ_LEN = 20
BASELINES = ["random", "cs_reversal", "cs_low_vol", "cs_momentum"]
LEARNED = ["logistic", "lightgbm", "xgboost", "mlp", "lstm"]
ENSEMBLE = ["lightgbm", "xgboost", "mlp", "lstm"]

PLAN = {
    # name: (drop alternative features?, {horizon: models})
    "final": (True, {5: BASELINES + LEARNED, 20: BASELINES + LEARNED,
                     60: BASELINES + ["lightgbm"], 120: BASELINES + ["lightgbm"]}),
    "final_alt": (False, {5: LEARNED, 20: LEARNED, 60: ["lightgbm"], 120: ["lightgbm"]}),
}


def ensemble_ic(preds: dict[str, pd.DataFrame]) -> float:
    ranks = [preds[m]["score"].groupby(level="date").rank(pct=True) for m in ENSEMBLE if m in preds]
    if len(ranks) < len(ENSEMBLE):
        return float("nan")
    base = next(iter(preds.values()))[["fwd_return"]]
    return float(rank_ic(base.assign(score=pd.concat(ranks, axis=1).mean(axis=1))).mean())


def main() -> None:
    pd.set_option("display.width", 220)
    t0 = time.time()
    panel = experiment.load_sharadar_panel(horizons=[5, 20, 60, 120], seq_len=SEQ_LEN, alt_features=True)
    alt_cols = [c for c in panel.columns if c.startswith("alt_")]
    print(f"Panel: {len(panel):,} rows, {panel.shape[1]} columns, {len(alt_cols)} alternative "
          f"({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for name, (drop_alt, horizons) in PLAN.items():
        data = panel.drop(columns=alt_cols) if drop_alt else panel
        for h, models in horizons.items():
            kept = {}
            for model in models:
                t = time.time()
                preds, sc = experiment.run(data, horizon=h, model_names=[model], **PROTOCOL)
                if sc.empty:
                    continue
                metrics = sc.iloc[0].drop(["model", "horizon", "target"]).to_dict()
                store.save_experiment(name, model, h, "cross_sectional", "point_in_time",
                                      {**PROTOCOL, "seq_len": SEQ_LEN, "alt_features": not drop_alt},
                                      preds, metrics)
                rows.append({"variant": name, "horizon": h, "model": model,
                             "ic": metrics["ic_mean"], "t_folds": metrics.get("ic_tstat_folds"),
                             "minutes": round((time.time() - t) / 60, 1)})
                print(rows[-1], flush=True)
                if model in ENSEMBLE:
                    kept[model] = preds[["score", "fwd_return"]]
            if set(ENSEMBLE) <= set(kept):
                rows.append({"variant": name, "horizon": h, "model": "ensemble",
                             "ic": ensemble_ic(kept), "t_folds": None, "minutes": 0})
                print(rows[-1], flush=True)
            del kept
            gc.collect()
        del data

    summary = pd.DataFrame(rows)
    out = Path("results/final")
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv", index=False)
    print("\n===== IC: price/market only vs with alternative data =====")
    print(summary.pivot_table(index=["horizon", "model"], columns="variant", values="ic").round(4).to_string())
    print(f"\nTotal {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
